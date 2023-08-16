# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# We need to extend cupy's inner class because an actor is a daemonic processes
# which are not allowed to have children. However, the origin code in cupy
# will create children processes.

import queue
import socket
import threading
from ctypes import sizeof

from ...tests.core import lazy_import

cupy = lazy_import("cupy")

if cupy is not None:
    import cupyx.distributed
    from cupy.cuda import nccl
    from cupyx.distributed import _klv_utils, _store, _store_actions

    class ExceptionAwareThreading(threading.Thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._exception = None
            self.q = queue.Queue()

        def run(self):
            try:
                super().run()
                self.q.put(None)
            except Exception as e:
                self.q.put(e)

        def join(self):
            super().join()
            if not self.q.empty():
                exception = self.q.get()
                if exception is not None:
                    raise exception

    class TCPStore:
        # This is only used for initialization of nccl so we don't care
        # too much about performance
        def __init__(self, world_size):
            self.storage = {}
            self._thread = None
            self._world_size = world_size
            self._run = 1
            # For implementing a barrier
            self._lock = threading.Lock()
            self._current_barrier = None

        def __del__(self):
            if not _store._exit_mode:
                self.stop()

        def _thread_request(self, c_socket):
            with c_socket:
                # Receive in KLV format
                action_bytes = c_socket.recv(sizeof(_klv_utils.action_t))
                if len(action_bytes) > 0:
                    action_m = _klv_utils.action_t.from_buffer_copy(action_bytes)
                    if action_m.length > 256:
                        raise ValueError("Invalid length for message")
                    value = bytearray(action_m.value)[: action_m.length]
                    r = _store_actions.execute_action(action_m.action, value, self)
                    if r is not None:
                        c_socket.sendall(r.klv())

        def _server_loop(self, host, port):
            # This is for minimum info exchange during initialization
            # a single connection allows to implement locking mechanics easily
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                s.listen()
                s.settimeout(0.5)
                while self._run == 1:
                    try:
                        c_socket, addr = s.accept()
                    except socket.timeout:
                        continue

                    t = threading.Thread(
                        target=self._thread_request, args=(c_socket,), daemon=True
                    )
                    t.start()

        def run(self, host=_store._DEFAULT_HOST, port=_store._DEFAULT_PORT):
            # Run the TCP store in a different process
            t = ExceptionAwareThreading(target=self._server_loop, args=(host, port))
            t.start()
            self._thread = t

        def stop(self):
            if _store._exit_mode:
                return  # Prevent shutdown errors
            if self._thread is not None:
                # acquire the lock
                self._lock.acquire()
                self._run = 0
                self._lock.release()
                self._thread.join()

    class XoscarNCCLBackend(cupyx.distributed.NCCLBackend):
        """Interface that uses NVIDIA's NCCL to perform communications.

        Args:
            n_devices (int): Total number of devices that will be used in the
                distributed execution.
            rank (int): Unique id of the GPU that the communicator is associated to
                its value needs to be `0 <= rank < n_devices`.
            host (str, optional): host address for the process rendezvous on
                initialization. Defaults to `"127.0.0.1"`.
            port (int, optional): port used for the process rendezvous on
                initialization. Defaults to `13333`.
            use_mpi(bool, optional): switch between MPI and use the included TCP
                server for initialization & synchronization. Defaults to `False`.
        """

        def __init__(
            self,
            n_devices,
            rank,
            tcpstore,
            host=_store._DEFAULT_HOST,
            port=_store._DEFAULT_PORT,
            use_mpi=False,
        ):
            self._tcpstore = tcpstore
            super().__init__(n_devices, rank, host, port, use_mpi)

        def _init_with_tcp_store(self, n_devices, rank, host, port):
            nccl_id = None
            if rank == 0:
                self._tcpstore.run(host, port)
                nccl_id = nccl.get_unique_id()
                # get_unique_id return negative values due to cython issues
                # with bytes && c strings. We shift them by 128 to
                # make them positive and send them as bytes to the proxy store
                shifted_nccl_id = bytes([b + 128 for b in nccl_id])
                self._store_proxy["nccl_id"] = shifted_nccl_id
                self._store_proxy.barrier()
            else:
                self._store_proxy.barrier()
                nccl_id = self._store_proxy["nccl_id"]
                nccl_id = tuple([int(b) - 128 for b in nccl_id])
            self._comm = nccl.NcclCommunicator(n_devices, nccl_id, rank)
