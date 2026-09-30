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

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux fork and epoll")
def test_forked_child_does_not_clean_up_parent_thread_caller():
    """A subprocess must not close a client owned by a surviving parent thread."""
    program = textwrap.dedent(
        """
        import gc
        import os
        import select
        import subprocess
        import sys
        import threading
        import time

        from xoscar.backends.core import ActorCaller, ActorCallerThreadLocal

        read_fd, write_fd = os.pipe()
        original_stop = ActorCallerThreadLocal.stop

        async def traced_stop(self):
            os.write(write_fd, f"{os.getpid()}\\n".encode())
            return await original_stop(self)

        ActorCallerThreadLocal.stop = traced_stop
        caller = ActorCaller()
        ready = threading.Event()
        release = threading.Event()

        def health_thread():
            caller.cancel_tasks()
            ready.set()
            release.wait()

        thread = threading.Thread(target=health_thread)
        thread.start()
        assert ready.wait(5)
        for _ in range(3):
            subprocess.run(
                [sys.executable, "-c", "pass"],
                preexec_fn=lambda: (gc.collect(), time.sleep(0.2)),
                check=True,
            )
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        gc.collect()
        assert select.select([read_fd], [], [], 5)[0]
        os.close(write_fd)
        stopped_in = [int(line) for line in os.read(read_fd, 4096).splitlines()]
        assert stopped_in, "parent thread caller was not cleaned up"
        assert all(pid == os.getpid() for pid in stopped_in), stopped_in
        """
    )
    subprocess.run([sys.executable, "-c", program], check=True, timeout=15)
