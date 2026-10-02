NIXL buffer transfers
=====================

Xoscar optionally supports NIXL for writing CPU and CUDA buffers into buffers
allocated by another actor process. Actor RPC and transfer metadata use TCP;
``copy_to`` moves the buffer contents through NIXL's UCX backend. Ordinary
actor arguments and return values still use socket serialization. NCCL
collectives are independent of this option.

Installation and configuration
------------------------------

NIXL requires Linux. Install the optional dependency on each worker:

.. code-block:: bash

   pip install 'xoscar[nixl]'

The supported NIXL range is ``>=1.1,<2``. CPU and CUDA transfers were tested
with NIXL 1.1.0 and 1.5.x, using their UCX plugins. Install CuPy separately if
your application uses CuPy or RMM buffers. GPU peer access and cross-host RDMA
depend on the CUDA/UCX installation and hardware.

Select ``nixl`` for the external addresses of the participating worker
processes. For example, with two GPU workers:

.. code-block:: python

   import xoscar as xo

   pool = await xo.create_actor_pool(
       "127.0.0.1",
       n_process=2,
       external_address_schemes=[None, "nixl", "nixl"],
       subprocess_start_method="spawn",
   )

The first entry belongs to the main pool. Place the actors in the two worker
processes with ``ProcessIndex(1)`` and ``ProcessIndex(2)`` and select their CUDA
devices inside those processes. Do not initialize CUDA before forking workers.

The buffer API stays the same:

.. code-block:: python

   # On the receiving actor; keep the allocation alive.
   self.buffer = torch.empty(size, dtype=torch.uint8, device="cuda:1")
   return xo.buffer_ref(self.address, self.buffer)

   # On the sending actor, after obtaining the reference through actor RPC.
   await xo.copy_to([local_buffer], [remote_ref])

The NIXL route is also used between different worker processes in one pool,
where ordinary actor RPC may use a Unix socket. Copies inside one process
retain the existing local-copy behavior.

Buffer and completion semantics
-------------------------------

* Source and destination buffers must be C-contiguous and have equal byte
  sizes. NumPy/buffer-protocol objects, PyTorch CPU/CUDA tensors and CuPy/RMM
  CUDA buffers are supported. Read-only destinations are rejected. Empty
  buffers are allowed. Copies preserve bytes, without dtype conversion.
* CPU and GPU buffers can be mixed in a batch. Xoscar groups transfers by
  source and destination memory type.
* ``await copy_to(...)`` waits for data transfer and a target acknowledgement.
  CUDA devices are synchronized before transfer and before acknowledging the
  destination. This covers non-default streams but introduces a device-wide
  fence; stream-ordered overlap is not implemented. CUDA synchronization and
  memory registration/deregistration currently run synchronously on the pool's
  event loop, so long-running kernels or registration calls can delay other
  actor RPCs. Agent initialization runs in a worker thread.
* A copy cancelled while queued returns without starting a transfer.
  Cancelling an active copy waits for the transfer and acknowledgement before
  propagating cancellation. A cancellation request is not a guarantee of
  immediate return. A failed copy may have modified some destination bytes;
  it is not an atomic transaction and is not retried automatically.
  Completion polling backs off to 1 ms. There is no transfer deadline: a stalled
  backend that never reports a terminal state can delay cancellation and channel
  shutdown. The process must be terminated if the backend cannot recover.
* Registrations and metadata are reused for the most recent buffer batch on
  each connection. A batch of at most 256 MiB is retained until replacement
  or connection close; larger batches are deregistered after each copy.
  This retains strong references to the cached allocations; a view can keep
  its larger backing allocation alive. Do not resize or
  change the storage of a registered buffer. Concurrent copies on a connection
  are serialized so its registrations cannot be replaced during a transfer.
* ``block_size`` controls the existing socket fallback. It does not split a
  NIXL transfer. File-object transfers retain their existing implementation.
  Proxy-routed ``copy_to`` is not supported.

Benchmarking
------------

Run the benchmark from the source checkout, once per backend:

.. code-block:: bash

   UCX_TLS=tcp,cuda_copy,cuda_ipc python python/benchmarks/benchmark_copy_to.py --backend nixl
   UCX_TLS=tcp,cuda_copy,cuda_ipc python python/benchmarks/benchmark_copy_to.py --backend ucx

The benchmark allocates buffers on two GPUs in separate actor processes,
checks the destination data, and reports cold latency and the median of
repeated copies in both directions. Registration, metadata, control messages,
and synchronization are part of the measured ``copy_to`` path. Allocation
and data validation are outside the timing. Run each backend in a fresh
process: NIXL and UCXX distributions may bundle different UCX libraries.

NIXL is opt-in: performance depends on buffer sizes, batching, registration
reuse and the transport configuration. A faster batched transfer does not
imply higher throughput for a single large buffer.

Client-only processes
---------------------

API/frontend processes that only call remote actors do not need to install a
Router manually. Xoscar initializes and retains the default Router on first
use, so subsequent RPCs reuse its connection cache. Starting an actor pool
later adds its routes to that same Router.
