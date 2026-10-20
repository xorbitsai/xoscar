.. _actor-pool:

==========
Actor pool
==========

An actor pool serves as a container and entry point for managing actors. It is also a
self-contained computational unit that in most cases runs within an individual process.

Before creating any actor, it is necessary to initialize the actor pools. In scenarios involving
multiple machines, it is recommended to initialize an actor pool on each machine to effectively
utilize the resources of the entire cluster.

Manual creation of each actor pool is not required. Instead, you can specify the desired number of
actor pools using the ``n_process`` parameter when invoking ``xoscar.create_actor_pool``. Xoscar
will automatically handle the creation of the specified number of actor pools for you. Normally,
``n_process`` should be set to the number of CPUs.

.. seealso::
   :ref:`ref_actor-pool`


Create an actor pool
--------------------

To create an actor pool, you are required to provide the address and specify the desired level of
parallelism.

.. code-block:: python

   import asyncio
   import xoscar as xo

   async def _main():
       await xo.create_actor_pool(address="localhost:9999", n_process=4)

   loop = asyncio.get_event_loop()
   loop.run_until_complete(_main())


Replace a main process without restarting sub pools
---------------------------------------------------

An ordinary native actor pool can hand its existing sub pools to another main
pool on the same machine. There is no creation-time option or extra host
process. Sub pools keep their PIDs, addresses, actors, in-memory state, and
active calls or generators. The replacement main runs in the new application's
process and Python environment.

Start a normal main pool in the replacement process with ``n_process=0``. Create
any application control actors there and check that the replacement is ready.
Then, in the old process, hand over its sub pools:

.. code-block:: python

   # Old process: created normally, with no handoff option.
   old_pool = await xo.create_actor_pool("127.0.0.1:9000", n_process=2)

   # Replacement process: starts its own main pool without loading actors again.
   new_pool = await xo.create_actor_pool("127.0.0.1:9001", n_process=0)

   # Old process: transfer the running sub pools to the new main.
   addresses = await old_pool.handoff_to("127.0.0.1:9001")
   await old_pool.stop()
   # The old process may now exit. The same sub pools are managed by new_pool.

   # Replacement process: find an existing actor through the new main.
   actor = await xo.actor_ref(address=new_pool.external_address, uid="model")
   # Normal shutdown still stops all sub pools owned by this main.
   await new_pool.stop()

The example's old-process and replacement-process blocks run in separate
processes; ``new_pool`` is a normal local ``MainActorPool``, not a remote handle.
Applications arrange readiness, pass the replacement address to the old
process, recreate main-pool actors, and switch their API traffic.

During handoff Xoscar briefly rejects actor creation/destruction and sub-pool
append/removal; ordinary actor calls keep running. It transfers the actor
registry and pool configuration, changes the children's main address and
process watchers, and removes the old main's cleanup and recovery authority.
The new main performs normal monitoring, recovery, append/removal, and shutdown
for the adopted processes. Stopping the old main afterwards does not stop them.

Failures before the commit decision roll back to the old main. If restoration
cannot be confirmed for a live child, lifecycle changes remain blocked; retry
with the same target to finish rollback, or stop the old pool. After the commit
decision, retry ``old_pool.handoff_to()`` with the same address to finish a lost
acknowledgement. Cancelling the coroutine waits for commit or rollback before
propagating cancellation. The ``timeout`` argument bounds preparation (30 seconds
by default); it does not revoke a committed transfer. Ordinary stop and main
process failure retain their existing cleanup behavior outside a handoff.
A prepared replacement also checks abandoned handoffs: it completes a fully
staged transfer whose children already switched owners, or reclaims an
incomplete transfer if the source has exited.

Handoff currently supports POSIX native subprocess pools. Windows rejects the
operation before making changes; its job-object ownership needs separate
support. Both mains must use the same Python major/minor and compatible Xoscar
serialization and actor interfaces, with matching transport and proxy settings.
After a handoff, children watch their new main in userspace rather than relying
on Linux's binding to their original OS parent. This is a planned upgrade
operation, not a guarantee of preserving actors after an arbitrary main crash
or a crash midway through handoff.

Only sub-pool actors are retained. Actors in the old main stay there and must be
recreated by the application. The retained sub pools keep their original Python
environment and actor code; upgrading that code or model engines is separate
application work. Existing actor references pointing directly at sub pools stay
valid. Connections and references to the old main address must be redirected or
recreated; Xoscar does not migrate those connections or move GPU memory.


Trust and transport authentication
----------------------------------

Actor pools exchange Python objects, including executable Python definitions.
Every process permitted to send actor messages must therefore be trusted to run
code in the receiving process. Listening on a network interface is needed for
distributed operation; deployment administrators are responsible for restricting
actor ports to their intended cluster peers. Authentication on an application's
HTTP API does not protect Xoscar's actor transport.

By default Xoscar retains its existing, unauthenticated transport for compatible
deployments on trusted networks. Upgrading alone does not authenticate this mode
or make an actor port suitable for access by untrusted clients.

To enable message authentication, set the same nonempty ``XOSCAR_AUTH_KEY`` in
every supervisor, worker, sub-pool and external actor client's environment before
starting the processes. Generate one random deployment secret, for example with
``python -c "import secrets; print(secrets.token_hex(32))"``, and distribute it
through your deployment's secret management. Do not generate a separate key on
each node or transmit the key in an actor message.

With a key configured, both the Socket and UCX transports authenticate framing
metadata and the complete message, including every payload buffer, with
HMAC-SHA256 before deserializing any Python object. Keyed receivers reject legacy
unauthenticated messages and invalid signatures. An unkeyed receiver rejects
authenticated messages. All communicating processes must be upgraded and
configured together; there is no automatic downgrade or mixed-mode rolling
upgrade. Changing the key also requires restarting all communicating processes.

This mechanism authenticates messages; it does not encrypt traffic or establish
a session that prevents replay of captured messages. Use an authenticated,
encrypted tunnel when those properties are required, and retain the deployment's
network access controls. Anyone holding the shared key is a trusted cluster peer
with Python execution privileges.

Authenticated GPU transfers copy device buffers to host memory for hashing,
including on UCX. This adds CPU work and host staging compared with the default
UCX zero-copy path. Authenticated frames limit their fixed-width framing metadata
to 16 MiB and their serialized Python header to 64 MiB; payload size is determined
by the authenticated metadata. The legacy unkeyed wire format is unchanged.
