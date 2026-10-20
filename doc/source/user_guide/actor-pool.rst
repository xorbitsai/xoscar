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


Switch managers without restarting actors
------------------------------------------

By default, actor pools keep the existing process hierarchy: stopping
the main pool stops its sub pools, and sub pools exit if their main process dies.
For resources that must survive a manager replacement, use
``enable_manager_handoff=True``. Xoscar starts an independent local host process and
returns a ``HandoffActorPool`` management handle. The host owns the actual
main pool, actor registry, sub pools, and automatic recovery. Replacing the
manager transfers ownership of that host; actors keep their PIDs, addresses,
in-memory state, and active calls or generators.

.. code-block:: python

   import xoscar as xo

   pool = await xo.create_actor_pool(
       "127.0.0.1",
       n_process=1,
       enable_manager_handoff=True,
       lease_timeout=15,
       orphan_timeout=300,
   )
   # Use the normal actor APIs with pool.external_address.
   actor = await xo.create_actor(
       MyActor, uid="model", address=pool.external_address,
       allocate_strategy=xo.allocate_strategy.RandomSubPool(),
   )

   # Hand over management to the replacement process.
   credential = await pool.handoff()
   replacement = await xo.take_over_actor_pool(credential)
   actors = await replacement.list_actors()
   same_actor = await xo.actor_ref(address=replacement.external_address, uid="model")

   # Explicitly stop the host and all its sub pools when no longer needed.
   await replacement.stop()

``handoff()`` releases the current owner's lease and returns a one-use credential.
It closes the old local management handle without stopping the host. The next
manager can take over immediately; two managers racing to take over cannot both win.
Each successful takeover rotates the credential, so the old owner cannot renew
its lease or stop the host through an obsolete handle.

The credential's ``to_dict()`` result is JSON-compatible and may be passed to
``take_over_actor_pool()`` in another process on the same machine. Treat the token
as a private management credential: store it in a file restricted to its owner
when transferring it through the filesystem. These credentials prevent stale
management operations; they do not replace transport authentication or restrict
the normal actor APIs available to trusted Xoscar clients.

The management handle renews its lease in the background on its owning event
loop. Before an unexpected manager failure, ``pool.handoff_handle`` can be saved for
recovery. Once the old lease expires, a replacement can take over using that saved
credential. An expired lease cannot be revived by a late heartbeat. Actors
continue serving during the handoff window; if no manager takes over within
``orphan_timeout`` after handoff or lease expiration, the host stops and reclaims
its sub pools. Keep the owner's event loop responsive, and explicitly stop or
hand off a pool before closing its handle. A handed-off pool also survives normal manager
event-loop shutdown.

``HandoffActorPool`` provides ``external_address``, ``pid``, ``handoff_handle``,
``get_config()``, ``list_actors()``, ``append_sub_pool()``, ``remove_sub_pool()``,
``handoff()``, ``join()`` and ``stop()``. Its async context manager stops the host
on exit. It is a remote management handle, so code that accesses private
``MainActorPool`` attributes must use the public methods instead.

Manager handoff currently requires a native TCP address and local, same-user
management. Taking over checks the host's process identity, handoff protocol, and
Python major/minor version. The receiving environment must also provide
compatible Xoscar serialization and actor interfaces. A handoff preserves the
host's original Python environment and running actor code; upgrading that code
or its dependencies requires creating a replacement pool. Xoscar does not
migrate actor memory, GPU allocations, or in-flight calls to a different host.
Applications decide compatibility, readiness, request draining, and traffic
switching when replacing their resource pools.


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
