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
