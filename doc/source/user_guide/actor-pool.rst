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
