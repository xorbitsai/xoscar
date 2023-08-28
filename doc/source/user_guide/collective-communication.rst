.. _colletive-communication:

=========================
Collective communitcation
=========================

Collective communication is a global communication operation in which all processes in a process 
group participate.

xoscar supports collective communication among actors. It utilizes the Gloo backend on CPU and 
the NCCL backend on GPU. You can determine which backend to use by setting the parameter ``backend``
of function ``init_process_group`` when establishing the process group.

.. seealso::
   :ref:`ref_collective_communication`


Collective communication example
--------------------------------

To perform collective communication, you need to create an actor to invoke the relevant interfaces 
for collective communication. First, you need to initialize the process group. After initializing 
the process group, you can create smaller process groups within this overall process group for 
collective communication. Here is an example of how to perform a broadcast operation:

.. code-block:: python

   from xoscar import Actor, ActorRefType, actor_ref, create_actor_pool, get_pool_config
   from xoscar.context import get_context
   from xoscar.collective.common import(
      RANK_ADDRESS_ENV_KEY,
      RENDEZVOUS_MASTER_IP_ENV_KEY,
      RENDEZVOUS_MASTER_PORT_ENV_KEY,
   )
   from xoscar.collective.core import (
      RankActor,
      broadcast,
      init_process_group,
      new_group,
   )
   import os
   import numpy as np
   import asyncio

   class WorkerActor(Actor):
      def __init__(self, rank, world, *args, **kwargs):
         self._rank = rank
         self._world = world

      async def init_process_group(self):
         os.environ[RANK_ADDRESS_ENV_KEY] = self.address
         return await init_process_group(self._rank, self._world)

      async def test_broadcast(self):
         root = 1
         _group = [0, 1]
         sendbuf = np.zeros((2, 3), dtype=np.int64)
         if self._rank == _group[root]:
               sendbuf = sendbuf + self._rank
         recvbuf = np.zeros_like(sendbuf, dtype=np.int64)
         group = await new_group(_group)
         if group is not None:
               await broadcast(sendbuf, recvbuf, root=root, group_name=group)
         print(np.equal(recvbuf, np.zeros_like(recvbuf) + _group[root]))

   pool = await create_actor_pool(
      "127.0.0.1",
      n_process=2,
      envs=[
         {
               RENDEZVOUS_MASTER_IP_ENV_KEY: "127.0.0.1",
               RENDEZVOUS_MASTER_PORT_ENV_KEY: "25001",
         }
      ]
      * 2,
   )
   main_addr = pool.external_address
   config = (await get_pool_config(pool.external_address)).as_dict()
   all_addrs = list(config["mapping"].keys())
   all_addrs.remove(main_addr)

   async with pool:
      ctx = get_context()
      r0 = await ctx.create_actor(WorkerActor, 0, 2, address=all_addrs[0])
      r1 = await ctx.create_actor(WorkerActor, 1, 2, address=all_addrs[1])
      t0 = r0.init_process_group()
      t1 = r1.init_process_group()
      await asyncio.gather(*[t0, t1])

      t0 = r0.test_broadcast()
      t1 = r1.test_broadcast()
      await asyncio.gather(*[t0, t1])
