.. _quickstart:

==========
Quickstart
==========

This concise introduction demonstrates how to estimate Pi in a parallel manner
using the Monte Carlo method with Xoscar.

We import :code:`xoscar` at the very beginning:

.. code-block:: python

   import asyncio
   import xoscar as xo


Create actor pools
------------------

To begin, we need to create actor pools, each of which will run within its own individual process.

.. seealso::
   :ref:`actor-pool`


.. code-block:: python

   dop = 4 # degree of parallelism
   loop = asyncio.get_event_loop()
   loop.run_until_complete(xo.create_actor_pool(address="localhost:9999", n_process=dop))


After successfully creating the actor pools, we gather the address of each pool for following
steps.

.. code-block:: python

   pool_config = await xo.get_pool_config("localhost:9999")
   pool_addresses = pool_config.get_external_addresses()

Define an actor
---------------

Next, we define an actor that will perform the estimation. This actor includes a method called
``estimate`` that takes the total number of points as input and returns the number of points inside
the circle. Since this actor doesn't have any internal state, it inherits from
``xo.StatelessActor`` to ensure lock-free execution.

.. seealso::
   :ref:`actor`

.. code-block:: python

   class MyActor(xo.StatelessActor):
       def estimate(self, n):
           import random
           from math import sqrt

           inside = 0
           for _ in range(n):
               x = random.uniform(-1, 1)
               y = random.uniform(-1, 1)
               if sqrt(x ** 2 + y ** 2) < 1:
                   inside += 1
           return inside

Create actors
-------------

Finally, we create an actor within each actor pool.

.. code-block:: python

   actors = []
   for i, address in enumerate(pool_addresses):
       actor = await xo.create_actor(
           MyActor,
           address=address,
           uid=str(i),
       )
       actors.append(actor)

Compute Pi
----------

Finally, we invoke the ``estimate`` method on each actor, leveraging the parallelism provided by Xoscar for efficient computation and distribution of the estimation task, and finally gather their individual outputs to calculate the value of Pi.

.. code-block:: python

   N = 10 ** 7
   tasks = []
   for actor in actors:
      tasks.append(actor.estimate(N))

   inside = sum(await asyncio.gather(*tasks))
   pi = 4 * inside / (len(actors) * N)
   print('pi: %.5f' % pi)
