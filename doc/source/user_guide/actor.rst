.. _actor:

=====
Actor
=====

Actors are self-contained computational entities that represent individual units of computation
within the framework. They encapsulate both state and behavior, and communicate through message
passing.

Xoscar supports both stateful and stateless actors. Stateful actors ensure thread safety for
concurrent systems while stateless actors can handle massive volumes of concurrent messages.

.. seealso::
   :ref:`ref_actor`


Define an actor
---------------

To define a stateful actor, your actor should inherit from the base class ``xoscar.Actor``. For
stateless actors, the inheritance should be from ``xoscar.StatelessActor``.

Two special methods are available for customization. The first method is invoked before the actor
is created, allowing you to set up any necessary initialization logic. The second method is called
after the actor is destroyed, providing an opportunity for cleanup or finalization tasks.

.. code-block:: python

   import xoscar as xo

   # a stateful actor.
   # to define a stateless actor, inherit from xo.StatelessActor.
   class MyActor(xo.Actor):
       def __init__(self, *args, **kwargs):
           pass
       async def __post_create__(self):
           # called after created
           pass
       async def __pre_destroy__(self):
           # called before destroy
           pass
       def method_a(self, arg_1, arg_2, **kw_1):  # user-defined function
           pass
       async def method_b(self, arg_1, arg_2, **kw_1):  # user-defined async function
           pass


Create an actor
---------------

To create an actor, you need to provide the address of the actor pool where you want the actor to
reside, along with a unique ID for the actor. Additionally, you need to provide any required
positional and keyword arguments during the actor's initialization.

.. code-block:: python

   actor_ref = await xo.create_actor(
       MyActor, 1, 2, a=1, b=2,
       address='<ip>:<port>', uid='UniqueActorName'
   )

Create a actor reference
------------------------

To create a reference to a specific actor, you need to provide both the ID of the actor and the
address of the actor pool in which the actor is located.

.. code-block:: python

   actor_ref = await xo.actor_ref(address, actor_id)

Check the existence of an actor
-------------------------------

To check the existence of an actor, you need to provide a reference to the actor.

.. code-block:: python

   await xo.has_actor(actor_ref)

Invoke an actor's method
------------------------

You can invoke an actor's method by its reference.

.. code-block:: python

   await actor_ref.method_a(1, 2, a=1, b=2)

Destroy an actor
----------------

You can destroy an actor and release corresponding resources by its reference.

.. code-block:: python

   await xo.destroy_actor(actor_ref)
