.. _oscar_usage:

Oscar APIs
==========

Actor Definition
----------------

.. code-block:: python

    import mars.oscar as mo

    # stateful actor, for stateless actor, inherit from mo.StatelessActor
    class MyActor(mo.Actor):
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


Creating Actors
---------------

.. code-block:: python

    import mars.oscar as mo

    actor_ref = await mo.create_actor(
        MyActor, 1, 2, a=1, b=2,
        address='<ip>:<port>', uid='UniqueActorName')


Destroying Actors
-----------------

.. code-block:: python

    import mars.oscar as mo

    await mo.destroy_actor(actor_ref)
    # or
    await actor_ref.destroy()


Checking Existence of Actors
----------------------------

.. code-block:: python

    import mars.oscar as mo

    await mo.has_actor(mo.ActorRef(worker_addr, actor_uid))


Getting Actor Reference
-----------------------

.. code-block:: python

    import mars.oscar as mo

    actor_ref = await mo.actor_ref(worker_addr, actor_id)


Calling Actor Method
--------------------

.. code-block:: python

    # send
    await actor_ref.method_a.send(1, 2, a=1, b=2)
    # equivalent to actor_ref.method_a.send
    await actor_ref.method_a(1, 2, a=1, b=2)
    # tell
    await actor_ref.method_a.tell(1, 2, a=1, b=2)

