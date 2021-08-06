.. _batch_index:

Create and Use Batch Methods
============================

Oscar provides a set of APIs to write batch methods. You can simply add a
``@extensible`` decorator to your actor method and create a batch version.  All
calls wrapped in a batch will be sent together, reducing possible RPC cost.

Create a Batch Method
---------------------

You can create a batch method with a ``@extensible`` decorator:

.. code-block:: python

    import mars.oscar as mo

    class ExampleActor(mo.Actor):
        @mo.extensible
        async def batch_method(self, a, b=None):
            pass

Sometimes we need to process received batch requests. For instance, we need to
group requests by certain keys and resent them to different handlers in
batches. Oscar supports creating a batch version of the method:

.. code-block:: python

    class ExampleActor(mo.Actor):
        @mo.extensible
        async def batch_method(self, a, b=None):
            raise NotImplementedError  # this will redirect all requests to the batch version

        @batch_method.batch
        async def batch_method(self, args_list, kwargs_list):
            results = []
            for args, kwargs in zip(args_list, kwargs_list):
                a, b = self.batch_method.bind(*args, **kwargs)
                # process the request
                results.append(result)
            return results  # return a list of results

In the code above, we simply raises a ``NotImplementedError`` to let the batch
version handle all requests. The batch version have two arguments accepting
``args`` and ``kwargs`` of all batched calls as lists. To make argument
extraction easier, a utility function ``bind`` is added as an attribute of the
method which extracts ``args`` and ``kwargs`` into real arguments.

Call Batch Methods
------------------

Calling batch methods is easy. You can use ``<method_name>.delay`` to make a
batched call and use ``<method_name>.batch`` to send them:

.. code-block:: python

    ref = await mo.actor_ref(uid='ExampleActor', address='127.0.0.1:13425')
    results = await ref.batch_method.batch(
        ref.batch_method.delay(10, b=20),
        ref.batch_method.delay(20),
    )
