.. _remote:

==========
User Guide
==========

.. Note:: New in version 0.4.1

Mars remote API provides a simple yet powerful way to run Python functions in
parallel.

The main API for Mars remote is :meth:`mars.remote.spawn`. It returns a Mars
Object while no execution happens yet. When ``.execute()`` is called, spawned
function will be submitted to Mars for execution, thus if multiple spawned
functions are executed together, they may run in parallel.

.. code-block:: python

   >>> import mars.remote as mr
   >>> def inc(x):
   >>>     return x + 1
   >>>
   >>> result = mr.spawn(inc, args=(0,))
   >>> result
   Object <op=RemoteFunction, key=e0b31261d70dd9b1e00da469666d72d9>
   >>> result.execute().fetch()
   1

List of spawned functions can be converted to
:class:`mars.remote.ExecutableTuple`, and ``.execute()`` can be called to run
these functions together.

.. code-block:: python

   >>> results = [mr.spawn(inc, args=(i,)) for i in range(10)]
   >>> mr.ExecutableTuple(results).execute().fetch()
   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Mars Objects returned by :meth:`mars.remote.spawn` can be treated
as arguments for other spawn functions.

.. code-block:: python

   >>> results = [mr.spawn(inc, args=(i,)) for i in range(10)]   # list of spawned functions
   >>> def sum_all(xs):
           return sum(xs)
   >>> mr.spawn(sum_all, args=(results,)).execute().fetch()
   55

Mars ensures that ``sum_all`` can be called only when the previous 10 ``inc``
called are finished.  Users need not to worry about the data of dependency,
e.g.  when ``sum_all`` is called, the argument ``xs`` has already been replaced by
real outputs of the previous ``inc`` functions.

For the distributed setting, 10 ``inc`` function may be distributed to different
workers. Users need not to care about how the functions are distributed, as
well as how the outputs of spawned functions are moved between workers.

User can also spawn new functions inside a spawned function.

.. code-block:: python

   >>> def driver():
   >>>     results = [mr.spawn(inc, args=(i,)) for i in range(10)]
   >>>     return mr.ExecutableTuple(results).execute().fetch()
   >>>
   >>> mr.spawn(driver).execute().fetch()
   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Mars tensor, DataFrame and so forth is available in spawned functions as well.

.. code-block:: python

   >>> import mars.tensor as mt
   >>> def driver2():
   >>>     t = mt.random.rand(10, 10)
   >>>     return t.sum().to_numpy()
   >>>
   >>> mr.spawn(driver2).execute().fetch()
   52.47844223908132

The argument ``n_output`` can indicate the number of outputs the spawned function
will return. This is important when different outputs are passed to different
functions.

.. code-block:: python

   >>> def triage(alist):
   >>>     ret = [], []
   >>>     for i in alist:
   >>>         if i < 0.5:
   >>>             ret[0].append(i)
   >>>         else:
   >>>             ret[1].append(i)
   >>>     return ret
   >>>
   >>> def sum_all(xs):
   >>>     return sum(xs)
   >>>
   >>> l = [0.4, 0.7, 0.2, 0.8]
   >>> la, lb = mr.spawn(triage, args=(l,), n_output=2)
   >>>
   >>> sa = mr.spawn(sum_all, args=(la,))
   >>> sb = mr.spawn(sum_all, args=(lb,))
   >>> mr.ExecutableTuple([sa, sb]).execute().fetch()
   >>> [0.6000000000000001, 1.5]

