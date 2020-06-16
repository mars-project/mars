.. _lazy_evaluation:

Lazy Evaluation
===============

You need to call ``.execute()`` on Mars tensors, DataFrames and remote functions
to trigger execution.

.. code-block:: python

   >>> import mars.tensor as mt
   >>> import mars.dataframe as md
   >>> df = md.DataFrame(mt.random.rand(3, 3))
   >>> df
   DataFrame <op=DataFrameFromTensor, key=182b756be8a9f15c937a04223f11ffba>
   >>> df.execute()
             0         1         2
   0  0.167771  0.568741  0.877450
   1  0.037518  0.796745  0.072169
   2  0.052900  0.936048  0.307194

Calling ``.execute()`` will return Mars object itself, ``.fetch()`` could be called
on executed objects to get the result.

.. code-block:: python

   >>> import mars.remote as mr
   >>> f = mr.spawn(lambda x: x + 1, args=(10,))
   >>> f.execute()
   Object <op=RemoteFunction, key=8a9ef53cb93cd7698d71512ec813682b>
   >>> f.fetch()
   11

However, there are exceptions that some functions will trigger execution
intermediately.

- Iterating over DataFrame, including :meth:`mars.dataframe.DataFrame.iterrows` and
  :meth:`mars.dataframe.DataFrame.itertuples`.
- All plot functions for DataFrame and Series, including :meth:`mars.dataframe.DataFrame.plot`,
  :meth:`mars.dataframe.DataFrame.plot.bar` and so forth.
- All functions in Mars learn like ``fit``, ``predict`` and so forth.
