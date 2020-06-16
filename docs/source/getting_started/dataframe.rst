.. _getting_started_dataframe:

Mars DataFrame
==============

For a quick tour of Mars DataFrame, please visit :ref:`10min`.

Mars DataFrame can be initialized just like pandas DataFrame.

.. code-block:: python

   >>> import mars.dataframe as md
   >>> md.DataFrame({'a': [1, 2, 3], 'b': ['s1', 's2', 's3']})
   DataFrame <op=DataFrameDataSource, key=12ee87049f2f1125ffaa84e91f790249>

Pandas DataFrame can be passed to Mars DataFrame either.

.. code-block:: python

   >>> import pandas as pd
   >>> md.DataFrame(pd.DataFrame([[1, 2], [3, 4]]))
   DataFrame <op=DataFrameDataSource, key=853b0d99cd26ec82751524899172eb8c>

Creating Mars DataFrame from Mars tensor.

.. code-block:: python

   >>> md.DataFrame(mt.random.rand(3, 3))
   DataFrame <op=DataFrameFromTensor, key=10a421ed18adfa42cb649aa575a1d763>

Mars DataFrame can read data from CSV files, SQL tables, etc.

.. code-block:: python

   >>> md.read_csv('Downloads/ratings.csv')
   DataFrame <op=DataFrameReadCSV, key=48550937383cbea63d4f9f24f3eb1a17>

For more information about DataFrame creation, refer to :ref:`api.io`.

Like Mars tensor, DataFrame is lazy evaluated as well.
If you want to get result, ``.execute()`` needs to be called.

.. code-block:: python

   >>> df = md.read_csv('Downloads/ratings.csv')
   >>> grouped = df.groupby('movieId').agg({'rating': ['min', 'max', 'mean', 'std']})
   >>> grouped.execute()
           rating
              min  max      mean       std
   movieId
   1          0.5  5.0  3.921240  0.889012
   2          0.5  5.0  3.211977  0.951150
   3          0.5  5.0  3.151040  1.006642
   4          0.5  5.0  2.861393  1.095702
   5          0.5  5.0  3.064592  0.982140
   ...        ...  ...       ...       ...
   131254     4.0  4.0  4.000000       NaN
   131256     4.0  4.0  4.000000       NaN
   131258     2.5  2.5  2.500000       NaN
   131260     3.0  3.0  3.000000       NaN
   131262     4.0  4.0  4.000000       NaN

   [26744 rows x 4 columns]


Remember that ``DataFrame.execute()`` will return DataFrame itself.

For more implemented DataFrame API, refer to :ref:`DataFrame API reference <dataframe_api>`.

In order to convert Mars DataFrame to pandas, ``.execute().fetch()`` can be
called.  An alternative is ``.to_pandas()``.

.. code-block:: python

   >>> type(grouped.execute())
   mars.dataframe.core.DataFrame

   >>> type(grouped.execute().fetch())
   pandas.core.frame.DataFrame

   >>> type(grouped.to_pandas())
   pandas.core.frame.DataFrame

   >>> grouped.to_pandas()
           rating
              min  max      mean       std
   movieId
   1          0.5  5.0  3.921240  0.889012
   2          0.5  5.0  3.211977  0.951150
   3          0.5  5.0  3.151040  1.006642
   4          0.5  5.0  2.861393  1.095702
   5          0.5  5.0  3.064592  0.982140
   ...        ...  ...       ...       ...
   131254     4.0  4.0  4.000000       NaN
   131256     4.0  4.0  4.000000       NaN
   131258     2.5  2.5  2.500000       NaN
   131260     3.0  3.0  3.000000       NaN
   131262     4.0  4.0  4.000000       NaN

   [26744 rows x 4 columns]


.. note::

    Users should always consider using ``.execute()`` instead of ``.to_pandas()``,
    because when the DataFrame is large,
    ``.execute()`` will only fetch head and tail rows for display purpose.
    On the other hand, ``.to_pandas()`` will try to generate
    the entire DataFrame on the server side and return it back to client,
    which is extremely inefficient and may cause OutOfMemory error.

If multiple DataFrames need to be executed together,
:class:`mars.dataframe.ExecutableTuple` could be used.

.. code-block:: python

   >>> df = md.DataFrame(mt.random.rand(3, 3))

   >>> md.ExecutableTuple([df, df.sum()]).execute()
   (          0         1         2
    0  0.604443  0.743964  0.281236
    1  0.778034  0.634661  0.237829
    2  0.886275  0.456751  0.340311,
    0    2.268752
    1    1.835377
    2    0.859376
    dtype: float64)

DataFrame can be saved to CSV etc.

.. code-block:: python

   >>> df.to_csv('Downloads/grouped.csv').execute()
   Empty DataFrame
   Columns: []
   Index: []

Refer to :ref:`api.dataframe.io` for more information.
