.. _getting_started:

Mars leverages parallel and distributed technology to accelerate numpy, pandas,
scikit-learn and Python functions.

There are four main APIs in Mars:

1. Mars tensor, which mimics numpy API and provide ability to process large tensors/ndarrays.
2. Mars DataFrame, which mimics pandas API and be able to process large DataFrames.
3. Mars learn, which mimics scikit-learn API and scales machine learning algorithms.
4. Mars Remote, which provide the ability to execute Python functions in parallel.


Mars Tensor
===========

Mars tensors can be created from numpy ndarrays or external files.

Creating a Mars tensor from numpy ndarray.

.. code-block:: python

   >>> import mars.tensor as mt
   >>> import numpy as np
   >>> t = mt.tensor(np.random.rand(4, 4))

Reading a HDF5 file into a Mars tensor.

.. code-block:: python

   >>> import mars.tensor as mt
   >>> t = mt.from_hdf5('t.hdf5', dataset='t')


Refer to :ref:`tensor_creation` and :ref:`tensor_read` for more information.

The usage of Mars tensor is very similar to numpy except that Mars tensor is
lazy evaluated.  You need to call `.execute()` to get the final result.

Remember that `.execute()` will return Mars tensor itself.

.. code-block:: python

   >>> (t - (t + 1).sum()).execute()
   array([[-23.06773811, -22.86112123, -23.03988405, -22.48884341],
          [-22.54959727, -22.13498645, -22.97627675, -23.09852276],
          [-23.11085224, -22.63999173, -22.27187961, -22.34163038],
          [-22.40633932, -22.17864095, -23.04577731, -22.76189835]])

For more implemented tensor API, refer to :ref:`tensor_routines`.

Once a tensor is executed, `.fetch()` could be called to get the result as
numpy ndarray.  A shortcut `.to_numpy()` is identical to `.execute().fetch()`.

.. code-block:: python

   >>> t.to_numpy()
   array([[0.06386055, 0.27047743, 0.09171461, 0.64275525],
          [0.5820014 , 0.99661221, 0.15532191, 0.0330759 ],
          [0.02074642, 0.49160693, 0.85971905, 0.78996828],
          [0.72525934, 0.95295771, 0.08582136, 0.36970032]])

   >>> type(t.execute())
   mars.tensor.core.Tensor

   >>> type(t.execute().fetch())
   numpy.ndarray

   >>> t.execute().fetch()
   array([[0.06386055, 0.27047743, 0.09171461, 0.64275525],
          [0.5820014 , 0.99661221, 0.15532191, 0.0330759 ],
          [0.02074642, 0.49160693, 0.85971905, 0.78996828],
          [0.72525934, 0.95295771, 0.08582136, 0.36970032]])

.. note::

    Users should always consider using `.execute()` instead of `.to_numpy()`,
    because when the tensor is large, `.execute()` will only fetch the edge items
    for display purpose. On the other hand, `.to_numpy()` will try to generate
    the entire array on the server side and return it back to client,
    which is extremely inefficient and may cause OutOfMemory error.

If multiple tensors need to be executed together,
:class:`mars.tensor.ExecutableTuple` could be used.

.. code-block:: python

   >>> import mars.tensor as mt
   >>> z = mt.zeros((3, 3))
   >>> t = mt.ones((3, 3))
   >>> mt.ExecutableTuple([z, t]).execute()
   (array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]]),
    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]]))

Tensors can be saved to external files, for instance, HDF5.

.. code-block:: python

   >>> import mars.tensor as mt
   >>> mt.to_hdf5('my.hdf5', mt.random.rand(3, 3), dataset='r').execute()
   array([], shape=(0, 0), dtype=float64)

Refer to :ref:`tensor_write` for more information about saving to external
files.


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
If you want to get result, `.execute()` needs to be called.

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


Remember that `DataFrame.execute()` will return DataFrame itself.

For more implemented DataFrame API, refer to :ref:`api`.

In order to convert Mars DataFrame to pandas, `.execute().fetch()` can be
called.  An alternative is `.to_pandas()`.

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

    Users should always consider using `.execute()` instead of `.to_pandas()`,
    because when the DataFrame is large,
    `.execute()` will only fetch head and tail rows for display purpose.
    On the other hand, `.to_pandas()` will try to generate
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


Mars Learn
==========

Mars learn mimics scikit-learn API and leverages the ability of Mars tensor and
DataFrame to process large data and execute in parallel.

Mars does not require installation of scikit-learn, but if you want to use Mars
learn, make sure scikit-learn is installed.

Install scikit-learn via:

.. code-block:: bash

   pip install scikit-learn

Refer to `installing scikit-learn <https://scikit-learn.org/stable/install.html>`_
for more information.

Let's take :class:`mars.learn.neighbors.NearestNeighbors` as an example.

.. code-block:: python

   >>> import mars.tensor as mt
   >>> from mars.learn.neighbors import NearestNeighbors
   >>> data = mt.random.rand(100, 3)
   >>> nn = NearestNeighbors(n_neighbors=3)
   >>> nn.fit(data)
   NearestNeighbors(algorithm='auto', leaf_size=30, metric='minkowski',
                    metric_params=None, n_neighbors=3, p=2, radius=1.0)
   >>> neighbors = nn.kneighbors(df)
   >>> neighbors
   (array([[0.0560703 , 0.1836808 , 0.19055679],
           [0.07100642, 0.08550266, 0.10617568],
           [0.13348483, 0.16597596, 0.20287617]]),
    array([[91, 10, 29],
           [68, 77, 29],
           [63, 82, 21]]))

Remember that functions like `fit`, `predict` will trigger execution instantly.
In the above example, `fit` and `kneighbors` will trigger execution internally.

For implemented learn API, refer to :ref:`api.learn`.

Mars learn can integrate with XGBoost, LightGBM, TensorFlow and PyTorch.

- For XGBoost, refer to :ref:`xgboost`.
- For LightGBM, refer to :ref:`lightgbm`.
- For TensorFlow, refer to :ref:`tensorflow`.
- For PyTorch, doc is coming soon.


Mars Remote
===========

Mars remote provides a simple but powerful way to execute Python functions in
parallel.

Assume we have the code below.

.. code-block:: python

   >>> def add_one(x):
   >>>     return x + 1
   >>>
   >>> def sum_all(xs):
   >>>     return sum(xs)
   >>>
   >>> x_list = []
   >>> for i in range(10):
   >>>     x_list.append(add_one(i))
   >>>
   >>> print(sum_all(x_list))
   55

Here we call `add_one` 10 times, then call `sum_all` to get the summation.

In order to make 10 `add_one` running in parallel, we can rewrite the code as
below.

.. code-block:: python

   >>> import mars.remote as mr
   >>>
   >>> def add_one(x):
   >>>     return x + 1
   >>>
   >>> def sum_all(xs):
   >>>     return sum(xs)
   >>>
   >>> x_list = []
   >>> for i in range(10):
   >>>    x_list.append(mr.spawn(add_one, args=(i,)))
   >>> print(mr.spawn(sum_all, args=(x_list,)).execute().fetch())
   55

The code is quite similar with the previous one, except that calls to `add_one`
and `sum_all` is replaced by `mars.remote.spawn`. `mars.remote.spawn` does not
trigger execution, but instead returns a Mars Object, and the object can be
passed to another `mars.remote.spawn` as an argument. Once `.execute()` is
triggered, the 10 `add_one` will run in parallel.  Once they were finished,
`sum_all` will be triggered. Mars can handle the dependencies correctly, and
for the distributed setting, Users need not to worry about the data movements
between different workers, Mars can handle them automatically.

Refer to :ref:`guidance for Mars remote <remote>` for more information.
