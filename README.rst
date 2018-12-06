Mars
====

Mars is a tensor-based unified framework for large-scale data computation.

Mars tensor
-----------

Mars tensor provides a familiar interface like Numpy.

+------------------------------------------------+----------------------------------------------------+
| **Numpy**                                      | **Mars tensor**                                    |
+------------------------------------------------+----------------------------------------------------+
|.. code-block:: python                          |.. code-block:: python                              |
|                                                |                                                    |
|    import numpy                                |    import mars.tensor as mt                        |
|    a = np.random.rand(1000, 2000)              |    a = mt.random.rand(1000, 2000)                  |
|    (a + 1).sum(axis=1)                         |    (a + 1).sum(axis=1).execute()                   |
|                                                |                                                    |
+------------------------------------------------+----------------------------------------------------+


The following is a brief overview of supported subset of Numpy interface.

- Arithmetic and mathematics: ``+``, ``-``, ``*``, ``/``, ``exp``, ``log``, etc.
- Reduction along axes (``sum``, ``max``, ``argmax``, etc).
- Most of the `array creation routines <https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html>`_
  (``empty``, ``ones_like``, ``diag``, etc). What's more, Mars does not only support create array/tensor on GPU,
  but also support create sparse tensor.
- Most of the `array manipulation routines <https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html>`_
  (``reshape``, ``rollaxis``, ``concatenate``, etc.)
- `Basic indexing <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_
  (indexing by ints, slices, newaxes, and Ellipsis)
- Fancy indexing along single axis with lists or numpy arrays, e.g. x[[1, 4, 8], :5]
- `universal functions <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_
  for elementwise operations.
- `Linear algebra functions <https://docs.scipy.org/doc/numpy/reference/routines.linalg.html>`_,
  including product (``dot``, ``matmul``, etc.) and decomposition (``cholesky``, ``svd``, etc.).

However, Mars has not implemented entire Numpy interface, either the time limitation or difficulty is the main handicap.
Any contribution from community is sincerely welcomed. The main feature not implemented are listed below:

- Tensor with unknown shape does not support all operations.
- Only small subset of ``np.linalg`` are implemented.
- Operations like ``sort`` which is hard to execute in parallel are not implemented.
- Mars tensor doesn't implement interface like ``tolist`` and ``nditer`` etc,
  because the iteration or loops over a large tensor is very inefficient.


Easy to scale in and scale out
------------------------------

Mars can scale in to a single machine, and scale out to a cluster with thousands of machines.
Both the local and distributed version share the same piece of code,
it's fairly simple to migrate from a single machine to a cluster due to the increase of data.
