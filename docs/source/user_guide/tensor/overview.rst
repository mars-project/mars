Overview
========

Mars tensor is the counterpart of Numpy :class:`numpy.ndarray` and implements a subset of the Numpy ndarray interface.
It tiles a large tensor into small chunks and describe the inner computation with a directed graph.
This lets us compute on tensors larger than memory and take advantage of the ability of multi-cores or distributed clusters.


The following is a brief overview of supported subset of Numpy interface.

- Arithmetic and mathematics: ``+``, ``-``, ``*``, ``/``, ``exp``, ``log``, etc.
- Reduction along axes (``sum``, ``max``, ``argmax``, etc).
- Most of the `array creation routines <https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html>`_
  (``empty``, ``ones_like``, ``diag``, etc). What's more, Mars does not only support create array/tensor on GPU,
  but also support create sparse tensor.
- Most of the `array manipulation routines <https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html>`_
  (``reshape``, ``rollaxis``, ``concatenate``, etc.)
- `Basic indexing <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_
  (indexing by ints, slices, newaxes, and Ellipsis).
- `Advanced indexing <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing>`_
  (except combing boolean array indexing and integer array indexing).
- `universal functions <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_
  for elementwise operations.
- `Linear algebra functions <https://docs.scipy.org/doc/numpy/reference/routines.linalg.html>`_,
  including product (``dot``, ``matmul``, etc.) and decomposition (``cholesky``, ``svd``, etc.).

However, Mars has not implemented entire Numpy interface, either the time limitation or difficulty is the main handicap.
Any contribution from community is sincerely welcomed. The main feature not implemented are listed below:

- Tensor with unknown shape does not support all operations.
- Only small subset of ``np.linalg`` are implemented.
- Mars tensor doesn't implement interface like ``tolist`` and ``nditer`` etc,
  because the iteration or loops over a large tensor is very inefficient.
