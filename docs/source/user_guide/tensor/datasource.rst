Create Mars tensor
==================

You can create mars tensor from Python array like object just like Numpy, or create from Numpy array directly.
More details on :doc:`array creation routine <creation>` and :doc:`random sampling <random>`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mars.tensor.tensor
   mars.tensor.array


Create tensor on GPU
--------------------

Mars tensor can run on GPU, for tensor creation, just add a ``gpu`` parameter, and set it to ``True``.

.. code-block:: python

    import mars.tensor as mt

    a = mt.random.rand(1000, 2000, gpu=True)  # allocate the tensor on GPU


Create sparse tensor
--------------------

Mars tensor can be sparse, unfortunately, only 2-D sparse tensors are supported for now,
multi-dimensional tensor will be supported later soon.

.. code-block:: python

    import mars.tensor as mt

    a = mt.eye(1000, sparse=True)  # create a sparse 2-D tensor with ones on the diagonal and zeros elsewhere


Chunks
------

In mars tensor, we tile a tensor into small chunks. Argument ``chunk_size`` is not always required,
a chunk's bytes occupation will be 128M for the default setting.
However, user can specify each chunk's size in a more flexible way which may be adaptive to the data scale.
The fact is that chunk's size may effect heavily on the performance of execution.

The options or arguments which will effect the chunk's size are listed below:

- Change ``options.tensor.chunk_size_limit`` which is 128*1024*1024(128M) by default.
- Specify ``chunk_size`` as integer, like ``5000``, means chunk's size is 5000 at most for all dimensions
- Specify ``chunk_size`` as tuple, like ``(5000, 3000)``
- Explicitly define sizes of all chunks along all dimensions, like ``((5000, 5000, 2000), (2000, 1000))``

Chunks Examples
~~~~~~~~~~~~~~~

Assume we have such a tensor with the data shown below.

.. code-block:: python

    0 9 6 7 6 6
    5 7 5 6 9 0
    1 6 7 8 6 1
    8 0 9 9 9 3
    5 4 3 5 8 2
    6 2 2 6 9 3
    4 2 4 6 2 0
    6 8 2 6 5 4

We will show how different ``chunk_size`` arguments will tile the tensor.

``chunk_size=3``:

.. code-block:: python

    0 9 6  7 6 6
    5 7 5  6 9 0
    1 6 7  8 6 1

    8 0 9  9 9 3
    5 4 3  5 8 2
    6 2 2  6 9 3

    4 2 4  6 2 0
    6 8 2  6 5 4

``chunk_size=2``:

.. code-block:: python

    0 9  6 7  6 6
    5 7  5 6  9 0

    1 6  7 8  6 1
    8 0  9 9  9 3

    5 4  3 5  8 2
    6 2  2 6  9 3

    4 2  4 6  2 0
    6 8  2 6  5 4

``chunk_size=(3, 2)``:

.. code-block:: python

    0 9  6 7  6 6
    5 7  5 6  9 0
    1 6  7 8  6 1

    8 0  9 9  9 3
    5 4  3 5  8 2
    6 2  2 6  9 3

    4 2  4 6  2 0
    6 8  2 6  5 4

``chunk_size=((3, 1, 2, 2), (3, 2, 1))``:

.. code-block:: python

    0 9 6  7 6  6
    5 7 5  6 9  0
    1 6 7  8 6  1

    8 0 9  9 9  3

    5 4 3  5 8  2
    6 2 2  6 9  3

    4 2 4  6 2  0
    6 8 2  6 5  4