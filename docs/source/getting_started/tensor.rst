.. _getting_started_tensor:

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
lazy evaluated.  You need to call ``.execute()`` to get the final result.

Remember that ``.execute()`` will return Mars tensor itself.

.. code-block:: python

   >>> (t - (t + 1).sum()).execute()
   array([[-23.06773811, -22.86112123, -23.03988405, -22.48884341],
          [-22.54959727, -22.13498645, -22.97627675, -23.09852276],
          [-23.11085224, -22.63999173, -22.27187961, -22.34163038],
          [-22.40633932, -22.17864095, -23.04577731, -22.76189835]])

For more implemented tensor API, refer to :ref:`tensor API reference <tensor_api>`.

Once a tensor is executed, ``.fetch()`` could be called to get the result as
numpy ndarray.  A shortcut ``.to_numpy()`` is identical to ``.execute().fetch()``.

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

    Users should always consider using ``.execute()`` instead of ``.to_numpy()``,
    because when the tensor is large, ``.execute()`` will only fetch the edge items
    for display purpose. On the other hand, ``.to_numpy()`` will try to generate
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
