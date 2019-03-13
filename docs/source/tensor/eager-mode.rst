Eager Mode
===========

.. Note:: New in version 0.2.0a2

Mars supports eager mode which makes it friendly for developing and easy to
debug.

Users can enable the eager mode by options, set options at the beginning of the
program or console session.

.. code-block:: python

    >>> from mars.config import options
    >>> options.eager_mode = True

Or use a context.

.. code-block:: python

    >>> from mars.config import option_context

    >>> with option_context() as options:
    >>>     options.eager_mode = True
    >>>     # the eager mode is on only for the with statement
    >>>     ...

If eager mode is on, tensor will be executed immediately by default session
once it is created.

.. code-block:: python

    >>> import mars.tensor as mt
    >>> from mars.config import options
    >>> options.eager_mode = True
    >>> t = mt.arange(6).reshape((2, 3))
    >>> print(t)
    Tensor(op=TensorRand, shape=(2, 3), data=
    [[0 1 2]
    [3 4 5]])

Use ``fetch`` to obtain numpy value from a tensor:

.. code-block:: python

    >>> t.fetch()
    array([[0, 1, 2],
           [3, 4, 5]])
