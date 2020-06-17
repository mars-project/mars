.. _eager_mode:

Eager Execution
===============

.. Note:: New in version 0.2.0a2

Mars supports eager mode, making it friendly for developing and easy to debug.

Users can enable eager mode by setting options at the beginning of the program
or console session.

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

If eager mode is on, Mars objects like tensors and DataFrames will be executed
immediately by default session once it is created.

.. code-block:: python

    >>> import mars.tensor as mt
    >>> import mars.dataframe as md
    >>> from mars.config import options
    >>> options.eager_mode = True
    >>> t = mt.arange(6).reshape(2, 3)
    >>> t
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> df = md.DataFrame(t)
    >>> df.sum()
    0    3
    1    5
    2    7
    dtype: int64

