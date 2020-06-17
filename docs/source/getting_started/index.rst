.. _getting_started_index:

===============
Getting started
===============

Mars leverages parallel and distributed technology to accelerate numpy, pandas,
scikit-learn and Python functions.

There are four main APIs in Mars:

1. :ref:`Mars tensor <getting_started_tensor>`, which mimics numpy API and
   provide ability to process large tensors/ndarrays.
2. :ref:`Mars DataFrame <getting_started_dataframe>`, which mimics pandas API and
   be able to process large DataFrames.
3. :ref:`Mars learn <getting_started_learn>`, which mimics scikit-learn API and
   scales machine learning algorithms.
4. :ref:`Mars Remote <getting_started_remote>`, which provide the ability to
   execute Python functions in parallel.

.. toctree::
   :maxdepth: 2
   :hidden:

   tensor
   dataframe
   learn
   remote

Mars is :ref:`lazy evaluated <lazy_evaluation>` by default,
``.execute()`` is required to perform computation,
however, :ref:`eager mode <eager_mode>` is supported as well,
if eager mode is on, execution will be triggered
every time when each tensor, DataFrame, and so forth is created.

.. toctree::
   :maxdepth: 2
   :hidden:

   execution
   eager
   session

Mars can :ref:`leverage NVIDIA GPU <gpu>` to accelerate computation.

.. toctree::
   :maxdepth: 2
   :hidden:

   gpu
