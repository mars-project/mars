.. mars documentation master file, created by
   sphinx-quickstart on Mon Mar 26 11:56:11 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Mars
====

Mars is a tensor-based unified framework for large-scale data computation.

Mars tensor
-----------

:doc:`documentation <tensor/index>`

Mars tensor provides a familiar interface like Numpy.

+------------------------------------------------+----------------------------------------------------+
| **Numpy**                                      | **Mars tensor**                                    |
+------------------------------------------------+----------------------------------------------------+
|.. code-block:: python                          |.. code-block:: python                              |
|                                                |                                                    |
|    import numpy as np                          |    import mars.tensor as mt                        |
|    a = np.random.rand(1000, 2000)              |    a = mt.random.rand(1000, 2000)                  |
|    (a + 1).sum(axis=1)                         |    (a + 1).sum(axis=1).execute()                   |
|                                                |                                                    |
+------------------------------------------------+----------------------------------------------------+

Easy to scale in and scale out
------------------------------

Mars can scale in to a single machine, and scale out to a cluster with hundreds of machines.
Both the local and distributed version share the same piece of code,
it's fairly simple to migrate from a single machine to a cluster due to the increase of data.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   install

.. toctree::
   :maxdepth: 2
   :caption: Tensor Interface
   :hidden:

   tensor/overview
   tensor/datasource
   tensor/ufunc
   tensor/routines
   tensor/sparse
   tensor/execution
   tensor/eager-mode

.. toctree::
   :maxdepth: 2
   :caption: Distributed Scheduling
   :hidden:

   distributed/architecture
   distributed/prepare
   distributed/schedule-policy
   distributed/states
   distributed/worker-schedule

.. toctree::
   :maxdepth: 2
   :caption: Contribution Guide

   contributing
