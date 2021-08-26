.. _dask:

============
Dask on Mars
============

.. Note:: New in version 0.8.0a2

Dask-on-Mars provides a simple way to execute the entire Dask ecosystem on top of Mars.

`Dask <https://dask.org/>`__ is a flexible library for parallel computing in Python, geared towards 
scaling analytics and scientific computing workloads. It provides `big data collections
<https://docs.dask.org/en/latest/user-interfaces.html>`__ and Dynamic task scheduling 
optimized for computation.

.. note::
  For execution on Mars, you should *not* use the
  `Dask.distributed <https://distributed.dask.org/en/latest/quickstart.html>`__
  client, simply use plain Dask collections and functionalities.

Scheduler
---------

The main API for Dask-on-Mars is :meth:`mars.contrib.dask.mars_scheduler`. It 
uses Dask’s scheduler API, which allows you to specify any callable as the 
scheduler that you would like Dask to use to execute your workload. 

.. code-block:: python

   >>> import dask
   >>> from mars.contrib.dask import mars_scheduler
   >>>
   >>> def inc(x):
   >>>     return x + 1
   >>>
   >>> dask_task = dask.delayed(inc)(1)
   >>> dask_task.compute(scheduler=mars_scheduler) # Run delayed object on top of Mars
   2

Convert Dask Collections
------------------------

:meth:`mars.contrib.dask.convert_dask_collection` can be used when user needs to 
manipulate dask collections with :ref:`Mars remote API <remote>` or other 
features. It converts dask collections like delayed or dask-dataframe to Mars Objects, 
which can be considered as results returned by :meth:`mars.remote.spawn`.

.. code-block:: python

   >>> import dask
   >>> import mars.remote as mr
   >>> from mars.contrib.dask import convert_dask_collection
   >>>
   >>> def inc(x):
   >>>     return x + 1
   >>>
   >>> dask_task = dask.delayed(inc)(1)
   >>> mars_obj = convert_dask_collection(dask_task) # Convert Dask object to Mars object
   >>> mars_task = mr.spawn(inc, args=(mars_obj,))
   >>> mars_task
   Object <op=RemoteFunction, key=14a77b28d32904002829b2e8c6474b56>
   >>> mars_task.execute().fetch()
   3

Dask-on-Mars is an ongoing project. Please open an issue if you find that one of 
these dask functionalities doesn’t run on Mars.