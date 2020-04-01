.. _api.groupby:

=======
GroupBy
=======
.. currentmodule:: mars.dataframe.groupby

GroupBy objects are returned by groupby
calls: :func:`mars.dataframe.DataFrame.groupby`, :func:`mars.dataframe.Series.groupby`, etc.

Indexing, iteration
-------------------
.. autosummary::
   :toctree: api/

.. currentmodule:: mars.dataframe.groupby

.. autosummary::
   :toctree: api/
   :template: autosummary/class_without_autosummary.rst

.. currentmodule:: mars.dataframe.groupby

Function application
--------------------
.. autosummary::
   :toctree: api/

   GroupBy.apply
   GroupBy.agg
   GroupBy.aggregate
   GroupBy.transform

Computations / descriptive stats
--------------------------------
.. autosummary::
   :toctree: api/

   GroupBy.count
   GroupBy.cumcount
   GroupBy.cummax
   GroupBy.cummin
   GroupBy.cumprod
   GroupBy.cumsum
   GroupBy.max
   GroupBy.mean
   GroupBy.min
   GroupBy.std
   GroupBy.sum
   GroupBy.var

The following methods are available in both ``SeriesGroupBy`` and
``DataFrameGroupBy`` objects, but may differ slightly, usually in that
the ``DataFrameGroupBy`` version usually permits the specification of an
axis argument, and often an argument indicating whether to restrict
application to columns of a specific data type.

.. autosummary::
   :toctree: api/

   DataFrameGroupBy.count
   DataFrameGroupBy.cummax
   DataFrameGroupBy.cummin
   DataFrameGroupBy.cumprod
   DataFrameGroupBy.cumsum

The following methods are available only for ``SeriesGroupBy`` objects.

.. autosummary::
   :toctree: api/


The following methods are available only for ``DataFrameGroupBy`` objects.

.. autosummary::
   :toctree: api/
