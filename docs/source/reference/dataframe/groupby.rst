.. _generated.groupby:

=======
GroupBy
=======
.. currentmodule:: mars.dataframe.groupby

GroupBy objects are returned by groupby
calls: :func:`mars.dataframe.DataFrame.groupby`, :func:`mars.dataframe.Series.groupby`, etc.

Indexing, iteration
-------------------
.. autosummary::
   :toctree: generated/

.. currentmodule:: mars.dataframe.groupby

.. autosummary::
   :toctree: generated/
   :template: autosummary/class_without_autosummary.rst

.. currentmodule:: mars.dataframe.groupby

Function application
--------------------
.. autosummary::
   :toctree: generated/

   GroupBy.apply
   GroupBy.agg
   GroupBy.aggregate
   GroupBy.transform

Computations / descriptive stats
--------------------------------
.. autosummary::
   :toctree: generated/

   GroupBy.all
   GroupBy.any
   GroupBy.backfill
   GroupBy.bfill
   GroupBy.count
   GroupBy.cumcount
   GroupBy.cummax
   GroupBy.cummin
   GroupBy.cumprod
   GroupBy.cumsum
   GroupBy.ffill
   GroupBy.head
   GroupBy.kurt
   GroupBy.kurtosis
   GroupBy.max
   GroupBy.mean
   GroupBy.min
   GroupBy.size
   GroupBy.sem
   GroupBy.skew
   GroupBy.std
   GroupBy.sum
   GroupBy.var

The following methods are available in both ``SeriesGroupBy`` and
``DataFrameGroupBy`` objects, but may differ slightly, usually in that
the ``DataFrameGroupBy`` version usually permits the specification of an
axis argument, and often an argument indicating whether to restrict
application to columns of a specific data type.

.. autosummary::
   :toctree: generated/

   DataFrameGroupBy.count
   DataFrameGroupBy.cummax
   DataFrameGroupBy.cummin
   DataFrameGroupBy.cumprod
   DataFrameGroupBy.cumsum
   DataFrameGroupBy.fillna
   DataFrameGroupBy.nunique
   DataFrameGroupBy.sample

The following methods are available only for ``SeriesGroupBy`` objects.

.. autosummary::
   :toctree: generated/


The following methods are available only for ``DataFrameGroupBy`` objects.

.. autosummary::
   :toctree: generated/
