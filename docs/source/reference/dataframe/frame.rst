.. _api.dataframe:

=========
DataFrame
=========
.. currentmodule:: mars.dataframe

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Axes**

.. autosummary::
   :toctree: api/

   DataFrame.index
   DataFrame.columns

.. autosummary::
   :toctree: api/

   DataFrame.dtypes
   DataFrame.select_dtypes
   DataFrame.ndim
   DataFrame.shape
   DataFrame.memory_usage

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.astype
   DataFrame.copy
   DataFrame.isna
   DataFrame.notna

Indexing, iteration
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.head
   DataFrame.at
   DataFrame.iat
   DataFrame.loc
   DataFrame.iloc
   DataFrame.insert
   DataFrame.iterrows
   DataFrame.itertuples
   DataFrame.pop
   DataFrame.tail
   DataFrame.where
   DataFrame.mask

Binary operator functions
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.add
   DataFrame.sub
   DataFrame.mul
   DataFrame.div
   DataFrame.truediv
   DataFrame.floordiv
   DataFrame.mod
   DataFrame.pow
   DataFrame.dot
   DataFrame.radd
   DataFrame.rsub
   DataFrame.rmul
   DataFrame.rdiv
   DataFrame.rtruediv
   DataFrame.rfloordiv
   DataFrame.rmod
   DataFrame.rpow
   DataFrame.lt
   DataFrame.gt
   DataFrame.le
   DataFrame.ge
   DataFrame.ne
   DataFrame.eq

Function application, GroupBy & window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.apply
   DataFrame.agg
   DataFrame.aggregate
   DataFrame.transform
   DataFrame.groupby
   DataFrame.rolling
   DataFrame.expanding
   DataFrame.ewm

.. _api.dataframe.stats:

Computations / descriptive stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.abs
   DataFrame.all
   DataFrame.any
   DataFrame.corr
   DataFrame.corrwith
   DataFrame.count
   DataFrame.cummax
   DataFrame.cummin
   DataFrame.cumprod
   DataFrame.cumsum
   DataFrame.describe
   DataFrame.max
   DataFrame.mean
   DataFrame.min
   DataFrame.nunique
   DataFrame.prod
   DataFrame.product
   DataFrame.quantile
   DataFrame.round
   DataFrame.std
   DataFrame.sum
   DataFrame.var

Reindexing / selection / label manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.drop
   DataFrame.drop_duplicates
   DataFrame.head
   DataFrame.reindex
   DataFrame.rename
   DataFrame.reset_index
   DataFrame.set_index
   DataFrame.tail

.. _api.dataframe.missing:

Missing data handling
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.dropna
   DataFrame.fillna

Reshaping, sorting, transposing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.melt
   DataFrame.sort_values
   DataFrame.sort_index
   DataFrame.stack

Combining / joining / merging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.append
   DataFrame.join
   DataFrame.merge

Time series-related
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.diff
   DataFrame.shift
   DataFrame.tshift

.. _api.dataframe.plotting:

Plotting
~~~~~~~~
``DataFrame.plot`` is both a callable method and a namespace attribute for
specific plotting methods of the form ``DataFrame.plot.<kind>``.

.. autosummary::
   :toctree: api/

   DataFrame.plot

.. autosummary::
   :toctree: api/

   DataFrame.plot.area
   DataFrame.plot.bar
   DataFrame.plot.barh
   DataFrame.plot.box
   DataFrame.plot.density
   DataFrame.plot.hexbin
   DataFrame.plot.hist
   DataFrame.plot.kde
   DataFrame.plot.line
   DataFrame.plot.pie
   DataFrame.plot.scatter

.. autosummary::
   :toctree: api/

   DataFrame.boxplot
   DataFrame.hist

.. _api.dataframe.io:

Serialization / IO / conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.to_csv
   DataFrame.to_parquet
   DataFrame.to_sql

Misc
~~~~

.. autosummary::
  :toctree: api/

   DataFrame.map_chunk
   DataFrame.rebalance
