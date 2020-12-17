======
Series
======
.. currentmodule:: mars.dataframe

Constructor
-----------
.. autosummary::
   :toctree: generated/

   Series

Attributes
----------
**Axes**

.. autosummary::
   :toctree: generated/

   Series.index

.. autosummary::
   :toctree: generated/

   Series.dtype
   Series.shape
   Series.ndim
   Series.name
   Series.memory_usage

Conversion
----------
.. autosummary::
   :toctree: generated/

   Series.astype
   Series.copy
   Series.to_frame
   Series.to_tensor

Indexing, iteration
-------------------
.. autosummary::
   :toctree: generated/

   Series.at
   Series.iat
   Series.loc
   Series.iloc
   Series.where
   Series.mask

Binary operator functions
-------------------------
.. autosummary::
   :toctree: generated/

   Series.add
   Series.sub
   Series.mul
   Series.div
   Series.truediv
   Series.floordiv
   Series.mod
   Series.pow
   Series.radd
   Series.rsub
   Series.rmul
   Series.rdiv
   Series.rtruediv
   Series.rfloordiv
   Series.rmod
   Series.rpow
   Series.lt
   Series.gt
   Series.le
   Series.ge
   Series.ne
   Series.eq
   Series.dot

Function application, groupby & window
--------------------------------------
.. autosummary::
   :toctree: generated/

   Series.apply
   Series.agg
   Series.aggregate
   Series.transform
   Series.map
   Series.groupby
   Series.rolling
   Series.expanding
   Series.ewm

.. _generated.series.stats:

Computations / descriptive stats
--------------------------------
.. autosummary::
   :toctree: generated/

   Series.abs
   Series.all
   Series.any
   Series.autocorr
   Series.corr
   Series.count
   Series.cummax
   Series.cummin
   Series.cumprod
   Series.cumsum
   Series.describe
   Series.kurt
   Series.kurtosis
   Series.max
   Series.mean
   Series.min
   Series.prod
   Series.product
   Series.quantile
   Series.round
   Series.sem
   Series.skew
   Series.std
   Series.sum
   Series.var
   Series.nunique
   Series.value_counts

Reindexing / selection / label manipulation
-------------------------------------------
.. autosummary::
   :toctree: generated/

   Series.drop
   Series.drop_duplicates
   Series.head
   Series.isin
   Series.reindex
   Series.rename
   Series.reset_index
   Series.tail

Missing data handling
---------------------
.. autosummary::
   :toctree: generated/

   Series.isna
   Series.notna
   Series.dropna
   Series.fillna

Reshgeneratedng, sorting
------------------
.. autosummary::
   :toctree: generated/

   Series.explode
   Series.sort_values
   Series.sort_index

Combining / joining / merging
-----------------------------
.. autosummary::
   :toctree: generated/

   Series.append

Time Series-related
-------------------
.. autosummary::
   :toctree: generated/

   Series.diff
   Series.shift
   Series.tshift

Accessors
---------

Pandas provides dtype-specific methods under various accessors.
These are separate namespaces within :class:`Series` that only apply
to specific data types.

=========================== =================================
Data Type                   Accessor
=========================== =================================
Datetime, Timedelta, Period :ref:`dt <generated.series.dt>`
String                      :ref:`str <generated.series.str>`
=========================== =================================

.. _generated.series.dt:

Datetimelike properties
~~~~~~~~~~~~~~~~~~~~~~~

``Series.dt`` can be used to access the values of the series as
datetimelike and return several properties.
These can be accessed like ``Series.dt.<property>``.

Datetime properties
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_attribute.rst

   Series.dt.date
   Series.dt.time
   Series.dt.timetz
   Series.dt.year
   Series.dt.month
   Series.dt.day
   Series.dt.hour
   Series.dt.minute
   Series.dt.second
   Series.dt.microsecond
   Series.dt.nanosecond
   Series.dt.week
   Series.dt.weekofyear
   Series.dt.dayofweek
   Series.dt.weekday
   Series.dt.dayofyear
   Series.dt.quarter
   Series.dt.is_month_start
   Series.dt.is_month_end
   Series.dt.is_quarter_start
   Series.dt.is_quarter_end
   Series.dt.is_year_start
   Series.dt.is_year_end
   Series.dt.is_leap_year
   Series.dt.daysinmonth
   Series.dt.days_in_month
   Series.dt.tz
   Series.dt.freq

Datetime methods
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   Series.dt.to_period
   Series.dt.to_pydatetime
   Series.dt.tz_localize
   Series.dt.tz_convert
   Series.dt.normalize
   Series.dt.strftime
   Series.dt.round
   Series.dt.floor
   Series.dt.ceil
   Series.dt.month_name
   Series.dt.day_name

Period properties
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_attribute.rst

   Series.dt.qyear
   Series.dt.start_time
   Series.dt.end_time

Timedelta properties
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_attribute.rst

   Series.dt.days
   Series.dt.seconds
   Series.dt.microseconds
   Series.dt.nanoseconds
   Series.dt.components

Timedelta methods
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   Series.dt.to_pytimedelta
   Series.dt.total_seconds


.. _generated.series.str:

String handling
~~~~~~~~~~~~~~~

``Series.str`` can be used to access the values of the series as
strings and apply several methods to it. These can be accessed like
``Series.str.<function/property>``.

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   Series.str.cgeneratedtalize
   Series.str.casefold
   Series.str.cat
   Series.str.center
   Series.str.contains
   Series.str.count
   Series.str.decode
   Series.str.encode
   Series.str.endswith
   Series.str.extract
   Series.str.extractall
   Series.str.find
   Series.str.findall
   Series.str.get
   Series.str.index
   Series.str.join
   Series.str.len
   Series.str.ljust
   Series.str.lower
   Series.str.lstrip
   Series.str.match
   Series.str.normalize
   Series.str.pad
   Series.str.partition
   Series.str.repeat
   Series.str.replace
   Series.str.rfind
   Series.str.rindex
   Series.str.rjust
   Series.str.rpartition
   Series.str.rstrip
   Series.str.slice
   Series.str.slice_replace
   Series.str.split
   Series.str.rsplit
   Series.str.startswith
   Series.str.strip
   Series.str.swapcase
   Series.str.title
   Series.str.translate
   Series.str.upper
   Series.str.wrap
   Series.str.zfill
   Series.str.isalnum
   Series.str.isalpha
   Series.str.isdigit
   Series.str.isspace
   Series.str.islower
   Series.str.isupper
   Series.str.istitle
   Series.str.isnumeric
   Series.str.isdecimal

..
    The following is needed to ensure the generated pages are created with the
    correct template (otherwise they would be created in the Series/Index class page)

..
    .. autosummary::
       :toctree: generated/
       :template: accessor.rst

       Series.str
       Series.dt

Plotting
--------
``Series.plot`` is both a callable method and a namespace attribute for
specific plotting methods of the form ``Series.plot.<kind>``.

.. autosummary::
   :toctree: generated/
   :template: accessor_callable.rst

   Series.plot

.. autosummary::
   :toctree: generated/
   :template: accessor_method.rst

   Series.plot.area
   Series.plot.bar
   Series.plot.barh
   Series.plot.box
   Series.plot.density
   Series.plot.hist
   Series.plot.kde
   Series.plot.line
   Series.plot.pie

.. autosummary::
   :toctree: generated/

   Series.hist

Serialization / IO / conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   Series.to_csv
   Series.to_sql

Misc
~~~~

.. autosummary::
  :toctree: generated/

   Series.map_chunk
