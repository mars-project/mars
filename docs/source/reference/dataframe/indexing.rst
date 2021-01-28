=============
Index objects
=============
.. currentmodule:: mars.dataframe

Constructor
-----------
.. autosummary::
   :toctree: generated/

   Index

Properties
----------
.. autosummary::
   :toctree: generated/

   Index.dtype
   Index.inferred_type
   Index.is_monotonic
   Index.is_monotonic_decreasing
   Index.is_monotonic_increasing
   Index.name
   Index.names
   Index.ndim
   Index.size
   Index.memory_usage

Modifying and computations
--------------------------
.. autosummary::
   :toctree: generated/

   Index.all
   Index.any
   Index.drop
   Index.drop_duplicates
   Index.max
   Index.min
   Index.rename

Compatibility with MultiIndex
-----------------------------
.. autosummary::
   :toctree: generated/

   Index.set_names


Missing values
--------------
.. autosummary::
   :toctree: generated/

   Index.fillna
   Index.dropna
   Index.isna
   Index.notna


Conversion
----------
.. autosummary::
   :toctree: generated/

   Index.astype
   Index.map
   Index.to_frame
   Index.to_series
