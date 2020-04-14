.. _10min:

****************************
10 minutes to Mars DataFrame
****************************

.. currentmodule:: mars.dataframe

This is a short introduction to Mars DataFrame
which is originated from
`10 minutes to pandas <https://pandas.pydata.org/docs/getting_started/10min.html>`_.

Customarily, we import as follows:

.. ipython:: python

   import mars.tensor as mt
   import mars.dataframe as md

Object creation
---------------

Creating a :class:`Series` by passing a list of values, letting it create
a default integer index:

.. ipython:: python
   :okwarning:

   s = md.Series([1, 3, 5, mt.nan, 6, 8])
   s.execute()

Creating a :class:`DataFrame` by passing a Mars tensor, with a datetime index
and labeled columns:

.. ipython:: python

   dates = md.date_range('20130101', periods=6)
   dates.execute()
   df = md.DataFrame(mt.random.randn(6, 4), index=dates, columns=list('ABCD'))
   df.execute()

Creating a :class:`DataFrame` by passing a dict of objects that can be converted to series-like.

.. ipython:: python

   df2 = md.DataFrame({'A': 1.,
                       'B': md.Timestamp('20130102'),
                       'C': md.Series(1, index=list(range(4)), dtype='float32'),
                       'D': mt.array([3] * 4, dtype='int32'),
                       'E': 'foo'})
   df2.execute()

The columns of the resulting :class:`DataFrame` have different dtypes.

.. ipython:: python

   df2.dtypes


Viewing data
------------

Here is how to view the top and bottom rows of the frame:

.. ipython:: python

   df.head().execute()
   df.tail(3).execute()

Display the index, columns:

.. ipython:: python

   df.index.execute()
   df.columns.execute()


:meth:`DataFrame.to_tensor` gives a Mars tensor representation of the underlying data.
Note that this can be an expensive operation when your :class:`DataFrame` has
columns with different data types, which comes down to a fundamental difference
between DataFrame and tensor: **tensors have one dtype for the entire tensor,
while DataFrames have one dtype per column**. When you call
:meth:`DataFrame.to_tensor`, Mars DataFrame will find the tensor dtype that can hold *all*
of the dtypes in the DataFrame. This may end up being ``object``, which requires
casting every value to a Python object.

For ``df``, our :class:`DataFrame` of all floating-point values,
:meth:`DataFrame.to_tensor` is fast and doesn't require copying data.

.. ipython:: python

   df.to_tensor().execute()

For ``df2``, the :class:`DataFrame` with multiple dtypes,
:meth:`DataFrame.to_tensor` is relatively expensive.

.. ipython:: python

   df2.to_tensor().execute()

.. note::

   :meth:`DataFrame.to_tensor` does *not* include the index or column
   labels in the output.

:func:`~DataFrame.describe` shows a quick statistic summary of your data:

.. ipython:: python

   df.describe().execute()

Sorting by an axis:

.. ipython:: python

   df.sort_index(axis=1, ascending=False).execute()

Sorting by values:

.. ipython:: python

   df.sort_values(by='B').execute()

Selection
---------

.. note::

   While standard Python / Numpy expressions for selecting and setting are
   intuitive and come in handy for interactive work, for production code, we
   recommend the optimized DataFrame data access methods, ``.at``, ``.iat``,
   ``.loc`` and ``.iloc``.


Getting
~~~~~~~

Selecting a single column, which yields a :class:`Series`,
equivalent to ``df.A``:

.. ipython:: python

   df['A'].execute()

Selecting via ``[]``, which slices the rows.

.. ipython:: python

   df[0:3].execute()
   df['20130102':'20130104'].execute()

Selection by label
~~~~~~~~~~~~~~~~~~

For getting a cross section using a label:

.. ipython:: python

   df.loc['20130101'].execute()

Selecting on a multi-axis by label:

.. ipython:: python

   df.loc[:, ['A', 'B']].execute()

Showing label slicing, both endpoints are *included*:

.. ipython:: python

   df.loc['20130102':'20130104', ['A', 'B']].execute()

Reduction in the dimensions of the returned object:

.. ipython:: python

   df.loc['20130102', ['A', 'B']].execute()

For getting a scalar value:

.. ipython:: python

   df.loc['20130101', 'A'].execute()

For getting fast access to a scalar (equivalent to the prior method):

.. ipython:: python

   df.at['20130101', 'A'].execute()

Selection by position
~~~~~~~~~~~~~~~~~~~~~

Select via the position of the passed integers:

.. ipython:: python

   df.iloc[3].execute()

By integer slices, acting similar to numpy/python:

.. ipython:: python

   df.iloc[3:5, 0:2].execute()

By lists of integer position locations, similar to the numpy/python style:

.. ipython:: python

   df.iloc[[1, 2, 4], [0, 2]].execute()

For slicing rows explicitly:

.. ipython:: python

   df.iloc[1:3, :].execute()

For slicing columns explicitly:

.. ipython:: python

   df.iloc[:, 1:3].execute()

For getting a value explicitly:

.. ipython:: python

   df.iloc[1, 1].execute()

For getting fast access to a scalar (equivalent to the prior method):

.. ipython:: python

   df.iat[1, 1].execute()

Boolean indexing
~~~~~~~~~~~~~~~~

Using a single column's values to select data.

.. ipython:: python

   df[df['A'] > 0].execute()

Selecting values from a DataFrame where a boolean condition is met.

.. ipython:: python

   df[df > 0].execute()


Operations
----------

Stats
~~~~~

Operations in general *exclude* missing data.

Performing a descriptive statistic:

.. ipython:: python

   df.mean().execute()

Same operation on the other axis:

.. ipython:: python

   df.mean(1).execute()


Operating with objects that have different dimensionality and need alignment.
In addition, pandas automatically broadcasts along the specified dimension.

.. ipython:: python

   s = md.Series([1, 3, 5, mt.nan, 6, 8], index=dates).shift(2)
   s.execute()
   df.sub(s, axis='index').execute()


Apply
~~~~~

Applying functions to the data:

.. ipython:: python

   df.apply(lambda x: x.max() - x.min()).execute()

String Methods
~~~~~~~~~~~~~~

Series is equipped with a set of string processing methods in the `str`
attribute that make it easy to operate on each element of the array, as in the
code snippet below. Note that pattern-matching in `str` generally uses `regular
expressions <https://docs.python.org/3/library/re.html>`__ by default (and in
some cases always uses them). See more at :ref:`Vectorized String Methods
<text.string_methods>`.

.. ipython:: python

   s = md.Series(['A', 'B', 'C', 'Aaba', 'Baca', mt.nan, 'CABA', 'dog', 'cat'])
   s.str.lower().execute()

Merge
-----

Concat
~~~~~~

Mars DataFrame provides various facilities for easily combining together Series and
DataFrame objects with various kinds of set logic for the indexes
and relational algebra functionality in the case of join / merge-type
operations.

Concatenating DataFrame objects together with :func:`concat`:

.. ipython:: python

   df = md.DataFrame(mt.random.randn(10, 4))
   df.execute()

   # break it into pieces
   pieces = [df[:3], df[3:7], df[7:]]

   md.concat(pieces).execute()

.. note::
   Adding a column to a :class:`DataFrame` is relatively fast. However, adding
   a row requires a copy, and may be expensive. We recommend passing a
   pre-built list of records to the :class:`DataFrame` constructor instead
   of building a :class:`DataFrame` by iteratively appending records to it.

Join
~~~~

SQL style merges. See the :ref:`Database style joining <merging.join>` section.

.. ipython:: python

   left = md.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
   right = md.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
   left.execute()
   right.execute()
   md.merge(left, right, on='key').execute()

Another example that can be given is:

.. ipython:: python

   left = md.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
   right = md.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
   left.execute()
   right.execute()
   md.merge(left, right, on='key').execute()

Grouping
--------

By "group by" we are referring to a process involving one or more of the
following steps:

 - **Splitting** the data into groups based on some criteria
 - **Applying** a function to each group independently
 - **Combining** the results into a data structure


.. ipython:: python

   df = md.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                            'foo', 'bar', 'foo', 'foo'],
                      'B': ['one', 'one', 'two', 'three',
                            'two', 'two', 'one', 'three'],
                      'C': mt.random.randn(8),
                      'D': mt.random.randn(8)})
   df.execute()

Grouping and then applying the :meth:`~mars.dataframe.groupby.GroupBy.sum` function to the resulting
groups.

.. ipython:: python

   df.groupby('A').sum().execute()

Grouping by multiple columns forms a hierarchical index, and again we can
apply the `sum` function.

.. ipython:: python

   df.groupby(['A', 'B']).sum().execute()

Plotting
--------

We use the standard convention for referencing the matplotlib API:

.. ipython:: python

   import matplotlib.pyplot as plt
   plt.close('all')

.. ipython:: python

   ts = md.Series(mt.random.randn(1000),
                  index=md.date_range('1/1/2000', periods=1000))
   ts = ts.cumsum()

   @savefig series_plot_basic.png
   ts.plot()

On a DataFrame, the :meth:`~DataFrame.plot` method is a convenience to plot all
of the columns with labels:

.. ipython:: python

   df = md.DataFrame(mt.random.randn(1000, 4), index=ts.index,
                     columns=['A', 'B', 'C', 'D'])
   df = df.cumsum()

   plt.figure()
   df.plot()
   @savefig frame_plot_basic.png
   plt.legend(loc='best')

Getting data in/out
-------------------

CSV
~~~

.. ipython:: python

   df.to_csv('foo.csv').execute()

:ref:`Reading from a csv file. <io.read_csv_table>`

.. ipython:: python

   md.read_csv('foo.csv').execute()

.. ipython:: python
   :suppress:

   import os
   os.remove('foo.csv')
