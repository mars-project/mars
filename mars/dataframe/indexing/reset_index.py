#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np

from ... import opcodes as OperandDef
from ...core import OutputType
from ...serialization.serializables import BoolField, AnyField, StringField
from ..core import IndexValue
from ..operands import DataFrameOperandMixin, DataFrameOperand, DATAFRAME_TYPE
from ..utils import parse_index, build_empty_df, build_empty_series, standardize_range_index


class DataFrameResetIndex(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.RESET_INDEX

    _level = AnyField('level')
    _drop = BoolField('drop')
    _name = StringField('name')
    _col_level = AnyField('col_level')
    _col_fill = AnyField('col_fill')
    _incremental_index = BoolField('incremental_index')

    def __init__(self, level=None, drop=None, name=None, col_level=None, col_fill=None,
                 incremental_index=None, output_types=None, **kwargs):
        super().__init__(_level=level, _drop=drop, _name=name, _col_level=col_level,
                         _col_fill=col_fill, _incremental_index=incremental_index,
                         _output_types=output_types, **kwargs)

    @property
    def level(self):
        return self._level

    @property
    def drop(self):
        return self._drop

    @property
    def name(self):
        return self._name

    @property
    def col_level(self):
        return self._col_level

    @property
    def col_fill(self):
        return self._col_fill

    @property
    def incremental_index(self):
        return self._incremental_index

    @classmethod
    def _tile_series(cls, op: "DataFrameResetIndex"):
        out_chunks = []
        out = op.outputs[0]
        is_range_index = out.index_value.has_value()
        cum_range = np.cumsum((0, ) + op.inputs[0].nsplits[0])
        for c in op.inputs[0].chunks:
            if is_range_index:
                index_value = parse_index(pd.RangeIndex(cum_range[c.index[0]], cum_range[c.index[0] + 1]))
            else:
                index_value = out.index_value
            chunk_op = op.copy().reset_key()
            if op.drop:
                out_chunk = chunk_op.new_chunk([c], shape=c.shape, index=c.index, dtype=c.dtype,
                                               name=c.name, index_value=index_value)
            else:
                shape = (c.shape[0], out.shape[1])
                out_chunk = chunk_op.new_chunk([c], shape=shape, index=c.index + (0,), dtypes=out.dtypes,
                                               index_value=index_value, columns_value=out.columns_value)
            out_chunks.append(out_chunk)
        if not is_range_index and isinstance(out.index_value.value, IndexValue.RangeIndex) and \
                op.incremental_index:
            out_chunks = standardize_range_index(out_chunks)
        new_op = op.copy()
        if op.drop:
            return new_op.new_seriess(op.inputs, op.inputs[0].shape, nsplits=op.inputs[0].nsplits,
                                      name=out.name, chunks=out_chunks, dtype=out.dtype,
                                      index_value=out.index_value)
        else:
            nsplits = (op.inputs[0].nsplits[0], (out.shape[1],))
            return new_op.new_dataframes(op.inputs, out.shape, nsplits=nsplits, chunks=out_chunks,
                                         index_value=out.index_value, columns_value=out.columns_value,
                                         dtypes=out.dtypes)

    @classmethod
    def _tile_dataframe(cls, op: "DataFrameResetIndex"):
        in_df = op.inputs[0]
        out_df = op.outputs[0]
        added_columns_num = len(out_df.dtypes) - len(in_df.dtypes)
        out_chunks = []
        index_has_value = out_df.index_value.has_value()
        chunk_has_nan = any(np.isnan(s) for s in in_df.nsplits[0])
        cum_range = np.cumsum((0, ) + in_df.nsplits[0])
        for c in in_df.chunks:
            if index_has_value:
                if chunk_has_nan:
                    index_value = parse_index(pd.RangeIndex(-1))
                else:
                    index_value = parse_index(pd.RangeIndex(cum_range[c.index[0]], cum_range[c.index[0] + 1]))
            else:
                index_value = out_df.index_value
            if c.index[1] == 0:
                chunk_op = op.copy().reset_key()
                dtypes = out_df.dtypes[:(added_columns_num + len(c.dtypes))]
                columns_value = parse_index(dtypes.index)
                new_chunk = chunk_op.new_chunk([c], shape=(c.shape[0], c.shape[1] + added_columns_num),
                                               index=c.index, index_value=index_value,
                                               columns_value=columns_value, dtypes=dtypes)
            else:
                chunk_op = op.copy().reset_key()
                chunk_op._drop = True
                new_chunk = chunk_op.new_chunk([c], shape=c.shape, index_value=index_value,
                                               index=c.index, columns_value=c.columns_value, dtypes=c.dtypes)
            out_chunks.append(new_chunk)
        if not index_has_value or chunk_has_nan:
            if isinstance(out_df.index_value.value, IndexValue.RangeIndex) and op.incremental_index:
                out_chunks = standardize_range_index(out_chunks)
        new_op = op.copy()
        columns_splits = list(in_df.nsplits[1])
        columns_splits[0] += added_columns_num
        nsplits = (in_df.nsplits[0], tuple(columns_splits))
        return new_op.new_dataframes(op.inputs, out_df.shape, nsplits=nsplits,
                                     chunks=out_chunks, dtypes=out_df.dtypes,
                                     index_value=out_df.index_value, columns_value=out_df.columns_value)

    @classmethod
    def tile(cls, op):
        if isinstance(op.inputs[0], DATAFRAME_TYPE):
            return cls._tile_dataframe(op)
        else:
            return cls._tile_series(op)

    @classmethod
    def execute(cls, ctx, op):
        in_data = ctx[op.inputs[0].key]
        out = op.outputs[0]

        kwargs = dict()
        if op.name is not None:
            kwargs['name'] = op.name
        if op.col_level is not None:
            kwargs['col_level'] = op.col_level
        if op.col_fill is not None:
            kwargs['col_fill'] = op.col_fill

        r = in_data.reset_index(level=op.level, drop=op.drop, **kwargs)
        if out.index_value.has_value():
            r.index = out.index_value.to_pandas()
        ctx[out.key] = r

    @classmethod
    def _get_out_index(cls, df, out_shape):
        if isinstance(df.index, pd.RangeIndex):
            range_value = -1 if np.isnan(out_shape[0]) else out_shape[0]
            index_value = parse_index(pd.RangeIndex(range_value))
        else:
            index_value = parse_index(df.index)
        return index_value

    def _call_series(self, a):
        if self.drop:
            range_value = -1 if np.isnan(a.shape[0]) else a.shape[0]
            index_value = parse_index(pd.RangeIndex(range_value))
            return self.new_series([a], shape=a.shape, dtype=a.dtype, name=a.name, index_value=index_value)
        else:
            empty_series = build_empty_series(dtype=a.dtype, index=a.index_value.to_pandas()[:0], name=a.name)
            empty_df = empty_series.reset_index(level=self.level, name=self.name)
            shape = (a.shape[0], len(empty_df.dtypes))
            index_value = self._get_out_index(empty_df, shape)
            return self.new_dataframe([a], shape=shape, index_value=index_value,
                                      columns_value=parse_index(empty_df.columns),
                                      dtypes=empty_df.dtypes)

    def _call_dataframe(self, a):
        if self.drop:
            shape = a.shape
            columns_value = a.columns_value
            dtypes = a.dtypes
            range_value = -1 if np.isnan(a.shape[0]) else a.shape[0]
            index_value = parse_index(pd.RangeIndex(range_value))
        else:
            empty_df = build_empty_df(a.dtypes)
            empty_df.index = a.index_value.to_pandas()[:0]
            empty_df = empty_df.reset_index(level=self.level, col_level=self.col_level, col_fill=self.col_fill)
            shape = (a.shape[0], len(empty_df.columns))
            columns_value = parse_index(empty_df.columns, store_data=True)
            dtypes = empty_df.dtypes
            index_value = self._get_out_index(empty_df, shape)
        return self.new_dataframe([a], shape=shape, columns_value=columns_value,
                                  index_value=index_value, dtypes=dtypes)

    def __call__(self, a):
        if isinstance(a, DATAFRAME_TYPE):
            return self._call_dataframe(a)
        else:
            return self._call_series(a)


def df_reset_index(df, level=None, drop=False, inplace=False, col_level=0,
                   col_fill='', incremental_index=False):
    """
    Reset the index, or a level of it.

    Reset the index of the DataFrame, and use the default one instead.
    If the DataFrame has a MultiIndex, this method can remove one or more
    levels.

    Parameters
    ----------
    level : int, str, tuple, or list, default None
        Only remove the given levels from the index. Removes all levels by
        default.
    drop : bool, default False
        Do not try to insert index into dataframe columns. This resets
        the index to the default integer index.
    inplace : bool, default False
        Modify the DataFrame in place (do not create a new object).
    col_level : int or str, default 0
        If the columns have multiple levels, determines which level the
        labels are inserted into. By default it is inserted into the first
        level.
    col_fill : object, default ''
        If the columns have multiple levels, determines how the other
        levels are named. If None then the index name is repeated.
    incremental_index: bool, default False
        Ensure RangeIndex incremental, when output DataFrame has multiple chunks,
        ensuring index incremental costs more computation,
        so by default, each chunk will have index which starts from 0,
        setting incremental_index=True，reset_index will guarantee that
        output DataFrame's index is from 0 to n - 1.

    Returns
    -------
    DataFrame or None
        DataFrame with the new index or None if ``inplace=True``.

    See Also
    --------
    DataFrame.set_index : Opposite of reset_index.
    DataFrame.reindex : Change to new indices or expand indices.
    DataFrame.reindex_like : Change to same indices as other DataFrame.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> import mars.dataframe as md
    >>> df = md.DataFrame([('bird', 389.0),
    ...                    ('bird', 24.0),
    ...                    ('mammal', 80.5),
    ...                    ('mammal', mt.nan)],
    ...                   index=['falcon', 'parrot', 'lion', 'monkey'],
    ...                   columns=('class', 'max_speed'))
    >>> df.execute()
             class  max_speed
    falcon    bird      389.0
    parrot    bird       24.0
    lion    mammal       80.5
    monkey  mammal        NaN

    When we reset the index, the old index is added as a column, and a
    new sequential index is used:

    >>> df.reset_index().execute()
        index   class  max_speed
    0  falcon    bird      389.0
    1  parrot    bird       24.0
    2    lion  mammal       80.5
    3  monkey  mammal        NaN

    We can use the `drop` parameter to avoid the old index being added as
    a column:

    >>> df.reset_index(drop=True).execute()
        class  max_speed
    0    bird      389.0
    1    bird       24.0
    2  mammal       80.5
    3  mammal        NaN

    You can also use `reset_index` with `MultiIndex`.

    >>> import pandas as pd
    >>> index = pd.MultiIndex.from_tuples([('bird', 'falcon'),
    ...                                    ('bird', 'parrot'),
    ...                                    ('mammal', 'lion'),
    ...                                    ('mammal', 'monkey')],
    ...                                   names=['class', 'name'])
    >>> columns = pd.MultiIndex.from_tuples([('speed', 'max'),
    ...                                      ('species', 'type')])
    >>> df = md.DataFrame([(389.0, 'fly'),
    ...                    ( 24.0, 'fly'),
    ...                    ( 80.5, 'run'),
    ...                    (mt.nan, 'jump')],
    ...                   index=index,
    ...                   columns=columns)
    >>> df.execute()
                   speed species
                     max    type
    class  name
    bird   falcon  389.0     fly
           parrot   24.0     fly
    mammal lion     80.5     run
           monkey    NaN    jump

    If the index has multiple levels, we can reset a subset of them:

    >>> df.reset_index(level='class').execute()
             class  speed species
                      max    type
    name
    falcon    bird  389.0     fly
    parrot    bird   24.0     fly
    lion    mammal   80.5     run
    monkey  mammal    NaN    jump

    If we are not dropping the index, by default, it is placed in the top
    level. We can place it in another level:

    >>> df.reset_index(level='class', col_level=1).execute()
                    speed species
             class    max    type
    name
    falcon    bird  389.0     fly
    parrot    bird   24.0     fly
    lion    mammal   80.5     run
    monkey  mammal    NaN    jump

    When the index is inserted under another level, we can specify under
    which one with the parameter `col_fill`:

    >>> df.reset_index(level='class', col_level=1, col_fill='species').execute()
                  species  speed species
                    class    max    type
    name
    falcon           bird  389.0     fly
    parrot           bird   24.0     fly
    lion           mammal   80.5     run
    monkey         mammal    NaN    jump

    If we specify a nonexistent level for `col_fill`, it is created:

    >>> df.reset_index(level='class', col_level=1, col_fill='genus').execute()
                    genus  speed species
                    class    max    type
    name
    falcon           bird  389.0     fly
    parrot           bird   24.0     fly
    lion           mammal   80.5     run
    monkey         mammal    NaN    jump
    """
    op = DataFrameResetIndex(level=level, drop=drop, col_level=col_level,
                             col_fill=col_fill, incremental_index=incremental_index,
                             output_types=[OutputType.dataframe])
    ret = op(df)
    if not inplace:
        return ret
    else:
        df.data = ret.data


def series_reset_index(series, level=None, drop=False, name=None,
                       inplace=False, incremental_index=False):
    """
    Generate a new DataFrame or Series with the index reset.

    This is useful when the index needs to be treated as a column, or
    when the index is meaningless and needs to be reset to the default
    before another operation.

    Parameters
    ----------
    level : int, str, tuple, or list, default optional
        For a Series with a MultiIndex, only remove the specified levels
        from the index. Removes all levels by default.
    drop : bool, default False
        Just reset the index, without inserting it as a column in
        the new DataFrame.
    name : object, optional
        The name to use for the column containing the original Series
        values. Uses ``self.name`` by default. This argument is ignored
        when `drop` is True.
    inplace : bool, default False
        Modify the Series in place (do not create a new object).
    incremental_index: bool, default False
        Ensure RangeIndex incremental, when output Series has multiple chunks,
        ensuring index incremental costs more computation,
        so by default, each chunk will have index which starts from 0,
        setting incremental_index=True，reset_index will guarantee that
        output Series's index is from 0 to n - 1.

    Returns
    -------
    Series or DataFrame
        When `drop` is False (the default), a DataFrame is returned.
        The newly created columns will come first in the DataFrame,
        followed by the original Series values.
        When `drop` is True, a `Series` is returned.
        In either case, if ``inplace=True``, no value is returned.

    See Also
    --------
    DataFrame.reset_index: Analogous function for DataFrame.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> import mars.dataframe as md
    >>> s = md.Series([1, 2, 3, 4], name='foo',
    ...               index=md.Index(['a', 'b', 'c', 'd'], name='idx'))

    Generate a DataFrame with default index.

    >>> s.reset_index().execute()
      idx  foo
    0   a    1
    1   b    2
    2   c    3
    3   d    4

    To specify the name of the new column use `name`.

    >>> s.reset_index(name='values').execute()
      idx  values
    0   a       1
    1   b       2
    2   c       3
    3   d       4

    To generate a new Series with the default set `drop` to True.

    >>> s.reset_index(drop=True).execute()
    0    1
    1    2
    2    3
    3    4
    Name: foo, dtype: int64

    To update the Series in place, without generating a new one
    set `inplace` to True. Note that it also requires ``drop=True``.

    >>> s.reset_index(inplace=True, drop=True)
    >>> s.execute()
    0    1
    1    2
    2    3
    3    4
    Name: foo, dtype: int64

    The `level` parameter is interesting for Series with a multi-level
    index.

    >>> import numpy as np
    >>> import pandas as pd
    >>> arrays = [np.array(['bar', 'bar', 'baz', 'baz']),
    ...           np.array(['one', 'two', 'one', 'two'])]
    >>> s2 = md.Series(
    ...     range(4), name='foo',
    ...     index=pd.MultiIndex.from_arrays(arrays,
    ...                                     names=['a', 'b']))

    To remove a specific level from the Index, use `level`.

    >>> s2.reset_index(level='a').execute()
           a  foo
    b
    one  bar    0
    two  bar    1
    one  baz    2
    two  baz    3

    If `level` is not set, all levels are removed from the Index.

    >>> s2.reset_index().execute()
         a    b  foo
    0  bar  one    0
    1  bar  two    1
    2  baz  one    2
    3  baz  two    3
    """
    op = DataFrameResetIndex(level=level, drop=drop, name=name,
                             incremental_index=incremental_index,
                             output_types=[OutputType.series])
    ret = op(series)
    if not inplace:
        return ret
    elif ret.ndim == 2:
        raise TypeError('Cannot reset_index inplace on a Series to create a DataFrame')
    else:
        series.data = ret.data
