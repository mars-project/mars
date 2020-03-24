# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

from ...serialize import ListField, ValueType
from ... import opcodes as OperandDef
from ...tensor.base.sort import _validate_sort_psrs_kinds
from ..utils import parse_index, validate_axis, build_concatenated_rows_frame
from ..core import IndexValue
from ..operands import ObjectType
from .core import DataFrameSortOperand
from .psrs import DataFramePSRSOperandMixin, execute_sort_values


class DataFrameSortValues(DataFrameSortOperand, DataFramePSRSOperandMixin):
    _op_type_ = OperandDef.SORT_VALUES

    _by = ListField('by', ValueType.string)

    def __init__(self, by=None, object_type=None, **kw):
        super(DataFrameSortValues, self).__init__(_by=by, _object_type=object_type, **kw)

    @property
    def by(self):
        return self._by

    @classmethod
    def _tile_dataframe(cls, op):
        df = build_concatenated_rows_frame(op.inputs[0])

        if df.chunk_shape[op.axis] == 1:
            out_chunks = []
            for chunk in df.chunks:
                chunk_op = op.copy().reset_key()
                out_chunks.append(chunk_op.new_chunk(
                    [chunk], shape=chunk.shape, index=chunk.index, index_value=op.outputs[0].index_value,
                    columns_value=chunk.columns_value, dtypes=chunk.dtypes))
            new_op = op.copy()
            kws = op.outputs[0].params.copy()
            kws['nsplits'] = df.nsplits
            kws['chunks'] = out_chunks
            return new_op.new_dataframes(op.inputs, **kws)
        else:
            if op.na_position != 'last':  # pragma: no cover
                raise NotImplementedError('Only support puts NaNs at the end.')
            # use parallel sorting by regular sampling
            return cls._tile_psrs(op, df)

    @classmethod
    def _tile_series(cls, op):
        series = op.inputs[0]
        if len(series.chunks) == 1:
            chunk = series.chunks[0]
            chunk_op = op.copy().reset_key()
            out_chunks = [chunk_op.new_chunk(series.chunks, shape=chunk.shape, index=chunk.index,
                                             index_value=op.outputs[0].index_value, dtype=chunk.dtype,
                                             name=chunk.name)]
            new_op = op.copy()
            kws = op.outputs[0].params.copy()
            kws['nsplits'] = series.nsplits
            kws['chunks'] = out_chunks
            return new_op.new_seriess(op.inputs, **kws)
        else:
            if op.na_position != 'last':  # pragma: no cover
                raise NotImplementedError('Only support puts NaNs at the end.')
            # use parallel sorting by regular sampling
            return cls._tile_psrs(op, series)

    @classmethod
    def tile(cls, op):
        if op.object_type == ObjectType.dataframe:
            return cls._tile_dataframe(op)
        else:
            return cls._tile_series(op)

    @classmethod
    def execute(cls, ctx, op):
        in_data = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = execute_sort_values(in_data, op)

    def __call__(self, a):
        assert self.axis == 0
        if self.ignore_index:
            index_value = parse_index(pd.RangeIndex(a.shape[0]))
        else:
            if isinstance(a.index_value.value, IndexValue.RangeIndex):
                index_value = parse_index(pd.Int64Index([]))
            else:
                index_value = a.index_value
        if a.op.object_type == ObjectType.dataframe:
            return self.new_dataframe([a], shape=a.shape, dtypes=a.dtypes,
                                      index_value=index_value,
                                      columns_value=a.columns_value)
        else:
            return self.new_series([a], shape=a.shape, dtype=a.dtype,
                                   index_value=index_value, name=a.name)


def dataframe_sort_values(df, by, axis=0, ascending=True, inplace=False, kind='quicksort',
                          na_position='last', ignore_index=False, parallel_kind='PSRS', psrs_kinds=None):
    """
    Sort by the values along either axis.
    :param df: input DataFrame.
    :param by: Name or list of names to sort by.
    :param axis: Axis to be sorted.
    :param ascending: Sort ascending vs. descending. Specify list for multiple sort orders.
    If this is a list of bools, must match the length of the by.
    :param inplace: If True, perform operation in-place.
    :param kind: Choice of sorting algorithm. See also ndarray.np.sort for more information.
    mergesort is the only stable algorithm. For DataFrames, this option is only applied
    when sorting on a single column or label.
    :param na_position: Puts NaNs at the beginning if first; last puts NaNs at the end.
    :param ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
    :param parallel_kind: {'PSRS'}, optional. Parallel sorting algorithm, for the details, refer to:
    http://csweb.cs.wfu.edu/bigiron/LittleFE-PSRS/build/html/PSRSalgorithm.html
    :param psrs_kinds: Sorting algorithms during PSRS algorithm.
    :return: sorted dataframe.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> raw = pd.DataFrame({
    ...     'col1': ['A', 'A', 'B', np.nan, 'D', 'C'],
    ...     'col2': [2, 1, 9, 8, 7, 4],
    ...     'col3': [0, 1, 9, 4, 2, 3],
    ... })
    >>> df = md.DataFrame(raw)
    >>> df.execute()
        col1 col2 col3
    0   A    2    0
    1   A    1    1
    2   B    9    9
    3   NaN  8    4
    4   D    7    2
    5   C    4    3

    Sort by col1

    >>> df.sort_values(by=['col1']).execute()
        col1 col2 col3
    0   A    2    0
    1   A    1    1
    2   B    9    9
    5   C    4    3
    4   D    7    2
    3   NaN  8    4

    Sort by multiple columns

    >>> df.sort_values(by=['col1', 'col2']).execute()
        col1 col2 col3
    1   A    1    1
    0   A    2    0
    2   B    9    9
    5   C    4    3
    4   D    7    2
    3   NaN  8    4

    Sort Descending

    >>> df.sort_values(by='col1', ascending=False).execute()
        col1 col2 col3
    4   D    7    2
    5   C    4    3
    2   B    9    9
    0   A    2    0
    1   A    1    1
    3   NaN  8    4

    """
    if na_position not in ['last', 'first']:  # pragma: no cover
        raise TypeError('invalid na_position: {}'.format(na_position))
    axis = validate_axis(axis, df)
    if axis != 0:
        raise NotImplementedError('Only support sort on axis 0')
    psrs_kinds = _validate_sort_psrs_kinds(psrs_kinds)
    by = by if isinstance(by, (list, tuple)) else [by]
    op = DataFrameSortValues(by=by, axis=axis, ascending=ascending, inplace=inplace, kind=kind,
                             na_position=na_position, ignore_index=ignore_index, parallel_kind=parallel_kind,
                             psrs_kinds=psrs_kinds, object_type=ObjectType.dataframe)
    sorted_df = op(df)
    if inplace:
        df.data = sorted_df.data
    else:
        return sorted_df


def series_sort_values(series, axis=0, ascending=True, inplace=False, kind='quicksort',
                       na_position='last', ignore_index=False, parallel_kind='PSRS', psrs_kinds=None):
    """
    Sort by the values.

    Sort a Series in ascending or descending order by some
    criterion.

    Parameters
    ----------
    series : input Series.
    axis : {0 or 'index'}, default 0
        Axis to direct sorting. The value 'index' is accepted for
        compatibility with DataFrame.sort_values.
    ascending : bool, default True
        If True, sort values in ascending order, otherwise descending.
    inplace : bool, default False
        If True, perform operation in-place.
    kind : {'quicksort', 'mergesort' or 'heapsort'}, default 'quicksort'
        Choice of sorting algorithm. See also :func:`numpy.sort` for more
        information. 'mergesort' is the only stable  algorithm.
    na_position : {'first' or 'last'}, default 'last'
        Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
        the end.
    ignore_index : bool, default False
         If True, the resulting axis will be labeled 0, 1, …, n - 1.

    Returns
    -------
    Series
        Series ordered by values.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> raw = pd.Series([np.nan, 1, 3, 10, 5])
    >>> s = md.Series(raw)
    >>> s.execute()
    0     NaN
    1     1.0
    2     3.0
    3     10.0
    4     5.0
    dtype: float64

    Sort values ascending order (default behaviour)

    >>> s.sort_values(ascending=True).execute()
    1     1.0
    2     3.0
    4     5.0
    3    10.0
    0     NaN
    dtype: float64

    Sort values descending order

    >>> s.sort_values(ascending=False).execute()
    3    10.0
    4     5.0
    2     3.0
    1     1.0
    0     NaN
    dtype: float64

    Sort values inplace

    >>> s.sort_values(ascending=False, inplace=True)
    >>> s.execute()
    3    10.0
    4     5.0
    2     3.0
    1     1.0
    0     NaN
    dtype: float64

    Sort values putting NAs first
    """
    if na_position not in ['last', 'first']:  # pragma: no cover
        raise TypeError('invalid na_position: {}'.format(na_position))
    axis = validate_axis(axis, series)
    if axis != 0:
        raise NotImplementedError('Only support sort on axis 0')
    psrs_kinds = _validate_sort_psrs_kinds(psrs_kinds)
    op = DataFrameSortValues(axis=axis, ascending=ascending, inplace=inplace, kind=kind,
                             na_position=na_position, ignore_index=ignore_index,
                             parallel_kind=parallel_kind, psrs_kinds=psrs_kinds,
                             object_type=ObjectType.series)
    sorted_series = op(series)
    if inplace:
        series.data = sorted_series.data
    else:
        return sorted_series
