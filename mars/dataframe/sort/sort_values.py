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

import numpy as np
import pandas as pd

from ...serialize import Int32Field, StringField, ListField, BoolField, ValueType
from ... import opcodes as OperandDef
from ...tensor.base.sort import _validate_sort_psrs_kinds
from ..utils import parse_index, standardize_range_index, validate_axis, build_concated_rows_frame
from ..operands import DataFrameOperand, DataFrameShuffleProxy, ObjectType
from .psrs import DataFramePSRSOperandMixin, sort_dataframe


class DataFrameSortValues(DataFrameOperand, DataFramePSRSOperandMixin):
    _op_type_ = OperandDef.SORT_VALUES

    _by = ListField('by', ValueType.string)
    _axis = Int32Field('axis')
    _ascending = BoolField('ascending')
    _inplace = BoolField('inplace')
    _kind = StringField('kind')
    _na_position = StringField('na_position')
    _ignore_index = BoolField('ignore_index')
    _parallel_kind = StringField('parallel_kind')
    _psrs_kinds = ListField('psrs_kinds', ValueType.string)

    def __init__(self, by=None, axis=None, ascending=None, inplace=None, kind=None,
                 na_position=None, ignore_index=None, parallel_kind=None, psrs_kinds=None, **kw):
        super(DataFrameSortValues, self).__init__(_by=by, _axis=axis, _ascending=ascending,
                                                  _inplace=inplace, _kind=kind,
                                                  _na_position=na_position,
                                                  _ignore_index=ignore_index,
                                                  _parallel_kind=parallel_kind,
                                                  _psrs_kinds=psrs_kinds,
                                                  _object_type=ObjectType.dataframe, **kw)

    @property
    def by(self):
        return self._by

    @property
    def axis(self):
        return self._axis

    @property
    def ascending(self):
        return self._ascending

    @property
    def inplace(self):
        return self._inplace

    @property
    def kind(self):
        return self._kind

    @property
    def na_position(self):
        return self._na_position

    @property
    def ignore_index(self):
        return self._ignore_index

    @property
    def parallel_kind(self):
        return self._parallel_kind

    @property
    def psrs_kinds(self):
        return self._psrs_kinds

    @classmethod
    def _tile_psrs(cls, op, in_data):
        out = op.outputs[0]
        in_df, axis_chunk_shape, _, _ = cls.preprocess(op, in_data=in_data)

        # stage 1: local sort and regular samples collected
        sorted_chunks, _, sampled_chunks = cls.local_sort_and_regular_sample(
            op, in_df, axis_chunk_shape, None, None)

        # stage 2: gather and merge samples, choose and broadcast p-1 pivots
        concat_pivot_chunk = cls.concat_and_pivot(
            op, axis_chunk_shape, (), sorted_chunks, sampled_chunks)

        # stage 3: Local data is partitioned
        partition_chunks = cls.partition_local_data(
            op, axis_chunk_shape, sorted_chunks, None, concat_pivot_chunk)

        proxy_chunk = DataFrameShuffleProxy(object_type=ObjectType.dataframe).new_chunk(
            partition_chunks, shape=())

        # stage 4: all *ith* classes are gathered and merged
        partition_sort_chunks = cls.partition_merge_data(
            op, False, None, partition_chunks, proxy_chunk)[0]

        if op.ignore_index:
            chunks = standardize_range_index(partition_sort_chunks, axis=op.axis)
        else:
            chunks = partition_sort_chunks

        if op.axis == 0:
            nsplits = ((np.nan,) * len(chunks), (out.shape[1],))
        else:
            nsplits = ((out.shape[0],), (np.nan,) * len(chunks))
        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, shape=out.shape, chunks=chunks,
                                     nsplits=nsplits, index_value=out.index_value,
                                     columns_value=out.columns_value, dtypes=out.dtypes)

    @classmethod
    def tile(cls, op):
        df = op.inputs[0]

        df = build_concated_rows_frame(df)

        if df.chunk_shape[op.axis] == 1:
            out_chunks = []
            for chunk in df.chunks:
                chunk_op = op.copy().reset_key()
                out_chunks.append(chunk_op.new_chunk(
                    [chunk], shape=chunk.shape, index=chunk.index, index_value=chunk.index_value,
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
    def execute(cls, ctx, op):
        in_data = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = sort_dataframe(in_data, op)

    def __call__(self, df):
        assert self.axis == 0
        if self.ignore_index:
            index_value = parse_index(pd.RangeIndex(df.shape[0]))
        else:
            index_value = df.index_value
        return self.new_dataframe([df], shape=df.shape, dtypes=df.dtypes,
                                  index_value=index_value,
                                  columns_value=df.columns_value)


def sort_values(df, by, axis=0, ascending=True, inplace=False, kind='quicksort',
                na_position='last', ignore_index=False, parallel_kind='PSRS', psrs_kinds=None):
    """
    Sort by the values along either axis.
    :param df: input data.
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
                             psrs_kinds=psrs_kinds)
    sorted_df = op(df)
    if inplace:
        df.data = sorted_df.data
    else:
        return sorted_df
