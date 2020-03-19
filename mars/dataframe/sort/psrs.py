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

from ... import opcodes as OperandDef
from ...utils import lazy_import, get_shuffle_input_keys_idxes
from ...operands import OperandStage
from ...serialize import ValueType, Int32Field, ListField, StringField, BoolField
from ...tensor.base.psrs import PSRSOperandMixin
from ..utils import parse_index
from ..operands import DataFrameOperandMixin, DataFrameOperand,\
    ObjectType, DataFrameMapReduceOperand


cudf = lazy_import('cudf', globals=globals())


class DataFramePSRSOperandMixin(DataFrameOperandMixin, PSRSOperandMixin):
    @classmethod
    def local_sort_and_regular_sample(cls, op, in_data, axis_chunk_shape, axis_offsets, out_idx):
        # stage 1: local sort and regular samples collected
        sorted_chunks, indices_chunks, sampled_chunks = [], [], []
        for i in range(axis_chunk_shape):
            in_chunk = in_data.chunks[i]
            kind = None if op.psrs_kinds is None else op.psrs_kinds[0]
            chunk_op = DataFramePSRSSortRegularSample(
                axis=op.axis, by=op.by, ascending=op.ascending,
                inplace=op.inplace, kind=kind, na_position=op.na_position,
                n_partition=axis_chunk_shape)
            kws = []
            sort_shape = in_chunk.shape
            kws.append({'shape': sort_shape,
                        'index_value': in_chunk.index_value,
                        'columns_value': in_chunk.columns_value,
                        'index': in_chunk.index,
                        'dtypes': in_chunk.dtypes})
            sampled_shape = (axis_chunk_shape, len(op.by))
            kws.append({'shape': sampled_shape,
                        'index_value': in_chunk.index_value,
                        'columns_value': parse_index(pd.Index(op.by), store_data=True),
                        'dtypes': in_chunk.dtypes,
                        'index': (i,),
                        'type': 'regular_sampled'})
            chunks = chunk_op.new_chunks([in_chunk], kws=kws, output_limit=len(kws))
            sort_chunk, sampled_chunk = chunks
            sorted_chunks.append(sort_chunk)
            sampled_chunks.append(sampled_chunk)
        return sorted_chunks, indices_chunks, sampled_chunks

    @classmethod
    def concat_and_pivot(cls, op, axis_chunk_shape, out_idx, sorted_chunks, sampled_chunks):
        # stage 2: gather and merge samples, choose and broadcast p-1 pivots
        concat_pivot_op = DataFramePSRSConcatPivot(
            by=op.by, axis=op.axis, ascending=op.ascending, na_position=op.na_position,
            kind=None if op.psrs_kinds is None else op.psrs_kinds[1])
        concat_pivot_shape = \
            sorted_chunks[0].shape[:op.axis] + (axis_chunk_shape - 1,) + \
            sorted_chunks[0].shape[op.axis + 1:]
        concat_pivot_index = out_idx[:op.axis] + (0,) + out_idx[op.axis:]
        concat_pivot_chunk = concat_pivot_op.new_chunk(sampled_chunks,
                                                       shape=concat_pivot_shape,
                                                       index=concat_pivot_index)
        return concat_pivot_chunk

    @classmethod
    def partition_local_data(cls, op, axis_chunk_shape, sorted_chunks,
                             indices_chunks, concat_pivot_chunk):
        # stage 3: Local data is partitioned
        partition_chunks = []
        length = len(sorted_chunks)
        for i in range(length):
            chunk_inputs = [sorted_chunks[i], concat_pivot_chunk]
            partition_shuffle_map = DataFramePSRSShuffle(
                by=op.by, stage=OperandStage.map, axis=op.axis, ascending=op.ascending,
                na_position=op.na_position, n_partition=axis_chunk_shape)
            partition_chunk = partition_shuffle_map.new_chunk(chunk_inputs,
                                                              shape=chunk_inputs[0].shape,
                                                              index=chunk_inputs[0].index,
                                                              index_value=chunk_inputs[0].index_value,
                                                              columns_value=chunk_inputs[0].columns_value,
                                                              dtypes=chunk_inputs[0].dtypes)
            partition_chunks.append(partition_chunk)
        return partition_chunks

    @classmethod
    def partition_merge_data(cls, op, need_align, return_value, partition_chunks, proxy_chunk):
        # stage 4: all *ith* classes are gathered and merged
        partition_sort_chunks, partition_indices_chunks, sort_info_chunks = [], [], []
        for i, partition_chunk in enumerate(partition_chunks):
            kind = None if op.psrs_kinds is None else op.psrs_kinds[2]
            partition_shuffle_reduce = DataFramePSRSShuffle(
                stage=OperandStage.reduce, axis=op.axis, by=op.by, ascending=op.ascending,
                inplace=op.inplace, na_position=op.na_position, kind=kind, shuffle_key=str(i))
            chunk_shape = list(partition_chunk.shape)
            chunk_shape[op.axis] = np.nan

            cs = partition_shuffle_reduce.new_chunks(
                [proxy_chunk], shape=tuple(chunk_shape), index=partition_chunk.index,
                index_value=partition_chunk.index_value, columns_value=partition_chunk.columns_value,
                dtypes=partition_chunk.dtypes)

            partition_sort_chunks.append(cs[0])
        return partition_sort_chunks, partition_indices_chunks, sort_info_chunks


def sort_dataframe(df, op, inplace=None):
    if inplace is None:
        inplace = op.inplace
    # ignore_index is new in Pandas version 1.0.0.
    ignore_index = getattr(op, 'ignore_index', False)
    if isinstance(df, pd.DataFrame):
        if inplace:
            try:
                df.sort_values(op.by, axis=op.axis, ascending=op.ascending, ignore_index=ignore_index,
                               inplace=True, na_position=op.na_position, kind=op.kind)
            except TypeError:  # pragma: no cover
                df.sort_values(op.by, axis=op.axis, ascending=op.ascending, inplace=True,
                               na_position=op.na_position, kind=op.kind)
            return df
        else:
            try:
                return df.sort_values(op.by, axis=op.axis, ascending=op.ascending, ignore_index=ignore_index,
                                      inplace=False, na_position=op.na_position, kind=op.kind)
            except TypeError:  # pragma: no cover
                return df.sort_values(op.by, axis=op.axis, ascending=op.ascending,
                                      inplace=False, na_position=op.na_position, kind=op.kind)

    else:  # pragma: no cover
        # cudf doesn't support axis and kind
        return df.sort_values(
            op.by, ascending=op.ascending, na_position=op.na_position)


class DataFramePSRSSortRegularSample(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.PSRS_SORT_REGULAR_SMAPLE

    _axis = Int32Field('axis')
    _by = ListField('by', ValueType.string)
    _ascending = BoolField('ascending')
    _inplace = BoolField('inplace')
    _kind = StringField('kind')
    _na_position = StringField('na_position')
    _n_partition = Int32Field('n_partition')

    def __init__(self, by=None, axis=None, ascending=None, inplace=None, kind=None,
                 na_position=None,  n_partition=None, **kw):
        super(DataFramePSRSSortRegularSample, self).__init__(
            _by=by, _axis=axis, _ascending=ascending, _inplace=inplace,
            _kind=kind, _na_position=na_position, _n_partition=n_partition,
            _object_type=ObjectType.dataframe, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def by(self):
        return self._by

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
    def n_partition(self):
        return self._n_partition

    @property
    def output_limit(self):
        return 2

    @classmethod
    def execute(cls, ctx, op):
        a = ctx[op.inputs[0].key]

        n = op.n_partition
        w = int(a.shape[op.axis] // n)

        ctx[op.outputs[0].key] = res = sort_dataframe(a, op)
        # do regular sample
        slc = (slice(None),) * op.axis + (slice(0, n * w, w),)
        ctx[op.outputs[-1].key] = res[op.by].iloc[slc]


class DataFramePSRSConcatPivot(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.PSRS_CONCAT_PIVOT

    _by = ListField('by', ValueType.string)
    _axis = Int32Field('axis')
    _ascending = BoolField('ascending')
    _na_position = StringField('na_position')
    _kind = StringField('kind')

    def __init__(self, by=None, axis=None, ascending=None, na_position=None, kind=None, **kw):
        super(DataFramePSRSConcatPivot, self).__init__(_by=by, _axis=axis, _ascending=ascending,
                                                       _na_position=na_position, _kind=kind,
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
    def na_position(self):
        return self._na_position

    @property
    def kind(self):
        return self._kind

    @classmethod
    def execute(cls, ctx, op):
        inputs = [ctx[c.key] for c in op.inputs]
        xdf = pd if isinstance(inputs[0], (pd.DataFrame, pd.Series)) else cudf

        a = xdf.concat(inputs, axis=op.axis)
        p = len(inputs)
        assert a.shape[op.axis] == p ** 2

        a = sort_dataframe(a, op, inplace=False)

        select = slice(p - 1, (p - 1) ** 2 + 1, p - 1)
        slc = (slice(None),) * op.axis + (select,)
        ctx[op.outputs[-1].key] = a.iloc[slc]


class DataFramePSRSShuffle(DataFrameMapReduceOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.PSRS_SHUFFLE

    # for shuffle map
    _axis = Int32Field('axis')
    _by = ListField('by', ValueType.string)
    _ascending = BoolField('ascending')
    _inplace = BoolField('inplace')
    _na_position = StringField('na_position')
    _n_partition = Int32Field('n_partition')

    # for shuffle reduce
    _kind = StringField('kind')

    def __init__(self, by=None, axis=None, ascending=None, n_partition=None, na_position=None,
                 inplace=None, kind=None, stage=None, shuffle_key=None, **kw):
        super(DataFramePSRSShuffle, self).__init__(
            _by=by, _axis=axis, _ascending=ascending, _n_partition=n_partition,
            _na_position=na_position, _inplace=inplace, _kind=kind, _stage=stage,
            _shuffle_key=shuffle_key, _object_type=ObjectType.dataframe, **kw)

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
    def na_position(self):
        return self._na_position

    @property
    def n_partition(self):
        return self._n_partition

    @property
    def kind(self):
        return self._kind

    @property
    def output_limit(self):
        return 1

    @classmethod
    def _execute_map(cls, ctx, op):
        a, pivots = [ctx[c.key] for c in op.inputs]
        out = op.outputs[0]

        if isinstance(a, pd.DataFrame):
            # use numpy.searchsorted to find split positions.
            records = a[op.by].to_records(index=False)
            p_records = pivots.to_records(index=False)
            if op.ascending:
                poses = records.searchsorted(p_records, side='right')
            else:
                poses = len(records) - records[::-1].searchsorted(p_records, side='right')

            poses = (None,) + tuple(poses) + (None,)
            for i in range(op.n_partition):
                values = a.iloc[poses[i]: poses[i + 1]]
                ctx[(out.key, str(i))] = values
        else:  # pragma: no cover
            # for cudf, find split positions in loops.
            if op.ascending:
                pivots.append(a.iloc[-1][op.by])
                for i in range(op.n_partition):
                    selected = a
                    for label in op.by:
                        selected = selected.loc[a[label] <= pivots.iloc[i][label]]
                    ctx[(out.key, str(i))] = selected
            else:
                pivots.append(a.iloc[-1][op.by])
                for i in range(op.n_partition):
                    selected = a
                    for label in op.by:
                        selected = selected.loc[a[label] >= pivots.iloc[i][label]]
                    ctx[(out.key, str(i))] = selected

    @classmethod
    def _execute_reduce(cls, ctx, op):
        input_keys, _ = get_shuffle_input_keys_idxes(op.inputs[0])
        raw_inputs = [ctx[(input_key, op.shuffle_key)] for input_key in input_keys]
        xdf = pd if isinstance(raw_inputs[0], pd.DataFrame) else cudf
        concat_values = xdf.concat(raw_inputs, axis=op.axis)
        ctx[op.outputs[0].key] = sort_dataframe(concat_values, op)

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        else:
            cls._execute_reduce(ctx, op)
