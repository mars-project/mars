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

import os

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...core.operand import OperandStage, MapReduceOperand
from ...utils import lazy_import
from ...serialization.serializables import Int32Field, ListField, StringField, BoolField
from ...tensor.base.psrs import PSRSOperandMixin
from ..core import IndexValue, OutputType
from ..utils import standardize_range_index, parse_index, is_cudf
from ..operands import DataFrameOperandMixin, DataFrameOperand, \
    DataFrameShuffleProxy


cudf = lazy_import('cudf', globals=globals())

_PSRS_DISTINCT_COL = '__PSRS_TMP_DISTINCT_COL'


class _Largest:
    """
    This util class resolve TypeError when
    comparing strings with None values
    """
    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return self is not other


_largest = _Largest()


class DataFramePSRSOperandMixin(DataFrameOperandMixin, PSRSOperandMixin):
    @classmethod
    def _collect_op_properties(cls, op):
        from .sort_values import DataFrameSortValues
        if isinstance(op, DataFrameSortValues):
            properties = dict(sort_type='sort_values', axis=op.axis, by=op.by, ascending=op.ascending,
                              inplace=op.inplace, na_position=op.na_position, gpu=op.is_gpu())
        else:
            properties = dict(sort_type='sort_index', axis=op.axis, level=op.level, ascending=op.ascending,
                              inplace=op.inplace, na_position=op.na_position, sort_remaining=op.sort_remaining,
                              gpu=op.is_gpu())
        return properties

    @classmethod
    def local_sort_and_regular_sample(cls, op, in_data, axis_chunk_shape, axis_offsets, out_idx):
        # stage 1: local sort and regular samples collected
        sorted_chunks, indices_chunks, sampled_chunks = [], [], []
        for i in range(axis_chunk_shape):
            in_chunk = in_data.chunks[i]
            kind = None if op.psrs_kinds is None else op.psrs_kinds[0]
            chunk_op = DataFramePSRSSortRegularSample(kind=kind, n_partition=axis_chunk_shape,
                                                      output_types=op.output_types,
                                                      **cls._collect_op_properties(op))
            kws = []
            sort_shape = in_chunk.shape
            kws.append({'shape': sort_shape,
                        'index_value': in_chunk.index_value,
                        'index': in_chunk.index})
            if chunk_op.sort_type == 'sort_values':
                sampled_shape = (axis_chunk_shape, len(op.by)) if \
                    op.by else (axis_chunk_shape,)
            else:
                sampled_shape = (axis_chunk_shape, sort_shape[1]) if\
                    len(sort_shape) == 2 else (axis_chunk_shape,)
            kws.append({'shape': sampled_shape,
                        'index_value': in_chunk.index_value,
                        'index': (i,),
                        'type': 'regular_sampled'})
            if op.outputs[0].ndim == 2:
                kws[0].update({'columns_value': in_chunk.columns_value, 'dtypes': in_chunk.dtypes})
                kws[1].update({'columns_value': in_chunk.columns_value, 'dtypes': in_chunk.dtypes})
            else:
                kws[0].update(({'dtype': in_chunk.dtype, 'name': in_chunk.name}))
                kws[1].update({'dtype': in_chunk.dtype})

            chunks = chunk_op.new_chunks([in_chunk], kws=kws, output_limit=len(kws))
            sort_chunk, sampled_chunk = chunks
            sorted_chunks.append(sort_chunk)
            sampled_chunks.append(sampled_chunk)
        return sorted_chunks, indices_chunks, sampled_chunks

    @classmethod
    def concat_and_pivot(cls, op, axis_chunk_shape, out_idx, sorted_chunks, sampled_chunks):
        from .sort_values import DataFrameSortValues

        # stage 2: gather and merge samples, choose and broadcast p-1 pivots
        kind = None if op.psrs_kinds is None else op.psrs_kinds[1]
        if isinstance(op, DataFrameSortValues):
            output_types = op.output_types
        else:
            output_types = [OutputType.index]
        concat_pivot_op = DataFramePSRSConcatPivot(kind=kind, n_partition=axis_chunk_shape,
                                                   output_types=output_types,
                                                   **cls._collect_op_properties(op))
        concat_pivot_shape = \
            sorted_chunks[0].shape[:op.axis] + (axis_chunk_shape - 1,) + \
            sorted_chunks[0].shape[op.axis + 1:]
        concat_pivot_index = out_idx[:op.axis] + (0,) + out_idx[op.axis:]
        concat_pivot_chunk = concat_pivot_op.new_chunk(sampled_chunks,
                                                       shape=concat_pivot_shape,
                                                       index=concat_pivot_index,
                                                       output_type=output_types[0])
        return concat_pivot_chunk

    @classmethod
    def partition_local_data(cls, op, axis_chunk_shape, sorted_chunks,
                             indices_chunks, concat_pivot_chunk):
        # stage 3: Local data is partitioned
        partition_chunks = []
        length = len(sorted_chunks)
        for i in range(length):
            chunk_inputs = [sorted_chunks[i], concat_pivot_chunk]
            partition_shuffle_map = DataFramePSRSShuffle(n_partition=axis_chunk_shape,
                                                         stage=OperandStage.map,
                                                         output_types=op.output_types,
                                                         **cls._collect_op_properties(op))
            if isinstance(chunk_inputs[0].index_value.value, IndexValue.RangeIndex):
                index_value = parse_index(pd.Int64Index([]))
            else:
                index_value = chunk_inputs[0].index_value
            kw = dict(shape=chunk_inputs[0].shape,
                      index=chunk_inputs[0].index,
                      index_value=index_value)
            if op.outputs[0].ndim == 2:
                kw.update(dict(columns_value=chunk_inputs[0].columns_value,
                               dtypes=chunk_inputs[0].dtypes))
            else:
                kw.update(dict(dtype=chunk_inputs[0].dtype, name=chunk_inputs[0].name))
            partition_chunk = partition_shuffle_map.new_chunk(chunk_inputs, **kw)
            partition_chunks.append(partition_chunk)
        return partition_chunks

    @classmethod
    def partition_merge_data(cls, op, need_align, return_value, partition_chunks, proxy_chunk):
        # stage 4: all *ith* classes are gathered and merged
        partition_sort_chunks, partition_indices_chunks, sort_info_chunks = [], [], []
        for i, partition_chunk in enumerate(partition_chunks):
            kind = None if op.psrs_kinds is None else op.psrs_kinds[2]
            partition_shuffle_reduce = DataFramePSRSShuffle(
                stage=OperandStage.reduce, kind=kind, reducer_index=(i,),
                output_types=op.output_types, **cls._collect_op_properties(op))
            chunk_shape = list(partition_chunk.shape)
            chunk_shape[op.axis] = np.nan

            kw = dict(shape=tuple(chunk_shape), index=partition_chunk.index,
                      index_value=partition_chunk.index_value)
            if op.outputs[0].ndim == 2:
                kw.update(dict(columns_value=partition_chunk.columns_value,
                               dtypes=partition_chunk.dtypes))
            else:
                kw.update(dict(dtype=partition_chunk.dtype, name=partition_chunk.name))
            cs = partition_shuffle_reduce.new_chunks([proxy_chunk], **kw)

            partition_sort_chunks.append(cs[0])
        return partition_sort_chunks, partition_indices_chunks, sort_info_chunks

    @classmethod
    def _tile_psrs(cls, op, in_data):
        out = op.outputs[0]
        in_df, axis_chunk_shape, _, _ = yield from cls.preprocess(op, in_data=in_data)

        # stage 1: local sort and regular samples collected
        sorted_chunks, _, sampled_chunks = cls.local_sort_and_regular_sample(
            op, in_df, axis_chunk_shape, None, None)

        # stage 2: gather and merge samples, choose and broadcast p-1 pivots
        concat_pivot_chunk = cls.concat_and_pivot(
            op, axis_chunk_shape, (), sorted_chunks, sampled_chunks)

        # stage 3: Local data is partitioned
        partition_chunks = cls.partition_local_data(
            op, axis_chunk_shape, sorted_chunks, None, concat_pivot_chunk)

        proxy_chunk = DataFrameShuffleProxy(output_types=op.output_types).new_chunk(
            partition_chunks, shape=())

        # stage 4: all *ith* classes are gathered and merged
        partition_sort_chunks = cls.partition_merge_data(
            op, False, None, partition_chunks, proxy_chunk)[0]

        if op.ignore_index:
            chunks = standardize_range_index(partition_sort_chunks, axis=op.axis)
        else:
            chunks = partition_sort_chunks

        if op.outputs[0].ndim == 2:
            nsplits = ((np.nan,) * len(chunks), (out.shape[1],))
            new_op = op.copy()
            return new_op.new_dataframes(op.inputs, shape=out.shape, chunks=chunks,
                                         nsplits=nsplits, index_value=out.index_value,
                                         columns_value=out.columns_value, dtypes=out.dtypes)
        else:
            nsplits = ((np.nan,) * len(chunks), )
            new_op = op.copy()
            return new_op.new_seriess(op.inputs, shape=out.shape, chunks=chunks,
                                      nsplits=nsplits, index_value=out.index_value,
                                      dtype=out.dtype, name=out.name)


def execute_sort_values(data, op, inplace=None, by=None):
    if inplace is None:
        inplace = op.inplace
    # ignore_index is new in Pandas version 1.0.0.
    ignore_index = getattr(op, 'ignore_index', False)
    if isinstance(data, (pd.DataFrame, pd.Series)):
        kwargs = dict(axis=op.axis, ascending=op.ascending, ignore_index=ignore_index,
                      na_position=op.na_position, kind=op.kind)
        if isinstance(data, pd.DataFrame):
            kwargs['by'] = by if by is not None else op.by
        if inplace:
            kwargs['inplace'] = True
            try:
                data.sort_values(**kwargs)
            except TypeError:  # pragma: no cover
                kwargs.pop('ignore_index', None)
                data.sort_values(**kwargs)
            return data
        else:
            try:
                return data.sort_values(**kwargs)
            except TypeError:  # pragma: no cover
                kwargs.pop('ignore_index', None)
                return data.sort_values(**kwargs)

    else:  # pragma: no cover
        # cudf doesn't support axis and kind
        if isinstance(data, cudf.DataFrame):
            return data.sort_values(
                op.by, ascending=op.ascending, na_position=op.na_position)
        else:
            return data.sort_values(
                ascending=op.ascending, na_position=op.na_position)


def execute_sort_index(data, op, inplace=None):
    if inplace is None:
        inplace = op.inplace
    # ignore_index is new in Pandas version 1.0.0.
    ignore_index = getattr(op, 'ignore_index', False)
    if isinstance(data, (pd.DataFrame, pd.Series)):
        kwargs = dict(level=op.level, ascending=op.ascending, ignore_index=ignore_index,
                      na_position=op.na_position, kind=op.kind, sort_remaining=op.sort_remaining)
        if inplace:
            kwargs['inplace'] = True
            try:
                data.sort_index(**kwargs)
            except TypeError:  # pragma: no cover
                kwargs.pop('ignore_index', None)
                data.sort_index(**kwargs)
            return data
        else:
            try:
                return data.sort_index(**kwargs)
            except TypeError:  # pragma: no cover
                kwargs.pop('ignore_index', None)
                return data.sort_index(**kwargs)

    else:  # pragma: no cover
        # cudf only support ascending
        return data.sort_index(ascending=op.ascending)


class DataFramePSRSChunkOperand(DataFrameOperand):
    # sort type could be 'sort_values' or 'sort_index'
    _sort_type = StringField('sort_type')

    _axis = Int32Field('axis')
    _by = ListField('by')
    _ascending = BoolField('ascending')
    _inplace = BoolField('inplace')
    _kind = StringField('kind')
    _na_position = StringField('na_position')

    # for sort_index
    _level = ListField('level')
    _sort_remaining = BoolField('sort_remaining')

    _n_partition = Int32Field('n_partition')

    def __init__(self, sort_type=None, by=None, axis=None, ascending=None, inplace=None, kind=None,
                 na_position=None, level=None, sort_remaining=None, n_partition=None,
                 output_types=None, **kw):
        super().__init__(_sort_type=sort_type, _by=by, _axis=axis, _ascending=ascending,
                         _inplace=inplace, _kind=kind, _na_position=na_position,
                         _level=level, _sort_remaining=sort_remaining, _n_partition=n_partition,
                         _output_types=output_types, **kw)

    @property
    def sort_type(self):
        return self._sort_type

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
    def level(self):
        return self._level

    @property
    def sort_remaining(self):
        return self._sort_remaining

    @property
    def n_partition(self):
        return self._n_partition


class DataFramePSRSSortRegularSample(DataFramePSRSChunkOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.PSRS_SORT_REGULAR_SMAPLE

    @property
    def output_limit(self):
        return 2

    @classmethod
    def execute(cls, ctx, op):
        a = ctx[op.inputs[0].key]
        xdf = pd if isinstance(a, (pd.DataFrame, pd.Series)) else cudf

        if len(a) == 0:
            # when chunk is empty, return the empty chunk itself
            ctx[op.outputs[0].key] = ctx[op.outputs[-1].key] = a
            return

        if op.sort_type == 'sort_values':
            ctx[op.outputs[0].key] = res = execute_sort_values(a, op)
        else:
            ctx[op.outputs[0].key] = res = execute_sort_index(a, op)

        by = op.by
        add_distinct_col = bool(int(os.environ.get('PSRS_DISTINCT_COL', '0')))
        if add_distinct_col and isinstance(a, xdf.DataFrame) and op.sort_type == 'sort_values':
            # when running under distributed mode, we introduce an extra column
            # to make sure pivots are distinct
            chunk_idx = op.inputs[0].index[0]
            distinct_col = _PSRS_DISTINCT_COL if a.columns.nlevels == 1 \
                else (_PSRS_DISTINCT_COL,) + ('',) * (a.columns.nlevels - 1)
            res[distinct_col] = np.arange(chunk_idx << 32, (chunk_idx << 32) + len(a), dtype=np.int64)
            by = list(by) + [distinct_col]

        n = op.n_partition
        if op.sort_type == 'sort_values' and a.shape[op.axis] < n:
            num = n // a.shape[op.axis] + 1
            res = execute_sort_values(xdf.concat([res] * num), op, by=by)

        w = res.shape[op.axis] * 1.0 / (n + 1)
        slc = np.linspace(max(w - 1, 0), res.shape[op.axis] - 1,
                          num=n, endpoint=False).astype(int)
        if op.axis == 1:
            slc = (slice(None), slc)
        if op.sort_type == 'sort_values':
            # do regular sample
            if op.by is not None:
                ctx[op.outputs[-1].key] = res[by].iloc[slc]
            else:
                ctx[op.outputs[-1].key] = res.iloc[slc]
        else:
            # do regular sample
            ctx[op.outputs[-1].key] = res.iloc[slc]


class DataFramePSRSConcatPivot(DataFramePSRSChunkOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.PSRS_CONCAT_PIVOT

    @property
    def output_limit(self):
        return 1

    @classmethod
    def execute(cls, ctx, op):
        inputs = [ctx[c.key] for c in op.inputs if len(ctx[c.key]) > 0]
        if len(inputs) == 0:
            # corner case: nothing sampled, we need to do nothing
            ctx[op.outputs[-1].key] = ctx[op.inputs[0].key]
            return

        xdf = pd if isinstance(inputs[0], (pd.DataFrame, pd.Series)) else cudf

        a = xdf.concat(inputs, axis=op.axis)
        p = len(inputs)
        assert a.shape[op.axis] == p * len(op.inputs)

        slc = np.linspace(p - 1, a.shape[op.axis] - 1,
                          num=len(op.inputs) - 1, endpoint=False).astype(int)
        if op.axis == 1:
            slc = (slice(None), slc)
        if op.sort_type == 'sort_values':
            a = execute_sort_values(a, op, inplace=False)
            ctx[op.outputs[-1].key] = a.iloc[slc]
        else:
            a = execute_sort_index(a, op, inplace=False)
            ctx[op.outputs[-1].key] = a.index[slc]


class DataFramePSRSShuffle(MapReduceOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.PSRS_SHUFFLE

    _sort_type = StringField('sort_type')

    # for shuffle map
    _axis = Int32Field('axis')
    _by = ListField('by')
    _ascending = BoolField('ascending')
    _inplace = BoolField('inplace')
    _na_position = StringField('na_position')
    _n_partition = Int32Field('n_partition')

    # for sort_index
    _level = ListField('level')
    _sort_remaining = BoolField('sort_remaining')

    # for shuffle reduce
    _kind = StringField('kind')

    def __init__(self, sort_type=None, by=None, axis=None, ascending=None, n_partition=None,
                 na_position=None, inplace=None, kind=None, level=None, sort_remaining=None,
                 output_types=None, **kw):
        super().__init__(_sort_type=sort_type, _by=by, _axis=axis, _ascending=ascending,
                         _n_partition=n_partition, _na_position=na_position, _inplace=inplace,
                         _kind=kind, _level=level, _sort_remaining=sort_remaining,
                         _output_types=output_types, **kw)

    @property
    def sort_type(self):
        return self._sort_type

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
    def level(self):
        return self._level

    @property
    def sort_remaining(self):
        return self._sort_remaining

    @property
    def n_partition(self):
        return self._n_partition

    @property
    def kind(self):
        return self._kind

    @property
    def output_limit(self):
        return 1

    @staticmethod
    def _calc_poses(src_cols, pivots, ascending=True):
        records = src_cols.to_records(index=False)
        p_records = pivots.to_records(index=False)
        if ascending:
            poses = records.searchsorted(p_records, side='right')
        else:
            poses = len(records) - records[::-1].searchsorted(p_records, side='right')
        del records, p_records
        return poses

    @classmethod
    def _execute_dataframe_map(cls, ctx, op):
        a, pivots = [ctx[c.key] for c in op.inputs]
        out = op.outputs[0]

        if len(a) == 0:
            # when the chunk is empty, no slices can be produced
            for i in range(op.n_partition):
                ctx[out.key, (i,)] = a
            return

        # use numpy.searchsorted to find split positions.
        by = op.by

        distinct_col = _PSRS_DISTINCT_COL if a.columns.nlevels == 1 \
            else (_PSRS_DISTINCT_COL,) + ('',) * (a.columns.nlevels - 1)
        if distinct_col in a.columns:
            by = list(by) + [distinct_col]

        try:
            poses = cls._calc_poses(a[by], pivots, op.ascending)
        except TypeError:
            poses = cls._calc_poses(
                a[by].fillna(_largest), pivots.fillna(_largest), op.ascending)

        poses = (None,) + tuple(poses) + (None,)
        for i in range(op.n_partition):
            values = a.iloc[poses[i]: poses[i + 1]]
            if is_cudf(values):  # pragma: no cover
                values = values.copy()
            ctx[out.key, (i,)] = values

    @classmethod
    def _calc_series_poses(cls, s, pivots, ascending=True):
        if ascending:
            poses = s.searchsorted(pivots, side='right')
        else:
            poses = len(s) - s.iloc[::-1].searchsorted(pivots, side='right')
        return poses

    @classmethod
    def _execute_series_map(cls, ctx, op):
        a, pivots = [ctx[c.key] for c in op.inputs]
        out = op.outputs[0]

        if len(a) == 0:
            # when the chunk is empty, no slices can be produced
            for i in range(op.n_partition):
                ctx[out.key, (i,)] = a
            return

        if isinstance(a, pd.Series):
            try:
                poses = cls._calc_series_poses(a, pivots, ascending=op.ascending)
            except TypeError:
                filled_a = a.fillna(_largest)
                filled_pivots = pivots.fillna(_largest)
                poses = cls._calc_series_poses(filled_a, filled_pivots, ascending=op.ascending)
            poses = (None,) + tuple(poses) + (None,)
            for i in range(op.n_partition):
                values = a.iloc[poses[i]: poses[i + 1]]
                ctx[out.key, (i,)] = values

    @classmethod
    def _execute_sort_index_map(cls, ctx, op):
        a, pivots = [ctx[c.key] for c in op.inputs]
        out = op.outputs[0]

        if op.ascending:
            poses = a.index.searchsorted(list(pivots), side='right')
        else:
            poses = len(a) - a.index[::-1].searchsorted(list(pivots), side='right')
        poses = (None,) + tuple(poses) + (None,)
        for i in range(op.n_partition):
            values = a.iloc[poses[i]: poses[i + 1]]
            ctx[out.key, (i,)] = values

    @classmethod
    def _execute_map(cls, ctx, op):
        a = [ctx[c.key] for c in op.inputs][0]
        if op.sort_type == 'sort_values':
            if len(a.shape) == 2:
                # DataFrame type
                cls._execute_dataframe_map(ctx, op)
            else:
                # Series type
                cls._execute_series_map(ctx, op)
        else:
            cls._execute_sort_index_map(ctx, op)

    @classmethod
    def _execute_reduce(cls, ctx, op: "DataFramePSRSShuffle"):
        out_chunk = op.outputs[0]
        raw_inputs = list(op.iter_mapper_data(ctx, pop=False))

        xdf = pd if isinstance(raw_inputs[0], (pd.DataFrame, pd.Series)) else cudf
        if xdf is pd:
            concat_values = xdf.concat(raw_inputs, axis=op.axis, copy=False)
        else:
            concat_values = xdf.concat(raw_inputs, axis=op.axis)
        del raw_inputs[:]

        if isinstance(concat_values, xdf.DataFrame):
            concat_values.drop(_PSRS_DISTINCT_COL, axis=1, inplace=True, errors='ignore')

            col_index_dtype = out_chunk.columns_value.to_pandas().dtype
            if concat_values.columns.dtype != col_index_dtype:
                concat_values.columns = concat_values.columns.astype(col_index_dtype)

        if op.sort_type == 'sort_values':
            ctx[op.outputs[0].key] = execute_sort_values(concat_values, op)
        else:
            ctx[op.outputs[0].key] = execute_sort_index(concat_values, op)

    @classmethod
    def estimate_size(cls, ctx, op):
        super().estimate_size(ctx, op)
        result = ctx[op.outputs[0].key]
        if op.stage == OperandStage.reduce:
            ctx[op.outputs[0].key] = (result[0], result[1] * 1.5)
        else:
            ctx[op.outputs[0].key] = result

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        else:
            cls._execute_reduce(ctx, op)
