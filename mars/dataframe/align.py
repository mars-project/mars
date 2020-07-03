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

import itertools
import operator

import numpy as np

import pandas as pd

from .. import opcodes as OperandDef
from ..operands import OperandStage
from ..serialize import ValueType, AnyField, BoolField, Int32Field, KeyField, ListField
from ..utils import get_shuffle_input_keys_idxes
from .core import SERIES_CHUNK_TYPE
from .utils import hash_dtypes, filter_dtypes
from .operands import DataFrameMapReduceOperand, DataFrameOperandMixin, ObjectType, \
    DataFrameShuffleProxy
from .utils import parse_index, split_monotonic_index_min_max, \
    build_split_idx_to_origin_idx, filter_index_value, hash_index


class DataFrameIndexAlign(DataFrameMapReduceOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_INDEX_ALIGN

    _index_min = AnyField('index_min')
    _index_min_close = BoolField('index_min_close')
    _index_max = AnyField('index_max')
    _index_max_close = BoolField('index_max_close')
    _index_shuffle_size = Int32Field('index_shuffle_size')
    _column_min = AnyField('column_min')
    _column_min_close = BoolField('column_min_close')
    _column_max = AnyField('column_max')
    _column_max_close = BoolField('column_max_close')
    _column_shuffle_size = Int32Field('column_shuffle_size')
    _column_shuffle_segments = ListField('column_shuffle_segments', ValueType.series)

    _input = KeyField('input')

    def __init__(self, index_min_max=None, index_shuffle_size=None, column_min_max=None,
                 column_shuffle_size=None, column_shuffle_segments=None,
                 sparse=None, dtype=None, dtypes=None, gpu=None, stage=None, shuffle_key=None,
                 object_type=None, **kw):
        if index_min_max is not None:
            kw.update(dict(_index_min=index_min_max[0], _index_min_close=index_min_max[1],
                           _index_max=index_min_max[2], _index_max_close=index_min_max[3]))
        if column_min_max is not None:
            kw.update(dict(_column_min=column_min_max[0], _column_min_close=column_min_max[1],
                           _column_max=column_min_max[2], _column_max_close=column_min_max[3]))
        super().__init__(
            _index_shuffle_size=index_shuffle_size, _column_shuffle_size=column_shuffle_size,
            _column_shuffle_segments=column_shuffle_segments, _sparse=sparse,
            _dtype=dtype, _dtypes=dtypes, _gpu=gpu, _stage=stage, _shuffle_key=shuffle_key,
            _object_type=object_type, **kw)

    @property
    def index_min(self):
        return self._index_min

    @property
    def index_min_close(self):
        return self._index_min_close

    @property
    def index_max(self):
        return self._index_max

    @property
    def index_max_close(self):
        return self._index_max_close

    @property
    def index_min_max(self):
        if getattr(self, '_index_min', None) is None:
            return None
        return self._index_min, self._index_min_close, \
            self._index_max, self._index_max_close

    @property
    def index_shuffle_size(self):
        return self._index_shuffle_size

    @property
    def column_min(self):
        return self._column_min

    @property
    def column_min_close(self):
        return self._column_min_close

    @property
    def column_max(self):
        return self._column_max

    @property
    def column_max_close(self):
        return self._column_max_close

    @property
    def column_min_max(self):
        if getattr(self, '_column_min', None) is None:
            return None
        return self._column_min, self._column_min_close, \
            self._column_max, self._column_max_close

    @property
    def column_shuffle_size(self):
        return self._column_shuffle_size

    @property
    def column_shuffle_segments(self):
        return self._column_shuffle_segments

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def _build_map_chunk_kw(self, **kw):
        inputs = self.inputs
        if kw.get('index_value', None) is None and inputs[0].index_value is not None:
            input_index_value = inputs[0].index_value
            index_min_max = self.index_min_max
            if index_min_max is not None:
                kw['index_value'] = filter_index_value(input_index_value, index_min_max)
            else:
                kw['index_value'] = parse_index(inputs[0].index_value.to_pandas(),
                                                input_index_value, type(self).__name__)
        if kw.get('columns_value', None) is None and getattr(inputs[0], 'columns_value', None) is not None:
            input_columns_value = inputs[0].columns_value
            input_dtypes = inputs[0].dtypes
            column_min_max = self.column_min_max
            if column_min_max is not None:
                kw['columns_value'] = filter_index_value(input_columns_value, column_min_max,
                                                         store_data=True)
            else:
                kw['columns_value'] = parse_index(inputs[0].columns_value.to_pandas(), input_columns_value,
                                                  type(self).__name__)
            kw['dtypes'] = input_dtypes[kw['columns_value'].to_pandas()]
            column_shuffle_size = self.column_shuffle_size
            if column_shuffle_size is not None:
                self._column_shuffle_segments = hash_dtypes(input_dtypes, column_shuffle_size)
        if kw.get('dtype', None) and getattr(inputs[0], 'dtype', None) is not None:
            kw['dtype'] = inputs[0].dtype
        if kw.get('name', None) and getattr(inputs[0], 'name', None) is not None:
            kw['name'] = inputs[0].dtype
        return kw

    def _build_reduce_chunk_kw(self, index, **kw):
        inputs = self.inputs
        if kw.get('index_value', None) is None and inputs[0].inputs[0].index_value is not None:
            index_align_map_chunks = inputs[0].inputs
            if index_align_map_chunks[0].op.index_min_max is not None:
                # shuffle on columns, all the DataFrameIndexAlignMap has the same index
                kw['index_value'] = filter_index_value(index_align_map_chunks[0].index_value,
                                                       index_align_map_chunks[0].op.index_min_max)
            else:
                # shuffle on index
                kw['index_value'] = parse_index(index_align_map_chunks[0].index_value.to_pandas(),
                                                [c.key for c in index_align_map_chunks], type(self).__name__)
        if kw.get('columns_value', None) is None and getattr(inputs[0].inputs[0], 'columns_value', None) is not None:
            index_align_map_chunks = inputs[0].inputs
            if index_align_map_chunks[0].op.column_min_max is not None:
                # shuffle on index
                kw['columns_value'] = filter_index_value(index_align_map_chunks[0].columns_value,
                                                         index_align_map_chunks[0].op.column_min_max,
                                                         store_data=True)
                kw['dtypes'] = index_align_map_chunks[0].dtypes[kw['columns_value'].to_pandas()]
            else:
                # shuffle on columns
                all_dtypes = [c.op.column_shuffle_segments[index[1]] for c in index_align_map_chunks
                              if c.index[0] == index_align_map_chunks[0].index[0]]
                kw['dtypes'] = pd.concat(all_dtypes)
                kw['columns_value'] = parse_index(kw['dtypes'].index, store_data=True)
        if kw.get('dtype', None) and getattr(inputs[0].inputs[0], 'dtype', None) is not None:
            kw['dtype'] = inputs[0].inputs[0].dtype
        if kw.get('name', None) and getattr(inputs[0].inputs[0], 'name', None) is not None:
            kw['name'] = inputs[0].inputs[0].dtype
        return kw

    def _create_chunk(self, output_idx, index, **kw):
        if self.stage == OperandStage.map:
            kw = self._build_map_chunk_kw(**kw)
        else:
            kw = self._build_reduce_chunk_kw(index, **kw)
        return super()._create_chunk(output_idx, index, **kw)

    @classmethod
    def execute_map(cls, ctx, op):
        # TODO(QIN): add GPU support here
        df = ctx[op.inputs[0].key]

        filters = [[], []]

        chunk = op.outputs[0]
        if op.index_shuffle_size == -1:
            # no shuffle and no min-max filter on index
            filters[0].append(slice(None, None, None))
        elif op.index_shuffle_size is None:
            # no shuffle on index
            comp_op = operator.ge if op.index_min_close else operator.gt
            index_cond = comp_op(df.index, op.index_min)
            comp_op = operator.le if op.index_max_close else operator.lt
            index_cond = index_cond & comp_op(df.index, op.index_max)
            filters[0].append(index_cond)
        else:
            # shuffle on index
            shuffle_size = op.index_shuffle_size
            filters[0].extend(hash_index(df.index, shuffle_size))

        if op.object_type == ObjectType.series:
            if len(filters[0]) == 1:
                # no shuffle
                ctx[chunk.key] = df.loc[filters[0][0]]
            else:
                for index_idx, index_filter in enumerate(filters[0]):
                    group_key = str(index_idx)
                    ctx[(chunk.key, group_key)] = df.loc[index_filter]
            return

        if op.column_shuffle_size == -1:
            # no shuffle and no min-max filter on columns
            filters[1].append(slice(None, None, None))
        if op.column_shuffle_size is None:
            # no shuffle on columns
            comp_op = operator.ge if op.column_min_close else operator.gt
            columns_cond = comp_op(df.columns, op.column_min)
            comp_op = operator.le if op.column_max_close else operator.lt
            columns_cond = columns_cond & comp_op(df.columns, op.column_max)
            filters[1].append(columns_cond)
        else:
            # shuffle on columns
            shuffle_size = op.column_shuffle_size
            filters[1].extend(hash_index(df.columns, shuffle_size))

        if all(len(it) == 1 for it in filters):
            # no shuffle
            ctx[chunk.key] = df.loc[filters[0][0], filters[1][0]]
        elif len(filters[0]) == 1:
            # shuffle on columns
            for column_idx, column_filter in enumerate(filters[1]):
                group_key = ','.join([str(chunk.index[0]), str(column_idx)])
                ctx[(chunk.key, group_key)] = df.loc[filters[0][0], column_filter]
        elif len(filters[1]) == 1:
            # shuffle on index
            for index_idx, index_filter in enumerate(filters[0]):
                group_key = ','.join([str(index_idx), str(chunk.index[1])])
                ctx[(chunk.key, group_key)] = df.loc[index_filter, filters[1][0]]
        else:
            # full shuffle
            shuffle_index_size = op.index_shuffle_size
            shuffle_column_size = op.column_shuffle_size
            out_idxes = itertools.product(range(shuffle_index_size), range(shuffle_column_size))
            out_index_columns = itertools.product(*filters)
            for out_idx, out_index_column in zip(out_idxes, out_index_columns):
                index_filter, column_filter = out_index_column
                group_key = ','.join(str(i) for i in out_idx)
                ctx[(chunk.key, group_key)] = df.loc[index_filter, column_filter]

    @classmethod
    def execute_reduce(cls, ctx, op):
        chunk = op.outputs[0]
        input_keys, input_idxes = get_shuffle_input_keys_idxes(op.inputs[0])
        input_idx_to_df = {idx: ctx[inp_key, ','.join(str(ix) for ix in chunk.index)]
                           for inp_key, idx in zip(input_keys, input_idxes)}
        row_idxes = sorted({idx[0] for idx in input_idx_to_df})
        if op.object_type == ObjectType.dataframe:
            col_idxes = sorted({idx[1] for idx in input_idx_to_df})

        ress = []
        for row_idx in row_idxes:
            if op.object_type == ObjectType.dataframe:
                row_dfs = []
                for col_idx in col_idxes:
                    row_dfs.append(input_idx_to_df[row_idx, col_idx])
                row_df = pd.concat(row_dfs, axis=1)
            else:
                row_df = input_idx_to_df[(row_idx,)]

            ress.append(row_df)

        ctx[chunk.key] = pd.concat(ress, axis=0)

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls.execute_map(ctx, op)
        else:
            cls.execute_reduce(ctx, op)


class _AxisMinMaxSplitInfo(object):
    def __init__(self, left_split, left_increase, right_split, right_increase, dummy=False):
        self._left_split = left_split
        self._right_split = right_split
        self._dummy = dummy

        self._left_split_idx_to_origin_idx = \
            build_split_idx_to_origin_idx(self._left_split, left_increase)
        self._right_split_idx_to_origin_idx = \
            build_split_idx_to_origin_idx(self._right_split, right_increase)

    def isdummy(self):
        return self._dummy

    def get_origin_left_idx(self, idx):
        return self._left_split_idx_to_origin_idx[idx][0]

    def get_origin_left_split(self, idx):
        left_idx, left_inner_idx = \
            self._left_split_idx_to_origin_idx[idx]
        return self._left_split[left_idx][left_inner_idx]

    def get_origin_right_idx(self, idx):
        return self._right_split_idx_to_origin_idx[idx][0]

    def get_origin_right_split(self, idx):
        right_idx, right_inner_idx = \
            self._right_split_idx_to_origin_idx[idx]
        return self._right_split[right_idx][right_inner_idx]


class _MinMaxSplitInfo(object):
    def __init__(self, row_min_max_split_info=None, col_min_max_split_info=None):
        self.row_min_max_split_info = row_min_max_split_info
        self.col_min_max_split_info = col_min_max_split_info

    def all_axes_can_split(self):
        return self.row_min_max_split_info is not None and \
               self.col_min_max_split_info is not None

    def one_axis_can_split(self):
        return (self.row_min_max_split_info is None) ^ \
               (self.col_min_max_split_info is None)

    def no_axis_can_split(self):
        return self.row_min_max_split_info is None and \
               self.col_min_max_split_info is None

    def __getitem__(self, i):
        return [self.row_min_max_split_info, self.col_min_max_split_info][i]

    def __setitem__(self, axis, axis_min_max_split_info):
        assert axis in {0, 1}
        if axis == 0:
            self.row_min_max_split_info = axis_min_max_split_info
        else:
            self.col_min_max_split_info = axis_min_max_split_info

    def get_row_left_idx(self, out_idx):
        return self.row_min_max_split_info.get_origin_left_idx(out_idx)

    def get_row_left_split(self, out_idx):
        return self.row_min_max_split_info.get_origin_left_split(out_idx)

    def get_col_left_idx(self, out_idx):
        return self.col_min_max_split_info.get_origin_left_idx(out_idx)

    def get_col_left_split(self, out_idx):
        return self.col_min_max_split_info.get_origin_left_split(out_idx)

    def get_row_right_idx(self, out_idx):
        return self.row_min_max_split_info.get_origin_right_idx(out_idx)

    def get_row_right_split(self, out_idx):
        return self.row_min_max_split_info.get_origin_right_split(out_idx)

    def get_col_right_idx(self, out_idx):
        return self.col_min_max_split_info.get_origin_right_idx(out_idx)

    def get_col_right_split(self, out_idx):
        return self.col_min_max_split_info.get_origin_right_split(out_idx)

    def get_axis_idx(self, axis, left_or_right, out_idx):
        if axis == 0:
            if left_or_right == 0:
                return self.get_row_left_idx(out_idx)
            else:
                assert left_or_right == 1
                return self.get_row_right_idx(out_idx)
        else:
            assert axis == 1
            if left_or_right == 0:
                return self.get_col_left_idx(out_idx)
            else:
                assert left_or_right == 1
                return self.get_col_right_idx(out_idx)

    def get_axis_split(self, axis, left_or_right, out_idx):
        if axis == 0:
            if left_or_right == 0:
                return self.get_row_left_split(out_idx)
            else:
                assert left_or_right == 1
                return self.get_row_right_split(out_idx)
        else:
            assert axis == 1
            if left_or_right == 0:
                return self.get_col_left_split(out_idx)
            else:
                assert left_or_right == 1
                return self.get_col_right_split(out_idx)


def _get_chunk_index_min_max(index_chunks):
    chunk_index_min_max = []
    for chunk in index_chunks:
        min_val = chunk.min_val
        min_val_close = chunk.min_val_close
        max_val = chunk.max_val
        max_val_close = chunk.max_val_close
        if min_val is None or max_val is None:
            return
        chunk_index_min_max.append((min_val, min_val_close, max_val, max_val_close))
    return chunk_index_min_max


def _get_monotonic_chunk_index_min_max(index, index_chunks):
    chunk_index_min_max = _get_chunk_index_min_max(index_chunks)
    if index.is_monotonic_decreasing:
        return list(reversed(chunk_index_min_max)), False

    for j in range(len(chunk_index_min_max) - 1):
        # overlap only if the prev max is close and curr min is close
        # and they are identical
        prev_max, prev_max_close = chunk_index_min_max[j][2:]
        curr_min, curr_min_close = chunk_index_min_max[j + 1][:2]
        if prev_max_close and curr_min_close and prev_max == curr_min:
            return
    return chunk_index_min_max, True


def _need_align_map(input_chunk, index_min_max, column_min_max,
                    dummy_index_splits=False, dummy_column_splits=False):
    if not dummy_index_splits:
        assert not index_min_max[0] is None and not index_min_max[2] is None
    if isinstance(input_chunk, SERIES_CHUNK_TYPE):
        if input_chunk.index_value is None:
            return True
        if input_chunk.index_value.min_max != index_min_max:
            return True
    else:
        if not dummy_index_splits:
            if input_chunk.index_value is None or input_chunk.index_value.min_max != index_min_max:
                return True
        if not dummy_column_splits:
            if input_chunk.columns_value is None or input_chunk.columns_value.min_max != column_min_max:
                return True
    return False


def _is_index_identical(left, right):
    if len(left) != len(right):
        return False
    for left_item, right_item in zip(left, right):
        if left_item.key != right_item.key:
            return False
    return True


def _axis_need_shuffle(left_axis, right_axis, left_axis_chunks, right_axis_chunks):
    if _is_index_identical(left_axis_chunks, right_axis_chunks):
        return False
    if not left_axis.is_monotonic_increasing_or_decreasing and len(left_axis_chunks) > 1:
        return True
    if not right_axis.is_monotonic_increasing_or_decreasing and len(right_axis_chunks) > 1:
        return True
    return False


def _calc_axis_splits(left_axis, right_axis, left_axis_chunks, right_axis_chunks):
    if _axis_need_shuffle(left_axis, right_axis, left_axis_chunks, right_axis_chunks):
        # do shuffle
        out_chunk_size = max(len(left_axis_chunks), len(right_axis_chunks))
        return None, [np.nan for _ in range(out_chunk_size)]
    else:
        # no need to do shuffle on this axis
        if _is_index_identical(left_axis_chunks, right_axis_chunks):
            left_chunk_index_min_max = _get_chunk_index_min_max(left_axis_chunks)
            right_splits = left_splits = [[c] for c in left_chunk_index_min_max]
            right_increase = left_increase = None
        elif len(left_axis_chunks) == 1 and len(right_axis_chunks) == 1:
            left_splits = [_get_chunk_index_min_max(left_axis_chunks)]
            left_increase = left_axis_chunks[0].is_monotonic_decreasing
            right_splits = [_get_chunk_index_min_max(right_axis_chunks)]
            right_increase = right_axis_chunks[0].is_monotonic_decreasing
        else:
            left_chunk_index_min_max, left_increase = _get_monotonic_chunk_index_min_max(left_axis,
                                                                                         left_axis_chunks)
            right_chunk_index_min_max, right_increase = _get_monotonic_chunk_index_min_max(right_axis,
                                                                                           right_axis_chunks)
            left_splits, right_splits = split_monotonic_index_min_max(
                left_chunk_index_min_max, left_increase, right_chunk_index_min_max, right_increase)
        splits = _AxisMinMaxSplitInfo(left_splits, left_increase, right_splits, right_increase)
        nsplits = [np.nan for _ in itertools.chain(*left_splits)]
        return splits, nsplits


def _build_dummy_axis_split(chunk_shape):
    axis_index_min_max, axis_increase = [(i, True, i + 1, True) for i in range(chunk_shape)], True
    if len(axis_index_min_max) == 1:
        left_splits, right_splits = [axis_index_min_max], [axis_index_min_max]
    else:
        left_splits, right_splits = split_monotonic_index_min_max(
            axis_index_min_max, axis_increase, axis_index_min_max, axis_increase)
    return _AxisMinMaxSplitInfo(left_splits, axis_increase,
                                right_splits, axis_increase, dummy=True)


def _gen_series_chunks(splits, out_shape, left_or_right, series):
    out_chunks = []
    if splits[0] is not None:
        # need no shuffle
        for out_idx in range(out_shape[0]):
            idx = splits.get_axis_idx(0, left_or_right, out_idx)
            index_min_max = splits.get_axis_split(0, left_or_right, out_idx)
            chunk = series.cix[(idx,)]
            if _need_align_map(chunk, index_min_max, None):
                align_op = DataFrameIndexAlign(
                    stage=OperandStage.map, index_min_max=index_min_max, column_min_max=None,
                    dtype=chunk.dtype, sparse=series.issparse(), object_type=ObjectType.series)
                out_chunk = align_op.new_chunk([chunk], shape=(np.nan,), index=(out_idx,))
            else:
                out_chunk = chunk
            out_chunks.append(out_chunk)
    else:
        # gen map chunks
        map_chunks = []
        for chunk in series.chunks:
            map_op = DataFrameIndexAlign(
                stage=OperandStage.map, sparse=chunk.issparse(), index_shuffle_size=out_shape[0],
                object_type=ObjectType.series)
            map_chunks.append(map_op.new_chunk([chunk], shape=(np.nan,), index=chunk.index))

        proxy_chunk = DataFrameShuffleProxy(object_type=ObjectType.series).new_chunk(
            map_chunks, shape=())

        # gen reduce chunks
        for out_idx in range(out_shape[0]):
            reduce_op = DataFrameIndexAlign(stage=OperandStage.reduce, i=out_idx,
                                            sparse=proxy_chunk.issparse(), shuffle_key=str(out_idx),
                                            object_type=ObjectType.series)
            out_chunks.append(
                reduce_op.new_chunk([proxy_chunk], shape=(np.nan,), index=(out_idx,)))

    return out_chunks


def _gen_dataframe_chunks(splits, out_shape, left_or_right, df):
    out_chunks = []
    if splits.all_axes_can_split():
        # no shuffle for all axes
        kw = {
            'index_shuffle_size': -1 if splits[0].isdummy() else None,
            'column_shuffle_size': -1 if splits[1].isdummy() else None,
        }
        for out_idx in itertools.product(*(range(s) for s in out_shape)):
            row_idx = splits.get_axis_idx(0, left_or_right, out_idx[0])
            col_idx = splits.get_axis_idx(1, left_or_right, out_idx[1])
            index_min_max = splits.get_axis_split(0, left_or_right, out_idx[0])
            column_min_max = splits.get_axis_split(1, left_or_right, out_idx[1])
            chunk = df.cix[row_idx, col_idx]
            if _need_align_map(chunk, index_min_max, column_min_max,
                               splits[0].isdummy(), splits[1].isdummy()):
                if splits[1].isdummy():
                    dtypes = chunk.dtypes
                else:
                    dtypes = filter_dtypes(chunk.dtypes, column_min_max)
                chunk_kw = {
                    'index_value': chunk.index_value if splits[0].isdummy() else None,
                    'columns_value': chunk.columns_value if splits[1].isdummy() else None,
                    'dtypes': chunk.dtypes if splits[1].isdummy() else None
                }
                align_op = DataFrameIndexAlign(
                    stage=OperandStage.map, index_min_max=index_min_max,
                    column_min_max=column_min_max, dtypes=dtypes, sparse=chunk.issparse(),
                    object_type=ObjectType.dataframe, **kw)
                out_chunk = align_op.new_chunk([chunk], shape=(np.nan, np.nan), index=out_idx, **chunk_kw)
            else:
                out_chunk = chunk
            out_chunks.append(out_chunk)
    elif splits.one_axis_can_split():
        # one axis needs shuffle
        shuffle_axis = 0 if splits[0] is None else 1
        align_axis = 1 - shuffle_axis

        for align_axis_idx in range(out_shape[align_axis]):
            if align_axis == 0:
                kw = {
                    'index_min_max': splits.get_axis_split(align_axis, left_or_right, align_axis_idx),
                    'index_shuffle_size': -1 if splits[0].isdummy() else None,
                    'column_shuffle_size': out_shape[shuffle_axis],
                }
                input_idx = splits.get_axis_idx(align_axis, left_or_right, align_axis_idx)
            else:
                kw = {
                    'column_min_max': splits.get_axis_split(align_axis, left_or_right, align_axis_idx),
                    'index_shuffle_size': out_shape[shuffle_axis],
                    'column_shuffle_size': -1 if splits[1].isdummy() else None,
                }
                input_idx = splits.get_axis_idx(align_axis, left_or_right, align_axis_idx)
            input_chunks = [c for c in df.chunks if c.index[align_axis] == input_idx]
            map_chunks = []
            for j, input_chunk in enumerate(input_chunks):
                chunk_kw = dict()
                if align_axis == 0:
                    chunk_kw['index_value'] = input_chunk.index_value if splits[0].isdummy() else None
                else:
                    chunk_kw['columns_value'] = input_chunk.columns_value if splits[1].isdummy() else None
                map_op = DataFrameIndexAlign(stage=OperandStage.map, sparse=input_chunk.issparse(),
                                             object_type=ObjectType.dataframe, **kw)
                idx = [None, None]
                idx[align_axis] = align_axis_idx
                idx[shuffle_axis] = j
                map_chunks.append(map_op.new_chunk([input_chunk], shape=(np.nan, np.nan), index=tuple(idx), **chunk_kw))
            proxy_chunk = DataFrameShuffleProxy(
                sparse=df.issparse(), object_type=ObjectType.dataframe).new_chunk(map_chunks, shape=())
            for j in range(out_shape[shuffle_axis]):
                chunk_kw = dict()
                if align_axis == 0:
                    chunk_kw['index_value'] = proxy_chunk.inputs[0].inputs[0].index_value \
                        if splits[0].isdummy() else None
                else:
                    chunk_kw['columns_value'] = proxy_chunk.inputs[0].inputs[0].columns_value \
                        if splits[1].isdummy() else None
                reduce_idx = (align_axis_idx, j) if align_axis == 0 else (j, align_axis_idx)
                reduce_op = DataFrameIndexAlign(stage=OperandStage.reduce, i=j, sparse=proxy_chunk.issparse(),
                                                shuffle_key=','.join(str(idx) for idx in reduce_idx),
                                                object_type=ObjectType.dataframe)
                out_chunks.append(
                    reduce_op.new_chunk([proxy_chunk], shape=(np.nan, np.nan), index=reduce_idx, **chunk_kw))
        out_chunks.sort(key=lambda c: c.index)
    else:
        # all axes need shuffle
        assert splits.no_axis_can_split()

        # gen map chunks
        map_chunks = []
        for chunk in df.chunks:
            map_op = DataFrameIndexAlign(
                stage=OperandStage.map, sparse=chunk.issparse(), index_shuffle_size=out_shape[0],
                column_shuffle_size=out_shape[1], object_type=ObjectType.dataframe)
            map_chunks.append(map_op.new_chunk([chunk], shape=(np.nan, np.nan), index=chunk.index))

        proxy_chunk = DataFrameShuffleProxy(object_type=ObjectType.dataframe).new_chunk(
            map_chunks, shape=())

        # gen reduce chunks
        for out_idx in itertools.product(*(range(s) for s in out_shape)):
            reduce_op = DataFrameIndexAlign(stage=OperandStage.reduce, i=out_idx,
                                            sparse=proxy_chunk.issparse(),
                                            shuffle_key=','.join(str(idx) for idx in out_idx),
                                            object_type=ObjectType.dataframe)
            out_chunks.append(
                reduce_op.new_chunk([proxy_chunk], shape=(np.nan, np.nan), index=out_idx))

    return out_chunks


def align_dataframe_dataframe(left, right):
    left_index_chunks = [c.index_value for c in left.cix[:, 0]]
    left_columns_chunks = [c.columns_value for c in left.cix[0, :]]
    right_index_chunks = [c.index_value for c in right.cix[:, 0]]
    right_columns_chunks = [c.columns_value for c in right.cix[0, :]]

    index_splits, index_nsplits = _calc_axis_splits(left.index_value, right.index_value,
                                                    left_index_chunks, right_index_chunks)
    if _is_index_identical(left_index_chunks, right_index_chunks):
        index_nsplits = left.nsplits[0]

    columns_splits, columns_nsplits = _calc_axis_splits(left.columns_value, right.columns_value,
                                                        left_columns_chunks, right_columns_chunks)
    if _is_index_identical(left_columns_chunks, right_columns_chunks):
        columns_nsplits = left.nsplits[1]

    nsplits = [index_nsplits, columns_nsplits]
    out_chunk_shape = tuple(len(ns) for ns in nsplits)
    splits = _MinMaxSplitInfo(index_splits, columns_splits)

    left_chunks = _gen_dataframe_chunks(splits, out_chunk_shape, 0, left)
    right_chunks = _gen_dataframe_chunks(splits, out_chunk_shape, 1, right)

    return nsplits, out_chunk_shape, left_chunks, right_chunks


def align_dataframe_series(left, right, axis='columns'):
    if axis == 'columns' or axis == 1:
        left_columns_chunks = [c.columns_value for c in left.cix[0, :]]
        right_index_chunks = [c.index_value for c in right.chunks]
        index_splits, index_nsplits = _calc_axis_splits(left.columns_value, right.index_value,
                                                        left_columns_chunks, right_index_chunks)
        if _is_index_identical(left_columns_chunks, right_index_chunks):
            index_nsplits = left.nsplits[1]
        dummy_splits, dummy_nsplits = _build_dummy_axis_split(left.chunk_shape[0]), left.nsplits[0]
        nsplits = [dummy_nsplits, index_nsplits]
        out_chunk_shape = tuple(len(ns) for ns in nsplits)
        left_chunks = _gen_dataframe_chunks(_MinMaxSplitInfo(dummy_splits, index_splits), out_chunk_shape, 0, left)
        right_chunks = _gen_series_chunks(_MinMaxSplitInfo(index_splits, None), (out_chunk_shape[1],), 1, right)
    else:
        assert axis == 'index' or axis == 0
        left_index_chunks = [c.index_value for c in left.cix[:, 0]]
        right_index_chunks = [c.index_value for c in right.chunks]
        index_splits, index_nsplits = _calc_axis_splits(left.index_value, right.index_value,
                                                        left_index_chunks, right_index_chunks)
        if _is_index_identical(left_index_chunks, right_index_chunks):
            index_nsplits = left.nsplits[0]
        dummy_splits, dummy_nsplits = _build_dummy_axis_split(left.chunk_shape[1]), left.nsplits[1]
        nsplits = [index_nsplits, dummy_nsplits]
        out_chunk_shape = tuple(len(ns) for ns in nsplits)
        left_chunks = _gen_dataframe_chunks(_MinMaxSplitInfo(index_splits, dummy_splits), out_chunk_shape, 0, left)
        right_chunks = _gen_series_chunks(_MinMaxSplitInfo(index_splits, None), (out_chunk_shape[0],), 1, right)

    return nsplits, out_chunk_shape, left_chunks, right_chunks


def align_series_series(left, right):
    left_index_chunks = [c.index_value for c in left.chunks]
    right_index_chunks = [c.index_value for c in right.chunks]

    index_splits, index_nsplits = _calc_axis_splits(left.index_value, right.index_value,
                                                    left_index_chunks, right_index_chunks)
    if _is_index_identical(left_index_chunks, right_index_chunks):
        index_nsplits = left.nsplits[0]
    nsplits = [index_nsplits]
    out_chunk_shape = (len(index_nsplits),)
    splits = _MinMaxSplitInfo(index_splits, None)

    left_chunks = _gen_series_chunks(splits, out_chunk_shape, 0, left)
    right_chunks = _gen_series_chunks(splits, out_chunk_shape, 1, right)

    return nsplits, out_chunk_shape, left_chunks, right_chunks
