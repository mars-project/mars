# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import numpy as np

from ....operands import Operand, ShuffleReduce
from .... import opcodes as OperandDef
from ....serialize import ValueType, AnyField, BoolField, Int32Field, KeyField, ListField
from ....utils import classproperty, tokenize
from ...core import DATAFRAME_TYPE
from ...utils import hash_index
from ..core import DataFrameOperandMixin, DataFrameShuffleProxy
from ..utils import parse_index, split_monotonic_index_min_max, \
    build_split_idx_to_origin_idx, filter_index_value
from .utils import infer_dtypes, infer_index_value, filter_dtypes


class DataFrameIndexAlignMap(Operand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_INDEX_ALIGN_MAP

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
    _column_shuffle_segments = ListField('column_shuffle_segments', ValueType.list)

    def __init__(self, index_min_max=None, index_shuffle_size=None, column_min_max=None,
                 column_shuffle_size=None, column_shuffle_segments=None,
                 sparse=None, dtypes=None, gpu=None, **kw):
        if index_min_max is not None:
            kw.update(dict(_index_min=index_min_max[0], _index_min_close=index_min_max[1],
                           _index_max=index_min_max[2], _index_max_close=index_min_max[3]))
        if column_min_max is not None:
            kw.update(dict(_column_min=column_min_max[0], _column_min_close=column_min_max[1],
                           _column_max=column_min_max[2], _column_max_close=column_min_max[3]))
        super(DataFrameIndexAlignMap, self).__init__(
            _index_shuffle_size=index_shuffle_size, _column_shuffle_size=column_shuffle_size,
            _column_shuffle_segments=column_shuffle_segments, _sparse=sparse,
            _dtypes=dtypes, _gpu=gpu, **kw)

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

    def _new_chunks(self, inputs, shape, index=None, output_limit=None, kws=None, **kw):
        if kw.get('index_value', None) is None and inputs[0].index_value is not None:
            input_index_value = inputs[0].index_value
            index_min_max = self.index_min_max
            if index_min_max is not None:
                kw['index_value'] = filter_index_value(input_index_value, index_min_max)
            else:
                kw['index_value'] = parse_index(inputs[0].index_value.to_pandas(),
                                                key=tokenize(input_index_value.key,
                                                             type(self).__name__))
        if kw.get('columns_value', None) is None and inputs[0].columns is not None:
            input_columns_value = inputs[0].columns
            column_min_max = self.column_min_max
            if column_min_max is not None:
                kw['columns_value'] = filter_index_value(input_columns_value, column_min_max,
                                                         store_data=True)
            else:
                kw['column_value'] = parse_index(inputs[0].columns.to_pandas(),
                                                 key=tokenize(input_columns_value.key,
                                                              type(self).__name__))
            column_shuffle_size = self.column_shuffle_size
            if column_shuffle_size is not None:
                pd_columns = input_columns_value.to_pandas()
                self._column_shuffle_segments = hash_index(pd_columns, column_shuffle_size)

        return super(DataFrameIndexAlignMap, self)._new_chunks(inputs, shape, index=index,
                                                               output_limit=output_limit, kws=kws,
                                                               **kw)


class DataFrameIndexAlignReduce(ShuffleReduce, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_INDEX_ALIGN_REDUCE

    _input = KeyField('input')

    def __init__(self, shuffle_key=None, sparse=None, **kw):
        super(DataFrameIndexAlignReduce, self).__init__(
            _shuffle_key=shuffle_key, _sparse=sparse, **kw)

    def _set_inputs(self, inputs):
        super(DataFrameIndexAlignReduce, self)._set_inputs(inputs)
        self._input = self._inputs[0]

    def calc_shape(self, *inputs_shape):
        return self.outputs[0].shape


class _AxisMinMaxSplitInfo(object):
    def __init__(self, left_split, left_increase, right_split, right_increase):
        self._left_split = left_split
        self._right_split =right_split

        self._left_split_idx_to_origin_idx = \
            build_split_idx_to_origin_idx(self._left_split, left_increase)
        self._right_split_idx_to_origin_idx = \
            build_split_idx_to_origin_idx(self._right_split, right_increase)

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
    def __init__(self):
        self.row_min_max_split_info = None
        self.col_min_max_split_info = None

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
        if axis == 0 and left_or_right == 0:
            return self.get_row_left_idx(out_idx)
        elif axis == 0 and left_or_right == 1:
            return self.get_row_right_idx(out_idx)
        elif axis == 1 and left_or_right == 0:
            return self.get_col_left_idx(out_idx)
        else:
            assert axis == 1 and left_or_right == 1
            return self.get_col_right_idx(out_idx)

    def get_axis_split(self, axis, left_or_right, out_idx):
        if axis == 0 and left_or_right == 0:
            return self.get_row_left_split(out_idx)
        elif axis == 0 and left_or_right == 1:
            return self.get_row_right_split(out_idx)
        elif axis == 1 and left_or_right == 0:
            return self.get_col_left_split(out_idx)
        else:
            assert axis == 1 and left_or_right == 1
            return self.get_col_right_split(out_idx)


class DataFrameBinOpMixin(DataFrameOperandMixin):
    __slots__ = ()

    @classmethod
    def _check_overlap(cls, chunk_index_min_max):
        for j in range(len(chunk_index_min_max) - 1):
            # overlap only if the prev max is close and curr min is close
            # and they are identical
            prev_max, prev_max_close = chunk_index_min_max[j][2:]
            curr_min, curr_min_close = chunk_index_min_max[j + 1][:2]
            if prev_max_close and curr_min_close and prev_max == curr_min:
                return True
        return False

    @classmethod
    def _get_chunk_index_min_max(cls, df, index_type, axis):
        index = getattr(df, index_type)
        if not index.is_monotonic_increasing_or_decreasing:
            return

        chunk_index_min_max = []
        for i in range(df.chunk_shape[axis]):
            chunk_idx = [0, 0]
            chunk_idx[axis] = i
            chunk = df.cix[tuple(chunk_idx)]
            chunk_index = getattr(chunk, index_type)
            min_val = chunk_index.min_val
            min_val_close = chunk_index.min_val_close
            max_val = chunk_index.max_val
            max_val_close = chunk_index.max_val_close
            if min_val is None or max_val is None:
                return
            chunk_index_min_max.append((min_val, min_val_close, max_val, max_val_close))

        if index.is_monotonic_decreasing:
            return list(reversed(chunk_index_min_max)), False

        if cls._check_overlap(chunk_index_min_max):
            return
        return chunk_index_min_max, True

    @classmethod
    def _need_align_map(cls, input_chunk, index_min_max, column_min_max):
        if input_chunk.index_value is None or input_chunk.columns is None:
            return True
        if input_chunk.index_value.min_max != index_min_max:
            return True
        if input_chunk.columns.min_max != column_min_max:
            return True
        return False

    @classmethod
    def _gen_out_chunks_without_shuffle(cls, op, splits, out_shape, left, right):
        out_chunks = []
        for out_idx in itertools.product(*(range(s) for s in out_shape)):
            # does not need shuffle
            left_row_idx = splits.get_row_left_idx(out_idx[0])
            left_col_idx = splits.get_col_left_idx(out_idx[1])
            left_index_min_max = splits.get_row_left_split(out_idx[0])
            left_column_min_max = splits.get_col_left_split(out_idx[1])
            left_chunk = left.cix[left_row_idx, left_col_idx]
            if cls._need_align_map(left_chunk, left_index_min_max, left_column_min_max):
                left_align_op = DataFrameIndexAlignMap(
                    index_min_max=left_index_min_max, column_min_max=left_column_min_max,
                    dtypes=filter_dtypes(left_chunk.dtypes, left_column_min_max),
                    sparse=left_chunk.issparse())
                left_out_chunk = left_align_op.new_chunk([left_chunk], (np.nan, np.nan),
                                                         index=out_idx)
            else:
                left_out_chunk = left_chunk

            right_row_idx = splits.get_row_right_idx(out_idx[0])
            right_col_idx = splits.get_col_right_idx(out_idx[1])
            right_index_min_max = splits.get_row_right_split(out_idx[0])
            right_column_min_max = splits.get_col_right_split(out_idx[1])
            right_chunk = right.cix[right_row_idx, right_col_idx]
            if cls._need_align_map(right_chunk, right_index_min_max, right_column_min_max):
                right_align_op = DataFrameIndexAlignMap(
                    index_min_max=right_index_min_max, column_min_max=right_column_min_max,
                    dtypes=filter_dtypes(right.dtypes, right_column_min_max),
                    sparse=right_chunk.issparse())
                right_out_chunk = right_align_op.new_chunk([right_chunk], (np.nan, np.nan),
                                                           index=out_idx)
            else:
                right_out_chunk = right_chunk

            out_op = op.copy().reset_key()
            out_chunks.append(
                out_op.new_chunk([left_out_chunk, right_out_chunk], (np.nan, np.nan),
                                 index=out_idx))

        return out_chunks

    @classmethod
    def _gen_out_chunks_with_one_shuffle(cls, op, splits, out_shape, left, right):
        shuffle_axis = 0 if splits[0] is None else 1
        shuffle_size = out_shape[shuffle_axis]
        align_axis = 1 - shuffle_axis

        out_chunks = []
        for align_axis_idx in range(out_shape[align_axis]):
            reduce_chunks = [[], []]
            for left_or_right in range(2):  # left and right
                if align_axis == 0:
                    kw = {
                        'index_min_max': splits.get_axis_split(align_axis, left_or_right, align_axis_idx),
                        'column_shuffle_size': shuffle_size,
                    }
                    input_idx = splits.get_axis_idx(align_axis, left_or_right, align_axis_idx)
                else:
                    kw = {
                        'index_shuffle_size': shuffle_size,
                        'column_min_max': splits.get_axis_split(align_axis, left_or_right, align_axis_idx)
                    }
                    input_idx = splits.get_axis_idx(align_axis, left_or_right, align_axis_idx)
                inp = left if left_or_right == 0 else right
                input_chunks = [c for c in inp.chunks if c.index[align_axis] == input_idx]
                map_chunks = []
                for j, input_chunk in enumerate(input_chunks):
                    map_op = DataFrameIndexAlignMap(sparse=input_chunks[0].issparse(), **kw)
                    idx = [None, None]
                    idx[align_axis] = align_axis_idx
                    idx[shuffle_axis] = j
                    map_chunks.append(map_op.new_chunk([input_chunk], (np.nan, np.nan), index=tuple(idx)))
                proxy_chunk = DataFrameShuffleProxy(sparse=inp.issparse()).new_chunk(map_chunks, ())
                for j in range(shuffle_size):
                    reduce_idx = (align_axis_idx, j) if align_axis == 0 else (j, align_axis_idx)
                    reduce_op = DataFrameIndexAlignReduce(i=j, sparse=proxy_chunk.issparse(),
                                                          shuffle_key=','.join(str(idx) for idx in reduce_idx))
                    reduce_chunks[left_or_right].append(
                        reduce_op.new_chunk([proxy_chunk], (np.nan, np.nan), index=reduce_idx))

            assert len(reduce_chunks[0]) == len(reduce_chunks[1])
            for left_chunk, right_chunk in zip(*reduce_chunks):
                bin_op = op.copy().reset_key()
                out_chunk = bin_op.new_chunk([left_chunk, right_chunk], (np.nan, np.nan),
                                             index=left_chunk.index)
                out_chunks.append(out_chunk)

        return out_chunks

    @classmethod
    def _gen_out_chunks_with_all_shuffle(cls, op, out_shape, left, right):
        out_chunks = []

        # gen map chunks
        reduce_chunks = [[], []]
        for i in range(2):  # left, right
            inp = left if i == 0 else right
            map_chunks = []
            for chunk in inp.chunks:
                map_op = DataFrameIndexAlignMap(
                    sparse=chunk.issparse(), index_shuffle_size=out_shape[0],
                    column_shuffle_size=out_shape[1])
                map_chunks.append(map_op.new_chunk([chunk], (np.nan, np.nan), index=chunk.index))

            proxy_chunk = DataFrameShuffleProxy().new_chunk(map_chunks, ())
            for out_idx in itertools.product(*(range(s) for s in out_shape)):
                reduce_op = DataFrameIndexAlignReduce(i=out_idx,
                                                      sparse=proxy_chunk.issparse(),
                                                      shuffle_key=','.join(str(idx) for idx in out_idx))
                reduce_chunks[i].append(
                    reduce_op.new_chunk([proxy_chunk], (np.nan, np.nan), index=out_idx))

        for left_chunk, right_chunk in zip(*reduce_chunks):
            bin_op = op.copy().reset_key()
            out_chunk = bin_op.new_chunk([left_chunk, right_chunk], (np.nan, np.nan),
                                         index=left_chunk.index)
            out_chunks.append(out_chunk)

        return out_chunks

    @classmethod
    def _tile_both_dataframes(cls, op):
        # if both of the inputs are DataFrames, axis is just ignored
        left, right = op.inputs
        nsplits = [[], []]
        splits = _MinMaxSplitInfo()

        # first, we decide the chunk size on each axis
        # we perform the same logic for both index and columns
        for axis, index_type in enumerate(['index_value', 'columns']):
            # if both of the indexes are monotonic increasing or decreasing
            left_chunk_index_min_max = cls._get_chunk_index_min_max(left, index_type, axis)
            right_chunk_index_min_max = cls._get_chunk_index_min_max(right, index_type, axis)
            if left_chunk_index_min_max is not None and right_chunk_index_min_max is not None:
                # no need to do shuffle on this axis
                left_splits, right_splits = split_monotonic_index_min_max(
                    *(left_chunk_index_min_max + right_chunk_index_min_max))
                left_increase = left_chunk_index_min_max[1]
                right_increase = right_chunk_index_min_max[1]
                splits[axis] = _AxisMinMaxSplitInfo(left_splits, left_increase,
                                                    right_splits, right_increase)
                nsplits[axis].extend(np.nan for _ in itertools.chain(*left_splits))
            else:
                # do shuffle
                left_chunk_size = left.chunk_shape[axis]
                right_chunk_size = right.chunk_shape[axis]
                out_chunk_size = max(left_chunk_size, right_chunk_size)
                nsplits[axis].extend(np.nan for _ in range(out_chunk_size))

        out_shape = tuple(len(ns) for ns in nsplits)
        if splits.all_axes_can_split():
            # no shuffle for all axes
            out_chunks = cls._gen_out_chunks_without_shuffle(op, splits, out_shape, left, right)
        elif splits.one_axis_can_split():
            # one axis needs shuffle
            out_chunks = cls._gen_out_chunks_with_one_shuffle(op, splits, out_shape, left, right)
        else:
            # all axes need shuffle
            assert splits.no_axis_can_split()
            out_chunks = cls._gen_out_chunks_with_all_shuffle(op, out_shape, left, right)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, op.outputs[0].shape,
                                     nsplits=tuple(tuple(ns) for ns in nsplits),
                                     chunks=out_chunks)

    @classmethod
    def tile(cls, op):
        if all(isinstance(inp, DATAFRAME_TYPE) for inp in op.inputs):
            return cls._tile_both_dataframes(op)

        raise NotImplementedError

    @classproperty
    def _operator(self):
        raise NotImplementedError

    @classmethod
    def _calc_properties(cls, x1, x2):
        dtypes = columns = index = None
        index_shape = column_shape = np.nan
        if x1.dtypes is not None and x2.dtypes is not None:
            dtypes = infer_dtypes(x1.dtypes, x2.dtypes, cls._operator)
            column_shape = len(dtypes)
            columns = parse_index(dtypes.index, store_data=True)
            columns.value.should_be_monotonic = True
        if x1.index_value is not None and x2.index_value is not None:
            index = infer_index_value(x1.index_value, x2.index_value, cls._operator)
            index.value.should_be_monotonic = True
            if index.key == x1.index_value.key == x2.index_value.key and \
                    (not np.isnan(x1.shape[0]) or not np.isnan(x2.shape[0])):
                index_shape = x1.shape[0] if not np.isnan(x1.shape[0]) else x2.shape[0]

        return {'shape': (index_shape, column_shape), 'dtypes': dtypes,
                'columns_value': columns, 'index_value': index}

    @staticmethod
    def _merge_shape(*shapes):
        ret = [np.nan, np.nan]
        for shape in shapes:
            for i, s in enumerate(shape):
                if np.isnan(ret[i]) and not np.isnan(s):
                    ret[i] = s
        return tuple(ret)

    def _new_chunks(self, inputs, shape, index=None, output_limit=None, kws=None, **kw):
        properties = self._calc_properties(*inputs)
        s = properties.pop('shape')
        shape = self._merge_shape(shape, s)
        for prop, value in properties.items():
            if kw.get(prop, None) is None:
                kw[prop] = value

        return super(DataFrameBinOpMixin, self)._new_chunks(
            inputs, shape, index=index, output_limit=output_limit,
            kws=kws, **kw)

    def _call(self, x1, x2):
        kw = self._calc_properties(x1, x2)
        shape = kw.pop('shape', None)
        return self.new_dataframe([x1, x2], shape, **kw)

    def __call__(self, x1, x2):
        return self._call(x1, x2)

    def rcall(self, x1, x2):
        return self._call(x1, x2)
