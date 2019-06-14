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
import copy

import numpy as np
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pass

from .... import opcodes as OperandDef
from ....serialize import ValueType, AnyField, BoolField, Int32Field, KeyField, ListField
from ....utils import classproperty, tokenize
from ...core import DATAFRAME_TYPE
from ...utils import hash_dtypes
from ..core import DataFrameOperand, DataFrameOperandMixin, DataFrameShuffleProxy, DataFrameShuffleReduce
from ..utils import parse_index, split_monotonic_index_min_max, \
    build_split_idx_to_origin_idx, filter_index_value
from .utils import infer_dtypes, infer_index_value, filter_dtypes


class DataFrameIndexAlignMap(DataFrameOperand, DataFrameOperandMixin):
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
    _column_shuffle_segments = ListField('column_shuffle_segments', ValueType.series)

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

    def _create_chunk(self, output_idx, index, **kw):
        inputs = self.inputs
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
            input_dtypes = inputs[0].dtypes
            column_min_max = self.column_min_max
            if column_min_max is not None:
                kw['columns_value'] = filter_index_value(input_columns_value, column_min_max,
                                                         store_data=True)
            else:
                kw['columns_value'] = parse_index(inputs[0].columns.to_pandas(),
                                                  key=tokenize(input_columns_value.key,
                                                               type(self).__name__))
            kw['dtypes'] = input_dtypes[kw['columns_value'].to_pandas()]
            column_shuffle_size = self.column_shuffle_size
            if column_shuffle_size is not None:
                self._column_shuffle_segments = hash_dtypes(input_dtypes, column_shuffle_size)
        return super(DataFrameIndexAlignMap, self)._create_chunk(output_idx, index, **kw)


class DataFrameIndexAlignReduce(DataFrameShuffleReduce, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_INDEX_ALIGN_REDUCE

    _input = KeyField('input')

    def __init__(self, shuffle_key=None, sparse=None, **kw):
        super(DataFrameIndexAlignReduce, self).__init__(_shuffle_key=shuffle_key, _sparse=sparse, **kw)

    def _set_inputs(self, inputs):
        super(DataFrameIndexAlignReduce, self)._set_inputs(inputs)
        self._input = self._inputs[0]

    def _create_chunk(self, output_idx, index, **kw):
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
                                                key=tokenize([c.key for c in index_align_map_chunks],
                                                             type(self).__name__))
        if kw.get('columns_value', None) is None and inputs[0].inputs[0].columns is not None:
            index_align_map_chunks = inputs[0].inputs
            if index_align_map_chunks[0].op.column_min_max is not None:
                # shuffle on index
                kw['columns_value'] = filter_index_value(index_align_map_chunks[0].columns,
                                                         index_align_map_chunks[0].op.column_min_max,
                                                         store_data=True)
                kw['dtypes'] = index_align_map_chunks[0].dtypes[kw['columns_value'].to_pandas()]
            else:
                # shuffle on columns
                all_dtypes = [c.op.column_shuffle_segments[index[1]] for c in index_align_map_chunks
                              if c.index[0] == index_align_map_chunks[0].index[0]]
                kw['dtypes'] = pd.concat(all_dtypes)
                kw['columns_value'] = parse_index(kw['dtypes'].index, store_data=True)
        return super(DataFrameIndexAlignReduce, self)._create_chunk(output_idx, index, **kw)


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
        assert not np.isnan(index_min_max[0]) and not np.isnan(index_min_max[2])
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
                left_out_chunk = left_align_op.new_chunk([left_chunk], shape=(np.nan, np.nan),
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
                right_out_chunk = right_align_op.new_chunk([right_chunk], shape=(np.nan, np.nan),
                                                           index=out_idx)
            else:
                right_out_chunk = right_chunk

            out_op = op.copy().reset_key()
            out_chunks.append(
                out_op.new_chunk([left_out_chunk, right_out_chunk], shape=(np.nan, np.nan),
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
                    map_chunks.append(map_op.new_chunk([input_chunk], shape=(np.nan, np.nan), index=tuple(idx)))
                proxy_chunk = DataFrameShuffleProxy(sparse=inp.issparse()).new_chunk(map_chunks, shape=())
                for j in range(shuffle_size):
                    reduce_idx = (align_axis_idx, j) if align_axis == 0 else (j, align_axis_idx)
                    reduce_op = DataFrameIndexAlignReduce(i=j, sparse=proxy_chunk.issparse(),
                                                          shuffle_key=','.join(str(idx) for idx in reduce_idx))
                    reduce_chunks[left_or_right].append(
                        reduce_op.new_chunk([proxy_chunk], shape=(np.nan, np.nan), index=reduce_idx))

            assert len(reduce_chunks[0]) == len(reduce_chunks[1])
            for left_chunk, right_chunk in zip(*reduce_chunks):
                bin_op = op.copy().reset_key()
                out_chunk = bin_op.new_chunk([left_chunk, right_chunk], shape=(np.nan, np.nan),
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
                map_chunks.append(map_op.new_chunk([chunk], shape=(np.nan, np.nan), index=chunk.index))

            proxy_chunk = DataFrameShuffleProxy().new_chunk(map_chunks, shape=())
            for out_idx in itertools.product(*(range(s) for s in out_shape)):
                reduce_op = DataFrameIndexAlignReduce(i=out_idx,
                                                      sparse=proxy_chunk.issparse(),
                                                      shuffle_key=','.join(str(idx) for idx in out_idx)
                                                      )
                reduce_chunks[i].append(
                    reduce_op.new_chunk([proxy_chunk], shape=(np.nan, np.nan), index=out_idx))

        for left_chunk, right_chunk in zip(*reduce_chunks):
            bin_op = op.copy().reset_key()
            out_chunk = bin_op.new_chunk([left_chunk, right_chunk], shape=(np.nan, np.nan),
                                         index=left_chunk.index)
            out_chunks.append(out_chunk)

        return out_chunks

    @classmethod
    def _is_index_identical(cls, left, right, index_type, axis):
        if left.chunk_shape[axis] != right.chunk_shape[axis]:
            return False

        for i in range(left.chunk_shape[axis]):
            idx = [0, 0]
            idx[axis] = i
            if getattr(left.cix[tuple(idx)], index_type).key != \
                    getattr(right.cix[tuple(idx)], index_type).key:
                return False

        return True

    @classmethod
    def _need_shuffle_on_axis(cls, left, right, index_type, axis):
        if cls._is_index_identical(left, right, index_type, axis):
            return False

        for df in (left, right):
            index = getattr(df, index_type)
            if not index.is_monotonic_increasing_or_decreasing and \
                    df.chunk_shape[axis] > 1:
                return True

        return False

    @classmethod
    def _tile_both_dataframes(cls, op):
        # if both of the inputs are DataFrames, axis is just ignored
        left, right = op.inputs
        df = op.outputs[0]
        nsplits = [[], []]
        splits = _MinMaxSplitInfo()

        # first, we decide the chunk size on each axis
        # we perform the same logic for both index and columns
        for axis, index_type in enumerate(['index_value', 'columns']):
            if not cls._need_shuffle_on_axis(left, right, index_type, axis):
                left_chunk_index_min_max = cls._get_chunk_index_min_max(left, index_type, axis)
                right_chunk_index_min_max = cls._get_chunk_index_min_max(right, index_type, axis)
                # no need to do shuffle on this axis
                if len(left_chunk_index_min_max[0]) == 1 and len(right_chunk_index_min_max[0]) == 1:
                    # both left and right has only 1 chunk
                    left_splits, right_splits = \
                        [left_chunk_index_min_max[0]],  [right_chunk_index_min_max[0]]
                else:
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
        return new_op.new_dataframes(op.inputs, df.shape,
                                     nsplits=tuple(tuple(ns) for ns in nsplits),
                                     chunks=out_chunks, dtypes=df.dtypes,
                                     index_value=df.index_value, columns_value=df.columns)

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

        if x1.columns.key == x2.columns.key:
            dtypes = x1.dtypes
            column_shape = len(dtypes)
            columns = copy.copy(x1.columns)
            columns.value.should_be_monotonic = True
        elif x1.dtypes is not None and x2.dtypes is not None:
            dtypes = infer_dtypes(x1.dtypes, x2.dtypes, cls._operator)
            column_shape = len(dtypes)
            columns = parse_index(dtypes.index, store_data=True)
            columns.value.should_be_monotonic = True

        if x1.index_value.key == x2.index_value.key:
            index = copy.copy(x1.index_value)
            index.value.should_be_monotonic = True
            index_shape = x1.shape[0]
        elif x1.index_value is not None and x2.index_value is not None:
            index = infer_index_value(x1.index_value, x2.index_value, cls._operator)
            index.value.should_be_monotonic = True
            if index.key == x1.index_value.key == x2.index_value.key and \
                    (not np.isnan(x1.shape[0]) or not np.isnan(x2.shape[0])):
                index_shape = x1.shape[0] if not np.isnan(x1.shape[0]) else x2.shape[0]

        return {'shape': (index_shape, column_shape), 'dtypes': dtypes,
                'columns_value': columns, 'index_value': index}

    def _new_chunks(self, inputs, kws=None, **kw):
        properties = self._calc_properties(*inputs)
        shapes = [properties.pop('shape')]
        shapes.extend(kw_item.pop('shape') for kw_item in kws or ())
        if 'shape' in kw:
            shapes.append(kw.pop('shape'))
        shape = self._merge_shape(*shapes)

        for prop, value in properties.items():
            if kw.get(prop, None) is None:
                kw[prop] = value

        return super(DataFrameBinOpMixin, self)._new_chunks(
            inputs, shape=shape, kws=kws, **kw)

    def _call(self, x1, x2):
        kw = self._calc_properties(x1, x2)
        shape = kw.pop('shape', None)
        return self.new_dataframe([x1, x2], shape, **kw)

    def __call__(self, x1, x2):
        return self._call(x1, x2)

    def rcall(self, x1, x2):
        return self._call(x1, x2)


class DataFrameUnaryOpMixin(DataFrameOperandMixin):
    __slots__ = ()

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        out_chunks = []
        for in_chunk in in_df.chunks:
            out_op = op.copy().reset_key()
            out_chunk = out_op.new_chunk([in_chunk], shape=in_chunk.shape, index=in_chunk.index,
                                     index_value=in_chunk.index_value, columns_value=in_chunk.columns)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, out_df.shape, dtypes=out_df.dtypes,
                                     index_value=out_df.index_value,
                                     columns_value=out_df.columns,
                                     chunks=out_chunks, nsplits=in_df.nsplits)

    @classproperty
    def _operator(self):
        raise NotImplementedError

    def __call__(self, df):
        return self.new_dataframe([df], df.shape, dtypes=df.dtypes,
                                  columns_value=df.columns, index_value=df.index_value)
