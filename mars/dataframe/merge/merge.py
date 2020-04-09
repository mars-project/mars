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
import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...operands import OperandStage
from ...serialize import AnyField, BoolField, StringField, TupleField, KeyField, Int32Field
from ...utils import get_shuffle_input_keys_idxes
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType, \
    DataFrameMapReduceOperand, DataFrameShuffleProxy
from ..utils import build_concatenated_rows_frame, build_df, parse_index, hash_dataframe_on, \
    infer_index_value

import logging
logger = logging.getLogger(__name__)


class DataFrameMergeAlign(DataFrameMapReduceOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_SHUFFLE_MERGE_ALIGN

    _index_shuffle_size = Int32Field('index_shuffle_size')
    _shuffle_on = AnyField('shuffle_on')

    _input = KeyField('input')

    def __init__(self, index_shuffle_size=None, shuffle_on=None, sparse=None,
                 stage=None, shuffle_key=None, **kw):
        super().__init__(
            _index_shuffle_size=index_shuffle_size, _shuffle_on=shuffle_on,
            _sparse=sparse, _object_type=ObjectType.dataframe, _stage=stage,
            _shuffle_key=shuffle_key, **kw)

    @property
    def index_shuffle_size(self):
        return self._index_shuffle_size

    @property
    def shuffle_on(self):
        return self._shuffle_on

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    @classmethod
    def execute_map(cls, ctx, op):
        chunk = op.outputs[0]
        df = ctx[op.inputs[0].key]
        shuffle_on = op.shuffle_on

        if shuffle_on is not None:
            # shuffle on field may be resident in index
            to_reset_index_names = []
            if not isinstance(shuffle_on, (list, tuple)):
                if shuffle_on not in df.dtypes:
                    to_reset_index_names.append(shuffle_on)
            else:
                for son in shuffle_on:
                    if son not in df.dtypes:
                        to_reset_index_names.append(shuffle_on)
            if len(to_reset_index_names) > 0:
                df = df.reset_index(to_reset_index_names)

        filters = hash_dataframe_on(df, shuffle_on, op.index_shuffle_size)

        # shuffle on index
        for index_idx, index_filter in enumerate(filters):
            group_key = ','.join([str(index_idx), str(chunk.index[1])])
            if index_filter is not None and index_filter is not list():
                ctx[(chunk.key, group_key)] = df.loc[index_filter]
            else:
                ctx[(chunk.key, group_key)] = None

    @classmethod
    def execute_reduce(cls, ctx, op):
        chunk = op.outputs[0]
        input_keys, input_idxes = get_shuffle_input_keys_idxes(op.inputs[0])
        input_idx_to_df = {idx: ctx[inp_key, ','.join(str(ix) for ix in chunk.index)]
                           for inp_key, idx in zip(input_keys, input_idxes)}
        row_idxes = sorted({idx[0] for idx in input_idx_to_df})

        res = []
        for row_idx in row_idxes:
            row_df = input_idx_to_df.get((row_idx, 0), None)
            if row_df is not None:
                res.append(row_df)
        ctx[chunk.key] = pd.concat(res, axis=0)

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls.execute_map(ctx, op)
        else:
            cls.execute_reduce(ctx, op)


class _DataFrameMergeBase(DataFrameOperand, DataFrameOperandMixin):
    _how = StringField('how')
    _on = AnyField('on')
    _left_on = AnyField('left_on')
    _right_on = AnyField('right_on')
    _left_index = BoolField('left_index')
    _right_index = BoolField('right_index')
    _sort = BoolField('sort')
    _suffixes = TupleField('suffixes')
    _copy = BoolField('copy')
    _indicator = BoolField('indicator')
    _validate = AnyField('validate')

    def __init__(self, how=None, on=None, left_on=None, right_on=None,
                 left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'),
                 copy=True, indicator=False, validate=None, sparse=False, object_type=None, **kw):
        super().__init__(
            _how=how, _on=on, _left_on=left_on, _right_on=right_on, _left_index=left_index, _right_index=right_index,
            _sort=sort, _suffixes=suffixes, _copy=copy, _indicator=indicator, _validate=validate,
            _sparse=sparse, _object_type=object_type, **kw)

    @property
    def how(self):
        return self._how

    @property
    def on(self):
        return self._on

    @property
    def left_on(self):
        return self._left_on

    @property
    def right_on(self):
        return self._right_on

    @property
    def left_index(self):
        return self._left_index

    @property
    def right_index(self):
        return self._right_index

    @property
    def sort(self):
        return self._sort

    @property
    def suffixes(self):
        return self._suffixes

    @property
    def copy_(self):
        return self._copy

    @property
    def indicator(self):
        return self._indicator

    @property
    def validate(self):
        return self._validate

    def __call__(self, left, right):
        empty_left, empty_right = build_df(left), build_df(right)
        # this `merge` will check whether the combination of those arguments is valid
        merged = empty_left.merge(empty_right, how=self.how, on=self.on,
                                  left_on=self.left_on, right_on=self.right_on,
                                  left_index=self.left_index, right_index=self.right_index,
                                  sort=self.sort, suffixes=self.suffixes,
                                  copy=self.copy_, indicator=self.indicator, validate=self.validate)

        # the `index_value` doesn't matter.
        index_tokenize_objects = [left, right, self.how, self.left_on,
                                  self.right_on, self.left_index, self.right_index]
        return self.new_dataframe([left, right], shape=(np.nan, merged.shape[1]), dtypes=merged.dtypes,
                                  index_value=parse_index(merged.index, *index_tokenize_objects),
                                  columns_value=parse_index(merged.columns, store_data=True))


class DataFrameShuffleMerge(_DataFrameMergeBase):
    _op_type_ = OperandDef.DATAFRAME_SHUFFLE_MERGE

    def __init__(self, **kw):
        super().__init__(**kw)

    @classmethod
    def _gen_shuffle_chunks(cls, op, out_shape, shuffle_on, df):
        # gen map chunks
        map_chunks = []
        for chunk in df.chunks:
            map_op = DataFrameMergeAlign(stage=OperandStage.map, shuffle_on=shuffle_on,
                                         sparse=chunk.issparse(),
                                         index_shuffle_size=out_shape[0])
            map_chunks.append(map_op.new_chunk([chunk], shape=(np.nan, np.nan), dtypes=chunk.dtypes, index=chunk.index,
                                               index_value=chunk.index_value, columns_value=chunk.columns_value))

        proxy_chunk = DataFrameShuffleProxy(object_type=ObjectType.dataframe).new_chunk(
            map_chunks, shape=(), dtypes=df.dtypes,
            index_value=df.index_value, columns_value=df.columns_value)

        # gen reduce chunks
        reduce_chunks = []
        for out_idx in itertools.product(*(range(s) for s in out_shape)):
            reduce_op = DataFrameMergeAlign(stage=OperandStage.reduce, sparse=proxy_chunk.issparse(),
                                            shuffle_key=','.join(str(idx) for idx in out_idx))
            reduce_chunks.append(
                reduce_op.new_chunk([proxy_chunk], shape=(np.nan, np.nan), dtypes=proxy_chunk.dtypes, index=out_idx,
                                    index_value=proxy_chunk.index_value, columns_value=proxy_chunk.columns_value))
        return reduce_chunks

    @classmethod
    def _tile_one_chunk(cls, op, left, right):
        df = op.outputs[0]
        if len(left.chunks) == 1 and len(right.chunks) == 1:
            merge_op = op.copy().reset_key()
            out_chunk = merge_op.new_chunk([left.chunks[0], right.chunks[0]],
                                           shape=df.shape,
                                           index=left.chunks[0].index,
                                           index_value=df.index_value,
                                           dtypes=df.dtypes,
                                           columns_value=df.columns_value)
            out_chunks = [out_chunk]
            nsplits = ((np.nan,), (df.shape[1],))
        elif len(left.chunks) == 1:
            out_chunks = []
            left_chunk = left.chunks[0]
            for c in right.chunks:
                merge_op = op.copy().reset_key()
                out_chunk = merge_op.new_chunk([left_chunk, c],
                                               shape=(np.nan, df.shape[1]),
                                               index=c.index,
                                               index_value=infer_index_value(left_chunk.index_value,
                                                                             c.index_value),
                                               dtypes=df.dtypes,
                                               columns_value=df.columns_value)
                out_chunks.append(out_chunk)
            nsplits = ((np.nan,) * len(right.chunks), (df.shape[1],))
        else:
            out_chunks = []
            right_chunk = right.chunks[0]
            for c in left.chunks:
                merge_op = op.copy().reset_key()
                out_chunk = merge_op.new_chunk([c, right_chunk],
                                               shape=(np.nan, df.shape[1]),
                                               index=c.index,
                                               index_value=infer_index_value(right_chunk.index_value,
                                                                             c.index_value),
                                               dtypes=df.dtypes,
                                               columns_value=df.columns_value)
                out_chunks.append(out_chunk)
            nsplits = ((np.nan,) * len(left.chunks), (df.shape[1],))

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, df.shape,
                                     nsplits=nsplits,
                                     chunks=out_chunks, dtypes=df.dtypes,
                                     index_value=df.index_value, columns_value=df.columns_value)

    @classmethod
    def tile(cls, op):
        df = op.outputs[0]
        left = build_concatenated_rows_frame(op.inputs[0])
        right = build_concatenated_rows_frame(op.inputs[1])

        if len(left.chunks) == 1 or len(right.chunks) == 1:
            return cls._tile_one_chunk(op, left, right)

        left_row_chunk_size = left.chunk_shape[0]
        right_row_chunk_size = right.chunk_shape[0]
        out_row_chunk_size = max(left_row_chunk_size, right_row_chunk_size)

        out_chunk_shape = (out_row_chunk_size, 1)
        nsplits = [[np.nan for _ in range(out_row_chunk_size)], [df.shape[1]]]

        left_on = _prepare_shuffle_on(op.left_index, op.left_on, op.on)
        right_on = _prepare_shuffle_on(op.right_index, op.right_on, op.on)

        # do shuffle
        left_chunks = cls._gen_shuffle_chunks(op, out_chunk_shape, left_on, left)
        right_chunks = cls._gen_shuffle_chunks(op, out_chunk_shape, right_on, right)

        out_chunks = []
        for left_chunk, right_chunk in zip(left_chunks, right_chunks):
            merge_op = op.copy().reset_key()
            out_chunk = merge_op.new_chunk([left_chunk, right_chunk], shape=(np.nan, df.shape[1]),
                                           index=left_chunk.index,
                                           index_value=infer_index_value(left_chunk.index_value,
                                                                         right_chunk.index_value),
                                           columns_value=df.columns_value)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, df.shape,
                                     nsplits=tuple(tuple(ns) for ns in nsplits),
                                     chunks=out_chunks, dtypes=df.dtypes,
                                     index_value=df.index_value, columns_value=df.columns_value)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        left, right = ctx[op.inputs[0].key], ctx[op.inputs[1].key]

        def execute_merge(x, y):
            if not op.gpu:
                kwargs = dict(copy=op.copy, validate=op.validate, indicator=op.indicator)
            else:  # pragma: no cover
                # cudf doesn't support 'validate' and 'copy'
                kwargs = dict(indicator=op.indicator)
            return x.merge(y, how=op.how, on=op.on,
                           left_on=op.left_on, right_on=op.right_on,
                           left_index=op.left_index, right_index=op.right_index,
                           sort=op.sort, suffixes=op.suffixes, **kwargs)

        # workaround for: https://github.com/pandas-dev/pandas/issues/27943
        try:
            r = execute_merge(left, right)
        except ValueError:
            r = execute_merge(left.copy(deep=True), right.copy(deep=True))

        # make sure column's order
        if not all(n1 == n2 for n1, n2 in zip(chunk.columns_value.to_pandas(), r.columns)):
            r = r[list(chunk.columns_value.to_pandas())]
        ctx[chunk.key] = r


def _prepare_shuffle_on(use_index, side_on, on):
    # consistent with pandas: `left_index` precedes `left_on` and `right_index` precedes `right_on`
    if use_index:
        # `None` means we will shuffle on df.index.
        return None
    elif side_on is not None:
        return side_on
    else:
        return on


def merge(df, right, how='inner', on=None, left_on=None, right_on=None,
          left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'),
          copy=True, indicator=False, strategy=None, validate=None):
    if strategy is not None and strategy != 'shuffle':
        raise NotImplementedError('Only shuffle merge is supported')
    op = DataFrameShuffleMerge(
        how=how, on=on, left_on=left_on, right_on=right_on,
        left_index=left_index, right_index=right_index, sort=sort, suffixes=suffixes,
        copy=copy, indicator=indicator, validate=validate, object_type=ObjectType.dataframe)
    return op(df, right)


def join(df, other, on=None, how='left', lsuffix='', rsuffix='', sort=False, strategy=None):
    return merge(df, other, left_on=on, how=how, left_index=on is None, right_index=True,
                 suffixes=(lsuffix, rsuffix), sort=sort, strategy=strategy)
