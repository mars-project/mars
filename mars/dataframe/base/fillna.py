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

from ... import opcodes
from ...config import options
from ...core import Base, Entity
from ...operands import OperandStage
from ...serialize import StringField, AnyField, BoolField, Int64Field, NDArrayField
from ..align import align_dataframe_dataframe, align_series_series
from ..core import DATAFRAME_TYPE
from ..operands import DataFrameOperandMixin, DataFrameOperand, ObjectType


class FillNA(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.FILL_NAN

    _value = AnyField('value')
    _method = StringField('method')
    _axis = AnyField('axis')
    _limit = Int64Field('limit')
    _downcast = AnyField('downcast')
    _use_inf_as_na = BoolField('use_inf_as_na')

    _output_limit = Int64Field('output_limit')
    _prev_chunk_sizes = NDArrayField('prev_chunk_sizes')

    def __init__(self, value=None, method=None, axis=None, limit=None, downcast=None,
                 use_inf_as_na=None, sparse=None, stage=None, gpu=None, object_type=None,
                 output_limit=None, prev_chunk_sizes=None, **kw):
        super().__init__(_value=value, _method=method, _axis=axis, _limit=limit, _downcast=downcast,
                         _use_inf_as_na=use_inf_as_na, _sparse=sparse, _stage=stage, _gpu=gpu,
                         _object_type=object_type, _output_limit=output_limit,
                         _prev_chunk_sizes=prev_chunk_sizes, **kw)

    @property
    def value(self):
        return self._value

    @property
    def method(self):
        return self._method

    @property
    def axis(self):
        return self._axis

    @property
    def limit(self):
        return self._limit

    @property
    def downcast(self):
        return self._downcast

    @property
    def prev_chunk_sizes(self):
        return self._prev_chunk_sizes

    @property
    def use_inf_as_na(self):
        return self._use_inf_as_na

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        input_iter = iter(inputs)
        next(input_iter)
        if isinstance(self._value, (Base, Entity)):
            self._value = next(input_iter)

    @property
    def output_limit(self):
        return self._output_limit or 1

    @staticmethod
    def _get_first_slice(op, df, end):
        if op.method == 'bfill':
            if op.object_type == ObjectType.series:
                return df.iloc[:end]
            else:
                if op.axis == 1:
                    return df.iloc[:, :end]
                else:
                    return df.iloc[:end, :]
        else:
            if op.object_type == ObjectType.series:
                return df.iloc[-end:]
            else:
                if op.axis == 1:
                    return df.iloc[:, -end:]
                else:
                    return df.iloc[-end:, :]

    @staticmethod
    def _get_last_slice(op, df, start):
        if op.object_type == ObjectType.series:
            return df.iloc[start:]
        else:
            if op.axis == 1:
                return df.iloc[:, start:]
            else:
                return df.iloc[start:, :]

    @classmethod
    def _execute_map(cls, ctx, op):
        input_data = ctx[op.inputs[0].key]
        limit = op.limit
        axis = op.axis
        method = op.method

        filled = input_data.fillna(method=op.method, axis=axis, limit=limit, downcast=op.downcast)
        last_slice = ctx[op.outputs[0].key] = cls._get_first_slice(op, filled, 1)
        del filled

        # generate remaining limits
        if limit is not None:
            if op.object_type == ObjectType.dataframe:
                if op.axis == 1:
                    idx = input_data.dtypes.index
                else:
                    idx = input_data.index
                idx_data = np.arange(1, 1 + len(idx))
                idx_series = pd.Series(idx_data, index=idx)
                max_valid = (~input_data.isna()).mul(idx_series, axis=axis).max(axis=axis)
                remain_nans = len(idx) - max_valid
                remain_limits = limit - remain_nans
                remain_limits[last_slice.iloc[0].isna() | (remain_limits <= 0)] = 0
                ctx[op.outputs[1].key] = remain_limits
            else:
                if last_slice.isna()[-1]:
                    remain_limits = np.nan
                else:
                    remain_nans = len(input_data) - 1 - (~input_data.isna()).argmax()
                    remain_limits = max(0, limit - remain_nans)
                ctx[op.outputs[1].key] = pd.Series([remain_limits])

    @classmethod
    def _execute_combine(cls, ctx, op):
        axis = op.axis
        method = op.method
        limit = op.limit

        input_data = ctx[op.inputs[0].key]
        if limit is not None:
            n_summaries = (len(op.inputs) - 1) // 2
            summaries = [ctx[inp.key] for inp in op.inputs[1:1 + n_summaries]]
            limits = [ctx[inp.key] for inp in op.inputs[1 + n_summaries:]]
        else:
            summaries = [ctx[inp.key] for inp in op.inputs[1:]]
            limits = []

        if not summaries:
            ctx[op.outputs[0].key] = input_data.fillna(method=method, axis=axis, limit=limit,
                                                       downcast=op.downcast)
            return

        valid_summary = cls._get_first_slice(
            op, pd.concat(summaries, axis=axis).fillna(method=method, axis=axis), 1)

        if not limits:
            summary_df = valid_summary
            cut_size = 1
        else:
            prev_chunk_sizes = op.prev_chunk_sizes
            valid_limit = limits[0]
            add_range = range(1, len(limits)) if method == 'ffill' else range(len(limits) - 2, -1, -1)
            for idx in add_range:
                valid_limit -= prev_chunk_sizes[idx]
                valid_limit[valid_limit < 0] = 0
                valid_limit += limits[idx]
            valid_limit = valid_limit.to_numpy()

            summary_df = input_data.copy()
            valid_poses = np.where(valid_limit > 0)
            if axis == 0:
                valid_pos_limit = len(summary_df.index) - limit + valid_limit[valid_limit > 0]
                summary_df.iloc[:, valid_poses] = np.nan
                summary_df.iloc[valid_pos_limit, valid_poses] = valid_summary[valid_limit > 0]
                cut_size = len(input_data.index)
            else:
                valid_pos_limit = len(summary_df.dtypes.index) - limit + valid_limit[valid_limit > 0]
                summary_df.iloc[valid_poses, :] = np.nan
                summary_df.iloc[valid_poses, valid_pos_limit] = valid_summary[valid_limit > 0]
                cut_size = len(input_data.dtypes.index)

        if method == 'bfill':
            concat_df = pd.concat([input_data, summary_df], axis=axis)
        else:
            concat_df = pd.concat([summary_df, input_data], axis=axis)
        concat_df.fillna(method=method, axis=axis, inplace=True, limit=limit,
                         downcast=op.downcast)

        ctx[op.outputs[0].key] = cls._get_first_slice(op, concat_df, -cut_size)

    @classmethod
    def execute(cls, ctx, op):
        try:
            pd.set_option('mode.use_inf_as_na', op.use_inf_as_na)
            if op.stage == OperandStage.map:
                cls._execute_map(ctx, op)
            elif op.stage == OperandStage.combine:
                cls._execute_combine(ctx, op)
            else:
                input_data = ctx[op.inputs[0].key]
                value = getattr(op, 'value', None)
                if isinstance(op.value, (Base, Entity)):
                    value = ctx[op.value.key]
                ctx[op.outputs[0].key] = input_data.fillna(
                    value=value, method=op.method, axis=op.axis, limit=op.limit, downcast=op.downcast)
        finally:
            pd.reset_option('mode.use_inf_as_na')

    @classmethod
    def _tile_one_by_one(cls, op):
        in_df = op.inputs[0]
        in_value_df = op.value if isinstance(op.value, (Base, Entity)) else None
        df = op.outputs[0]

        new_chunks = []
        for c in in_df.chunks:
            inputs = [c] if in_value_df is None else [c, in_value_df.chunks[0]]

            kw = dict(shape=c.shape, index=c.index, index_value=c.index_value)
            if op.object_type == ObjectType.dataframe:
                kw.update(dict(columns_value=c.columns_value, dtypes=c.dtypes))
            else:
                kw.update(dict(dtype=c.dtype))
            new_op = op.copy().reset_key()
            new_chunks.append(new_op.new_chunk(inputs, **kw))

        kw = dict(shape=df.shape, index_value=df.index_value)
        if op.object_type == ObjectType.dataframe:
            kw.update(dict(columns_value=df.columns_value, dtypes=df.dtypes))
        else:
            kw.update(dict(dtype=df.dtype))
        new_op = op.copy().reset_key()
        return new_op.new_tileables(op.inputs, shape=df.shape, nsplits=df.nsplits,
                                    chunks=new_chunks)

    @classmethod
    def _build_combine(cls, op, input_chunks, summary_chunks, limit_chunks, idx, forward=True):
        axis = op.axis
        c = input_chunks[idx]

        to_concat_chunks = [c]

        idx_range = list(range(idx) if forward else range(idx + 1, len(summary_chunks)))
        prev_chunk_sizes = np.empty(shape=(len(idx_range),), dtype=np.dtype('int64'))
        for i, j in enumerate(idx_range):
            prev_chunk_sizes[i] = input_chunks[j].shape[axis]
            to_concat_chunks.append(summary_chunks[j])
        if len(limit_chunks):
            for j in idx_range:
                to_concat_chunks.append(limit_chunks[j])

        new_chunk_op = op.copy().reset_key()
        new_chunk_op._stage = OperandStage.combine
        new_chunk_op._prev_chunk_sizes = prev_chunk_sizes
        if new_chunk_op.object_type == ObjectType.dataframe:
            return new_chunk_op.new_chunk(to_concat_chunks, shape=c.shape, dtypes=c.dtypes,
                                          index=c.index, index_value=c.index_value,
                                          columns_value=c.columns_value)
        else:
            return new_chunk_op.new_chunk(to_concat_chunks, shape=c.shape, dtype=c.dtype, index=c.index,
                                          index_value=c.index_value)

    @classmethod
    def _tile_directional_dataframe(cls, op):
        in_df = op.inputs[0]
        df = op.outputs[0]
        forward = op.method == 'ffill'

        n_rows, n_cols = in_df.chunk_shape

        # map to get individual results and summaries
        src_chunks = np.empty(in_df.chunk_shape, dtype=np.object)
        summary_chunks = np.empty(in_df.chunk_shape, dtype=np.object)
        if op.limit is not None:
            limit_chunks = np.empty(in_df.chunk_shape, dtype=np.object)
        else:
            limit_chunks = []
        for c in in_df.chunks:
            new_chunk_op = op.copy().reset_key()
            new_chunk_op._stage = OperandStage.map
            if op.axis == 1:
                summary_shape = (c.shape[0], 1)
            else:
                summary_shape = (1, c.shape[1])
            src_chunks[c.index] = c
            kws = [dict(shape=summary_shape, dtypes=df.dtypes)]
            if op.limit is not None:
                new_chunk_op._output_limit = 2
                kws.append(dict(shape=(1, c.shape[1]), dtype=np.dtype('int64')))
                summary_chunks[c.index], limit_chunks[c.index] = new_chunk_op.new_chunks([c], kws=kws)
            else:
                summary_chunks[c.index], = new_chunk_op.new_chunks([c], kws=kws)

        # combine summaries into results
        output_chunk_array = np.empty(in_df.chunk_shape, dtype=np.object)
        if op.axis == 1:
            for row in range(n_rows):
                row_src = src_chunks[row, :]
                row_summaries = summary_chunks[row, :]
                if len(limit_chunks):
                    row_limits = limit_chunks[row, :]
                else:
                    row_limits = []
                for col in range(n_cols):
                    output_chunk_array[row, col] = cls._build_combine(
                        op, row_src, row_summaries, row_limits, col, forward)
        else:
            for col in range(n_cols):
                col_src = src_chunks[:, col]
                col_summaries = summary_chunks[:, col]
                if len(limit_chunks):
                    col_limits = limit_chunks[:, col]
                else:
                    col_limits = []
                for row in range(n_rows):
                    output_chunk_array[row, col] = cls._build_combine(
                        op, col_src, col_summaries, col_limits, row, forward)

        output_chunks = list(output_chunk_array.reshape((n_rows * n_cols,)))
        new_op = op.copy().reset_key()
        return new_op.new_tileables(op.inputs, shape=in_df.shape, nsplits=in_df.nsplits,
                                    chunks=output_chunks, dtypes=df.dtypes,
                                    index_value=df.index_value, columns_value=df.columns_value)

    @classmethod
    def _tile_directional_series(cls, op):
        in_series = op.inputs[0]
        series = op.outputs[0]
        forward = op.method == 'ffill'

        # map to get individual results and summaries
        summary_chunks = np.empty(in_series.chunk_shape, dtype=np.object)
        if op.limit is not None:
            limit_chunks = np.empty(in_series.chunk_shape, dtype=np.object)
        else:
            limit_chunks = []
        for c in in_series.chunks:
            new_chunk_op = op.copy().reset_key()
            new_chunk_op._stage = OperandStage.map
            kws = [dict(shape=(1,), dtype=series.dtype)]
            if op.limit is not None:
                kws.append(dict(shape=(1,), dtype=np.dtype('int64')))
                summary_chunks[c.index], limit_chunks[c.index] = new_chunk_op.new_chunks([c], kws=kws)
            else:
                summary_chunks[c.index], = new_chunk_op.new_chunks([c], kws=kws)

        # combine summaries into results
        output_chunks = [
            cls._build_combine(op, in_series.chunks, summary_chunks, limit_chunks, i, forward)
            for i in range(len(in_series.chunks))
        ]
        new_op = op.copy().reset_key()
        return new_op.new_tileables(op.inputs, shape=in_series.shape, nsplits=in_series.nsplits,
                                    chunks=output_chunks, dtype=series.dtype,
                                    index_value=series.index_value)

    @classmethod
    def _tile_both_dataframes(cls, op):
        in_df = op.inputs[0]
        in_value = op.inputs[1]
        df = op.outputs[0]

        nsplits, out_shape, left_chunks, right_chunks = align_dataframe_dataframe(in_df, in_value)
        out_chunk_indexes = itertools.product(*(range(s) for s in out_shape))

        out_chunks = []
        for idx, left_chunk, right_chunk in zip(out_chunk_indexes, left_chunks, right_chunks):
            out_chunk = op.copy().reset_key().new_chunk([left_chunk, right_chunk],
                                                        shape=(np.nan, np.nan), index=idx)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, df.shape,
                                     nsplits=tuple(tuple(ns) for ns in nsplits),
                                     chunks=out_chunks, dtypes=df.dtypes,
                                     index_value=df.index_value, columns_value=df.columns_value)

    @classmethod
    def _tile_both_series(cls, op):
        left, right = op.lhs, op.rhs
        df = op.outputs[0]

        nsplits, out_shape, left_chunks, right_chunks = align_series_series(left, right)

        out_chunks = []
        for idx, left_chunk, right_chunk in zip(range(out_shape[0]), left_chunks, right_chunks):
            out_chunk = op.copy().reset_key().new_chunk([left_chunk, right_chunk],
                                                        shape=(np.nan,), index=(idx,))
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_seriess(op.inputs, df.shape,
                                  nsplits=tuple(tuple(ns) for ns in nsplits),
                                  chunks=out_chunks, dtype=df.dtype,
                                  index_value=df.index_value, name=df.name)

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        if len(in_df.chunks) == 1 and \
                (not isinstance(op.value, (Base, Entity)) or len(op.value.chunks) == 1):
            return cls._tile_one_by_one(op)
        elif op.method is not None:
            if op.object_type == ObjectType.dataframe:
                return cls._tile_directional_dataframe(op)
            else:
                return cls._tile_directional_series(op)
        elif not isinstance(op.value, (Base, Entity)):
            return cls._tile_one_by_one(op)
        elif isinstance(op.value, DATAFRAME_TYPE):
            return cls._tile_both_dataframes(op)
        elif op.object_type == ObjectType.dataframe:
            return cls._tile_dataframe_series(op)
        else:
            return cls._tile_both_series(op)

    def __call__(self, a, value_df=None):
        method = getattr(self, 'method', None)
        if method == 'backfill':
            method = 'bfill'
        if method == 'pad':
            method = 'ffill'
        self._method = method
        axis = getattr(self, 'axis', None) or 0
        if axis == 'index':
            axis = 0
        if axis == 'columns':
            axis = 1
        self._axis = axis

        inputs = [a]
        if value_df is not None:
            inputs.append(value_df)
        if isinstance(a, DATAFRAME_TYPE):
            return self.new_dataframe(inputs, shape=a.shape, dtypes=a.dtypes, index_value=a.index_value,
                                      columns_value=a.columns_value)
        else:
            return self.new_series(inputs, shape=a.shape, dtypes=a.dtypes, index_value=a.index_value)


def _validate_fillna_kwargs(value, method, validate_scalar_dict_value=True):
    from pandas.core.missing import clean_fill_method

    if value is None and method is None:
        raise ValueError("Must specify a fill 'value' or 'method'.")
    elif value is None and method is not None:
        method = clean_fill_method(method)

    elif value is not None and method is None:
        if validate_scalar_dict_value and isinstance(value, (list, tuple)):
            raise TypeError(
                '"value" parameter must be a scalar or dict, but '
                'you passed a "%s"' % type(value).__name__
            )

    elif value is not None and method is not None:
        raise ValueError("Cannot specify both 'value' and 'method'.")

    return value, method


def fillna(df, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
    _validate_fillna_kwargs(value, method)
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = FillNA(value=value, method=method, axis=axis, limit=limit, downcast=downcast,
                use_inf_as_na=use_inf_as_na, object_type=df.op.object_type)
    out_df = op(df)
    if inplace:
        df.data = out_df.data
    else:
        return out_df


def ffill(df, axis=None, inplace=False, limit=None, downcast=None):
    return fillna(df, method='ffill', axis=axis, inplace=inplace, limit=limit, downcast=downcast)


def bfill(df, axis=None, inplace=False, limit=None, downcast=None):
    return fillna(df, method='bfill', axis=axis, inplace=inplace, limit=limit, downcast=downcast)
