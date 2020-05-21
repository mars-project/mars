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

from collections import OrderedDict

import numpy as np
import pandas as pd

from ...config import options
from ...operands import OperandStage
from ...utils import lazy_import
from ...serialize import BoolField, AnyField, DataTypeField, Int32Field
from ..utils import parse_index, build_df, build_empty_df, build_series, validate_axis
from ..operands import DataFrameOperandMixin, DataFrameOperand, ObjectType, DATAFRAME_TYPE
from ..merge import DataFrameConcat

cudf = lazy_import('cudf', globals=globals())


class DataFrameReductionOperand(DataFrameOperand):
    _axis = AnyField('axis')
    _skipna = BoolField('skipna')
    _level = AnyField('level')
    _numeric_only = BoolField('numeric_only')
    _min_count = Int32Field('min_count')
    _use_inf_as_na = BoolField('use_inf_as_na')

    _dtype = DataTypeField('dtype')
    _combine_size = Int32Field('combine_size')

    def __init__(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=None,
                 stage=None, dtype=None, combine_size=None, gpu=None, sparse=None, object_type=None,
                 use_inf_as_na=None, **kw):
        super().__init__(_axis=axis, _skipna=skipna, _level=level, _numeric_only=numeric_only,
                         _min_count=min_count, _stage=stage, _dtype=dtype, _combine_size=combine_size,
                         _gpu=gpu, _sparse=sparse, _object_type=object_type, _use_inf_as_na=use_inf_as_na,
                         **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def skipna(self):
        return self._skipna

    @property
    def level(self):
        return self._level

    @property
    def numeric_only(self):
        return self._numeric_only

    @property
    def min_count(self):
        return self._min_count

    @property
    def dtype(self):
        return self._dtype

    @property
    def combine_size(self):
        return self._combine_size

    @property
    def use_inf_as_na(self):
        return self._use_inf_as_na


class DataFrameCumReductionOperand(DataFrameOperand):
    _axis = AnyField('axis')
    _skipna = BoolField('skipna')
    _use_inf_as_na = BoolField('use_inf_as_na')

    _dtype = DataTypeField('dtype')

    def __init__(self, axis=None, skipna=None, dtype=None, gpu=None, sparse=None,
                 object_type=None, use_inf_as_na=None, stage=None, **kw):
        super().__init__(_axis=axis, _skipna=skipna, _dtype=dtype, _gpu=gpu, _sparse=sparse,
                         _object_type=object_type, _stage=stage, _use_inf_as_na=use_inf_as_na, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def skipna(self):
        return self._skipna

    @property
    def dtype(self):
        return self._dtype

    @property
    def use_inf_as_na(self):
        return self._use_inf_as_na


class DataFrameReductionMixin(DataFrameOperandMixin):
    @classmethod
    def _tile_one_chunk(cls, op):
        df = op.outputs[0]

        chk = op.inputs[0].chunks[0]
        chunk_params = {k: v for k, v in chk.params.items()
                        if k in df.params}
        chunk_params['shape'] = df.shape
        chunk_params['index'] = chk.index
        if op.object_type == ObjectType.series:
            chunk_params.update(dict(dtype=df.dtype, index_value=df.index_value, index=(0,)))
        elif op.object_type == ObjectType.dataframe:
            chunk_params.update(dict(dtypes=df.dtypes, index_value=df.index_value,
                                     columns_value=df.columns_value))
        else:
            chunk_params.update(dict(dtype=df.dtype, order=df.order))
        new_chunk_op = op.copy().reset_key()
        chunk = new_chunk_op.new_chunk(op.inputs[0].chunks, kws=[chunk_params])

        new_op = op.copy()
        nsplits = tuple((s,) for s in chunk.shape)
        params = df.params.copy()
        params.update(dict(chunks=[chunk], nsplits=nsplits))
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def _tree_reduction(cls, chunks, op, combine_size, idx):
        while len(chunks) > combine_size:
            new_chunks = []
            for i in range(0, len(chunks), combine_size):
                chks = chunks[i: i + combine_size]
                for j, c in enumerate(chunks):
                    c._index = (j,)

                # concatenate chunks into one chunk
                concat_op = DataFrameConcat(axis=op.axis, object_type=ObjectType.dataframe)
                if op.axis == 0:
                    concat_index = parse_index(pd.RangeIndex(len(chks)))
                    concat_dtypes = chks[0].dtypes
                    concat_shape = (sum([c.shape[0] for c in chks]), chks[0].shape[1])
                else:
                    concat_index = chks[0].index_value
                    concat_dtypes = pd.Series([c.dtypes[0] for c in chks])
                    concat_shape = (chks[0].shape[0], (sum([c.shape[1] for c in chks])))
                chk = concat_op.new_chunk(chks, shape=concat_shape, index=(i,),
                                          dtypes=concat_dtypes, index_value=concat_index)

                # do reduction
                if op.axis == 0:
                    reduced_shape = (1, chk.shape[1])
                    index_value = parse_index(pd.RangeIndex(1))
                    dtypes = chk.dtypes
                else:
                    reduced_shape = (chk.shape[0], 1)
                    index_value = chk.index_value
                    dtypes = pd.Series(op.outputs[0].dtype)
                new_op = op.copy().reset_key()
                new_op._stage = OperandStage.combine
                # all intermediate results' type is dataframe
                new_op._object_type = ObjectType.dataframe
                new_chunks.append(new_op.new_chunk([chk], shape=reduced_shape, index=(i,), dtypes=dtypes,
                                                   index_value=index_value))
            chunks = new_chunks

        if op.axis == 0:
            concat_shape = (sum([c.shape[0] for c in chunks]), chunks[0].shape[1])
        else:
            concat_shape = (chunks[0].shape[0], (sum([c.shape[1] for c in chunks])))
        concat_op = DataFrameConcat(axis=op.axis, object_type=ObjectType.dataframe)
        chk = concat_op.new_chunk(chunks, index=(idx,), shape=concat_shape)
        empty_df = build_df(chunks[0])
        reduced_df = cls._execute_reduction(empty_df, op)
        reduced_shape = (chk.shape[0],) if op.axis == 1 else reduced_df.shape
        new_op = op.copy().reset_key()
        new_op._stage = OperandStage.agg
        return new_op.new_chunk([chk], shape=reduced_shape, index=(idx,), dtype=reduced_df.dtype,
                                index_value=parse_index(reduced_df.index))

    @classmethod
    def _tile_dataframe(cls, op):
        in_df = op.inputs[0]
        df = op.outputs[0]
        combine_size = op.combine_size or options.combine_size

        n_rows, n_cols = in_df.chunk_shape

        chunk_dtypes = []
        if op.numeric_only and op.axis == 0:
            cum_nsplits = np.cumsum((0,) + in_df.nsplits[1])
            for i in range(len(cum_nsplits) - 1):
                chunk_empty_df = build_empty_df(in_df.dtypes[cum_nsplits[i]: cum_nsplits[i + 1]])
                chunk_dtypes.append(chunk_empty_df.select_dtypes([np.number, 'bool']).dtypes)

        # build reduction chunks
        reduction_chunks = np.empty(op.inputs[0].chunk_shape, dtype=np.object)
        for c in op.inputs[0].chunks:
            new_chunk_op = op.copy().reset_key()
            new_chunk_op._stage = OperandStage.map
            new_chunk_op._object_type = ObjectType.dataframe
            if op.axis == 0:
                if op.numeric_only:
                    dtypes = chunk_dtypes[c.index[1]]
                else:
                    dtypes = c.dtypes
                reduced_shape = (1, len(dtypes))
                index_value = parse_index(pd.RangeIndex(1), new_chunk_op)
            else:
                reduced_shape = (c.shape[0], 1)
                index_value = c.index_value
                dtypes = pd.Series(op.outputs[0].dtype)
            reduction_chunks[c.index] = new_chunk_op.new_chunk([c], shape=reduced_shape,
                                                               dtypes=dtypes, index_value=index_value)
        # Tree reduction
        out_chunks = []
        if op.axis is None or op.axis == 0:
            for col in range(n_cols):
                chunks = [reduction_chunks[i, col] for i in range(n_rows)]
                out_chunks.append(cls._tree_reduction(chunks, op, combine_size, col))
        elif op.axis == 1:
            for row in range(n_rows):
                chunks = [reduction_chunks[row, i] for i in range(n_cols)]
                out_chunks.append(cls._tree_reduction(chunks, op, combine_size, row))
        new_op = op.copy()
        nsplits = (tuple(c.shape[0] for c in out_chunks),)
        return new_op.new_seriess(op.inputs, df.shape, nsplits=nsplits, chunks=out_chunks,
                                  dtype=df.dtype, index_value=df.index_value)

    @classmethod
    def _tile_series(cls, op):
        series = op.outputs[0]
        combine_size = op.combine_size or options.combine_size

        chunks = np.empty(op.inputs[0].chunk_shape, dtype=np.object)
        for c in op.inputs[0].chunks:
            new_chunk_op = op.copy().reset_key()
            new_chunk_op._object_type = ObjectType.series
            new_chunk_op._stage = OperandStage.map
            chunks[c.index] = new_chunk_op.new_chunk([c], shape=(), dtype=series.dtype)

        while len(chunks) > combine_size:
            new_chunks = []
            for i in range(0, len(chunks), combine_size):
                chks = chunks[i: i + combine_size]
                concat_op = DataFrameConcat(object_type=ObjectType.series)
                length = sum([c.shape[0] for c in chks if len(c.shape) > 0])
                range_num = -1 if np.isnan(length) else length
                chk = concat_op.new_chunk(chks, shape=(length,), index=(i,), dtype=chks[0].dtype,
                                          index_value=parse_index(pd.RangeIndex(range_num), [c.key for c in chks]))
                new_op = op.copy().reset_key()
                new_op._object_type = ObjectType.series
                new_op._stage = OperandStage.combine
                new_chunks.append(new_op.new_chunk([chk], shape=(), index=(i,), dtype=chk.dtype,
                                                   index_value=parse_index(pd.RangeIndex(-1))))
            chunks = new_chunks

        concat_op = DataFrameConcat(object_type=ObjectType.series)
        length = sum([c.shape[0] for c in chunks if len(c.shape) > 0])
        range_num = -1 if np.isnan(length) else length
        chk = concat_op.new_chunk(chunks, shape=(length,), index=(0,), dtype=chunks[0].dtype,
                                  index_value=parse_index(pd.RangeIndex(range_num)))
        chunk_op = op.copy().reset_key()
        chunk_op._stage = OperandStage.agg

        chunk_params = series.params.copy()
        chunk_params.update(dict(index=(0,)))
        chunk = chunk_op.new_chunk([chk], **chunk_params)

        new_op = op.copy().reset_key()
        nsplits = tuple((s,) for s in chunk.shape)
        params = series.params.copy()
        params.update(dict(nsplits=nsplits, chunks=[chunk]))
        return new_op.new_tileables(op.inputs, **params)

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        if len(in_df.chunks) == 1:
            return cls._tile_one_chunk(op)
        if isinstance(in_df, DATAFRAME_TYPE):
            return cls._tile_dataframe(op)
        else:
            return cls._tile_series(op)

    @classmethod
    def _execute_reduction(cls, in_data, op, min_count=None, reduction_func=None):
        kwargs = dict()
        if op.axis is not None:
            kwargs['axis'] = op.axis
        if op.skipna is not None:
            kwargs['skipna'] = op.skipna
        if op.numeric_only is not None:
            kwargs['numeric_only'] = op.numeric_only
        if min_count is not None:
            kwargs['min_count'] = op.min_count
        reduction_func = reduction_func or getattr(cls, '_func_name')
        return getattr(in_data, reduction_func)(**kwargs)

    @classmethod
    def _execute_map_with_count(cls, ctx, op, reduction_func=None):
        # Execution with specified `min_count` in the map stage

        xdf = cudf if op.gpu else pd
        in_data = ctx[op.inputs[0].key]
        if isinstance(in_data, pd.Series):
            count = in_data.count()
        else:
            count = in_data.count(axis=op.axis, numeric_only=op.numeric_only)
        r = cls._execute_reduction(in_data, op, reduction_func=reduction_func)
        if isinstance(in_data, xdf.Series):
            ctx[op.outputs[0].key] = (r, count)
        else:
            # For dataframe, will keep dimensions for intermediate results.
            ctx[op.outputs[0].key] = (xdf.DataFrame(r), xdf.DataFrame(count)) if op.axis == 1 \
                else (xdf.DataFrame(r).transpose(), xdf.DataFrame(count).transpose())

    @classmethod
    def _execute_combine_with_count(cls, ctx, op, reduction_func=None):
        # Execution with specified `min_count` in the combine stage

        xdf = cudf if op.gpu else pd
        in_data, concat_count = ctx[op.inputs[0].key]
        count = concat_count.sum(axis=op.axis)
        r = cls._execute_reduction(in_data, op, reduction_func=reduction_func)
        if isinstance(in_data, xdf.Series):
            ctx[op.outputs[0].key] = (r, count)
        else:
            # For dataframe, will keep dimensions for intermediate results.
            ctx[op.outputs[0].key] = (xdf.DataFrame(r), xdf.DataFrame(count)) if op.axis == 1 \
                else (xdf.DataFrame(r).transpose(), xdf.DataFrame(count).transpose())

    @classmethod
    def _execute_agg_with_count(cls, ctx, op, reduction_func=None):
        # Execution with specified `min_count` in the aggregate stage

        # When specify `min_count`, the terminal chunk has two inputs one of which is for count.
        # The output should be determined by comparing `count` with `min_count`.
        in_data, concat_count = ctx[op.inputs[0].key]
        count = concat_count.sum(axis=op.axis)
        r = cls._execute_reduction(in_data, op, reduction_func=reduction_func)
        if np.isscalar(r):
            ctx[op.outputs[0].key] = np.nan if count < op.min_count else r
        else:
            r[count < op.min_count] = np.nan
            ctx[op.outputs[0].key] = r

    @classmethod
    def _execute_without_count(cls, ctx, op, reduction_func=None):
        # Execution for normal reduction operands.

        # For dataframe, will keep dimensions for intermediate results.
        xdf = cudf if op.gpu else pd
        in_data = ctx[op.inputs[0].key]
        r = cls._execute_reduction(in_data, op, min_count=op.min_count, reduction_func=reduction_func)
        if isinstance(in_data, xdf.Series) or op.object_type == ObjectType.series:
            ctx[op.outputs[0].key] = r
        else:
            if op.axis == 0:
                # cannot just do xdf.DataFrame(r).T
                # cuz the dtype will be object since pandas 1.0
                df = xdf.DataFrame(OrderedDict((d, [v]) for d, v in r.iteritems()))
            else:
                df = xdf.DataFrame(r)
            ctx[op.outputs[0].key] = df

    @classmethod
    def _execute_map(cls, ctx, op):
        if op.min_count is not None and op.min_count > 0:
            cls._execute_map_with_count(ctx, op)
        else:
            cls._execute_without_count(ctx, op)

    @classmethod
    def _execute_combine(cls, ctx, op):
        if op.min_count is not None and op.min_count > 0:
            cls._execute_combine_with_count(ctx, op)
        else:
            cls._execute_without_count(ctx, op)

    @classmethod
    def _execute_agg(cls, ctx, op):
        if op.min_count is not None and op.min_count > 0:
            cls._execute_agg_with_count(ctx, op)
        else:
            cls._execute_without_count(ctx, op)

    @classmethod
    def execute(cls, ctx, op):
        try:
            pd.set_option('mode.use_inf_as_na', op.use_inf_as_na)
            if op.stage == OperandStage.combine:
                cls._execute_combine(ctx, op)
            elif op.stage == OperandStage.agg:
                cls._execute_agg(ctx, op)
            elif op.stage == OperandStage.map:
                cls._execute_map(ctx, op)
            else:
                in_data = ctx[op.inputs[0].key]
                min_count = getattr(op, 'min_count', None)
                ctx[op.outputs[0].key] = cls._execute_reduction(in_data, op, min_count)
        finally:
            pd.reset_option('mode.use_inf_as_na')

    def _call_dataframe(self, df):
        axis = getattr(self, 'axis', None) or 0
        level = getattr(self, 'level', None)
        skipna = getattr(self, 'skipna', None)
        numeric_only = getattr(self, 'numeric_only', None)
        self._axis = axis = validate_axis(axis, df)
        # TODO: enable specify level if we support groupby
        if level is not None:
            raise NotImplementedError('Not support specify level now')

        empty_df = build_df(df)
        func_name = getattr(self, '_func_name')
        if func_name == 'count':
            reduced_df = getattr(empty_df, func_name)(axis=axis, level=level, numeric_only=numeric_only)
        elif func_name == 'nunique':
            reduced_df = getattr(empty_df, func_name)(axis=axis)
        else:
            reduced_df = getattr(empty_df, func_name)(axis=axis, level=level, skipna=skipna,
                                                      numeric_only=numeric_only)

        reduced_shape = (df.shape[0],) if axis == 1 else reduced_df.shape
        return self.new_series([df], shape=reduced_shape, dtype=reduced_df.dtype,
                               index_value=parse_index(reduced_df.index, store_data=axis == 0))

    def _call_series(self, series):
        level = getattr(self, 'level', None)
        axis = getattr(self, 'axis', None)
        skipna = getattr(self, 'skipna', None)
        numeric_only = getattr(self, 'numeric_only', None)
        if axis == 'index':
            axis = 0
        self._axis = axis
        # TODO: enable specify level if we support groupby
        if level is not None:
            raise NotImplementedError('Not support specified level now')

        empty_series = build_series(series)
        func_name = getattr(self, '_func_name')
        if func_name == 'count':
            reduced_series = empty_series.count(level=level)
        elif func_name == 'nunique':
            reduced_series = empty_series.nunique()
        else:
            reduced_series = getattr(empty_series, func_name)(axis=axis, level=level, skipna=skipna,
                                                              numeric_only=numeric_only)

        return self.new_scalar([series], dtype=np.array(reduced_series).dtype)

    def __call__(self, a):
        if isinstance(a, DATAFRAME_TYPE):
            return self._call_dataframe(a)
        else:
            return self._call_series(a)


class DataFrameCumReductionMixin(DataFrameOperandMixin):
    @classmethod
    def _tile_one_chunk(cls, op):
        df = op.outputs[0]
        params = df.params.copy()

        chk = op.inputs[0].chunks[0]
        chunk_params = {k: v for k, v in chk.params.items()
                        if k in df.params}
        chunk_params['shape'] = df.shape
        chunk_params['index'] = chk.index
        new_chunk_op = op.copy().reset_key()
        chunk = new_chunk_op.new_chunk(op.inputs[0].chunks, kws=[chunk_params])

        new_op = op.copy()
        nsplits = tuple((s,) for s in chunk.shape)
        params['chunks'] = [chunk]
        params['nsplits'] = nsplits
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def _build_combine(cls, op, input_chunks, summary_chunks, idx):
        c = input_chunks[idx]
        to_concat_chunks = [c]
        for j in range(idx):
            to_concat_chunks.append(summary_chunks[j])

        new_chunk_op = op.copy().reset_key()
        new_chunk_op._stage = OperandStage.combine
        return new_chunk_op.new_chunk(to_concat_chunks, **c.params)

    @classmethod
    def _tile_dataframe(cls, op):
        in_df = op.inputs[0]
        df = op.outputs[0]

        n_rows, n_cols = in_df.chunk_shape

        # map to get individual results and summaries
        src_chunks = np.empty(in_df.chunk_shape, dtype=np.object)
        summary_chunks = np.empty(in_df.chunk_shape, dtype=np.object)
        for c in in_df.chunks:
            new_chunk_op = op.copy().reset_key()
            new_chunk_op._stage = OperandStage.map
            if op.axis == 1:
                summary_shape = (c.shape[0], 1)
            else:
                summary_shape = (1, c.shape[1])
            src_chunks[c.index] = c
            summary_chunks[c.index] = new_chunk_op.new_chunk([c], shape=summary_shape, dtypes=df.dtypes)

        # combine summaries into results
        output_chunk_array = np.empty(in_df.chunk_shape, dtype=np.object)
        if op.axis == 1:
            for row in range(n_rows):
                row_src = src_chunks[row, :]
                row_summaries = summary_chunks[row, :]
                for col in range(n_cols):
                    output_chunk_array[row, col] = cls._build_combine(op, row_src, row_summaries, col)
        else:
            for col in range(n_cols):
                col_src = src_chunks[:, col]
                col_summaries = summary_chunks[:, col]
                for row in range(n_rows):
                    output_chunk_array[row, col] = cls._build_combine(op, col_src, col_summaries, row)

        output_chunks = list(output_chunk_array.reshape((n_rows * n_cols,)))
        new_op = op.copy().reset_key()
        return new_op.new_tileables(op.inputs, shape=in_df.shape, nsplits=in_df.nsplits,
                                    chunks=output_chunks, dtypes=df.dtypes,
                                    index_value=df.index_value, columns_value=df.columns_value)

    @classmethod
    def _tile_series(cls, op):
        in_series = op.inputs[0]
        series = op.outputs[0]

        # map to get individual results and summaries
        summary_chunks = np.empty(in_series.chunk_shape, dtype=np.object)
        for c in in_series.chunks:
            new_chunk_op = op.copy().reset_key()
            new_chunk_op._stage = OperandStage.map
            summary_chunks[c.index] = new_chunk_op.new_chunk([c], shape=(1,), dtype=series.dtype)

        # combine summaries into results
        output_chunks = [
            cls._build_combine(op, in_series.chunks, summary_chunks, i) for i in range(len(in_series.chunks))
        ]
        new_op = op.copy().reset_key()
        return new_op.new_tileables(op.inputs, shape=in_series.shape, nsplits=in_series.nsplits,
                                    chunks=output_chunks, dtype=series.dtype,
                                    index_value=series.index_value, name=series.name)

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        if len(in_df.chunks) == 1:
            return cls._tile_one_chunk(op)
        if isinstance(in_df, DATAFRAME_TYPE):
            return cls._tile_dataframe(op)
        else:
            return cls._tile_series(op)

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
        in_data = ctx[op.inputs[0].key]
        kwargs = dict()
        if op.axis is not None:
            kwargs['axis'] = op.axis
        if op.skipna is not None:
            kwargs['skipna'] = op.skipna
        partial = getattr(in_data, getattr(cls, '_func_name'))(**kwargs)
        if op.skipna:
            partial.fillna(method='ffill', axis=op.axis, inplace=True)
        ctx[op.outputs[0].key] = cls._get_last_slice(op, partial, -1)

    @classmethod
    def _execute_combine(cls, ctx, op):
        kwargs = dict()
        if op.axis is not None:
            kwargs['axis'] = op.axis
        if op.skipna is not None:
            kwargs['skipna'] = op.skipna

        if len(op.inputs) > 1:
            ref_datas = [ctx[inp.key] for inp in op.inputs[1:]]
            concat_df = getattr(pd.concat(ref_datas, axis=op.axis), getattr(cls, '_func_name'))(**kwargs)
            if op.skipna:
                concat_df.fillna(method='ffill', axis=op.axis, inplace=True)

            in_data = ctx[op.inputs[0].key]
            concat_df = pd.concat([cls._get_last_slice(op, concat_df, -1), in_data], axis=op.axis)
            result = getattr(concat_df, getattr(cls, '_func_name'))(**kwargs)
            ctx[op.outputs[0].key] = cls._get_last_slice(op, result, 1)
        else:
            ctx[op.outputs[0].key] = getattr(ctx[op.inputs[0].key], getattr(cls, '_func_name'))(**kwargs)

    @classmethod
    def execute(cls, ctx, op):
        try:
            pd.set_option('mode.use_inf_as_na', op.use_inf_as_na)
            if op.stage == OperandStage.map:
                return cls._execute_map(ctx, op)
            else:
                return cls._execute_combine(ctx, op)
        finally:
            pd.reset_option('mode.use_inf_as_na')

    def _call_dataframe(self, df):
        axis = getattr(self, 'axis', None) or 0
        self._axis = axis = validate_axis(axis, df)

        empty_df = build_empty_df(df.dtypes)
        reduced_df = getattr(empty_df, getattr(self, '_func_name'))(axis=axis)
        return self.new_dataframe([df], shape=df.shape, dtypes=reduced_df.dtypes,
                                  index_value=df.index_value, columns_value=df.columns_value)

    def _call_series(self, series):
        axis = getattr(self, 'axis', None) or 0
        if axis == 'index':
            axis = 0
        self._axis = axis

        return self.new_series([series], shape=series.shape, dtype=series.dtype,
                               name=series.name, index_value=series.index_value)

    def __call__(self, a):
        if isinstance(a, DATAFRAME_TYPE):
            return self._call_dataframe(a)
        else:
            return self._call_series(a)
