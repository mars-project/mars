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

import functools
from collections import OrderedDict
from typing import NamedTuple, Any, List, Dict, Union, Callable

import numpy as np
import pandas as pd

from ...core import OutputType, Entity, Base
from ...operands import OperandStage
from ...utils import tokenize, is_build_mode, enter_mode, recursive_tile
from ...serialize import BoolField, AnyField, DataTypeField, Int32Field
from ..core import SERIES_TYPE
from ..utils import parse_index, build_df, build_empty_df, build_series, \
    build_empty_series, validate_axis
from ..operands import DataFrameOperandMixin, DataFrameOperand, DATAFRAME_TYPE


class DataFrameReductionOperand(DataFrameOperand):
    _axis = AnyField('axis')
    _skipna = BoolField('skipna')
    _level = AnyField('level')
    _numeric_only = BoolField('numeric_only')
    _bool_only = BoolField('bool_only')
    _min_count = Int32Field('min_count')
    _use_inf_as_na = BoolField('use_inf_as_na')

    _dtype = DataTypeField('dtype')
    _combine_size = Int32Field('combine_size')

    def __init__(self, axis=None, skipna=None, level=None, numeric_only=None, bool_only=None,
                 min_count=None, stage=None, dtype=None, combine_size=None, gpu=None,
                 sparse=None, output_types=None, use_inf_as_na=None, **kw):
        super().__init__(_axis=axis, _skipna=skipna, _level=level, _numeric_only=numeric_only,
                         _bool_only=bool_only, _min_count=min_count, _stage=stage, _dtype=dtype,
                         _combine_size=combine_size, _gpu=gpu, _sparse=sparse,
                         _output_types=output_types, _use_inf_as_na=use_inf_as_na, **kw)

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
    def bool_only(self):
        return self._bool_only

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
                 output_types=None, use_inf_as_na=None, stage=None, **kw):
        super().__init__(_axis=axis, _skipna=skipna, _dtype=dtype, _gpu=gpu, _sparse=sparse,
                         _output_types=output_types, _stage=stage, _use_inf_as_na=use_inf_as_na, **kw)

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


def _default_agg_fun(value, func_name=None, **kw):
    if value.ndim == 1:
        kw.pop('bool_only', None)
        kw.pop('numeric_only', None)
        return getattr(value, func_name)(**kw)
    else:
        return getattr(value, func_name)(**kw)


class DataFrameReductionMixin(DataFrameOperandMixin):
    @classmethod
    def _make_agg_object(cls, op):
        func_name = getattr(op, '_func_name')
        kw = dict(skipna=op.skipna, numeric_only=op.numeric_only,
                  bool_only=op.bool_only)
        kw = {k: v for k, v in kw.items() if v is not None}
        fun = functools.partial(_default_agg_fun, func_name=func_name, **kw)
        fun.__name__ = func_name
        return fun

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        if isinstance(out_df, SERIES_TYPE):
            output_type = OutputType.series
            dtypes = pd.Series([out_df.dtype], index=[out_df.name])
            index = out_df.index_value.to_pandas()
        elif out_df.ndim == 1:
            output_type = OutputType.tensor
            dtypes, index = out_df.dtype, None
        else:
            output_type = OutputType.scalar
            dtypes, index = out_df.dtype, None

        out_df = recursive_tile(in_df.agg(
            cls._make_agg_object(op), axis=op.axis or 0, numeric_only=op.numeric_only,
            bool_only=op.bool_only, combine_size=op.combine_size, _output_type=output_type,
            _dtypes=dtypes, _index=index
        ))
        return [out_df]

    def _call_dataframe(self, df):
        axis = getattr(self, 'axis', None) or 0
        level = getattr(self, 'level', None)
        skipna = getattr(self, 'skipna', None)
        numeric_only = getattr(self, 'numeric_only', None)
        bool_only = getattr(self, 'bool_only', None)
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
        elif func_name in ('all', 'any'):
            reduced_df = getattr(empty_df, func_name)(axis=axis, level=level, bool_only=bool_only)
        elif func_name == 'size':
            reduced_df = pd.Series(np.zeros(df.shape[1 - axis]), index=empty_df.columns if axis == 0 else None)
        elif func_name == 'custom_reduction':
            reduced_df = getattr(self, 'custom_reduction').call_agg(empty_df)
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
        bool_only = getattr(self, 'bool_only', None)
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
        elif func_name in ('all', 'any'):
            reduced_series = getattr(empty_series, func_name)(axis=axis, level=level, bool_only=bool_only)
        elif func_name == 'size':
            reduced_series = empty_series.size
        elif func_name == 'custom_reduction':
            reduced_series = getattr(self, 'custom_reduction').call_agg(empty_series)
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
        if op.output_types[0] == OutputType.series:
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


class CustomReduction:
    pre: Callable
    agg: Union[Callable, None]
    post: Union[Callable, None]
    name: Union[str, None]
    output_limit: Union[int, None]
    kwds: Dict

    def __init__(self, pre, agg=None, post=None, name=None, output_limit=None,
                 kwds=None, **kwargs):
        self.pre = pre
        self.agg = agg
        self.post = post
        self.name = name or '<custom>'
        self.output_limit = output_limit
        kwargs.update(kwds or dict())
        self.kwds = kwargs

    def to_tuple(self):
        return self.pre, self.agg, self.post, self.name, self.output_limit, self.kwds

    @classmethod
    def from_tuple(cls, tp):
        return cls(*tp)

    @property
    def __name__(self):
        return self.name

    def __call__(self, value):
        if is_build_mode():
            return value._custom_reduction(self)
        return self.call_agg(value)

    def call_agg(self, value):
        r = self.pre(value, **self.kwds)
        if not isinstance(r, tuple):
            r = (r,)
        self.output_limit = len(r)
        if self.post is not None:
            r = self.post(*r, **self.kwds)
        if isinstance(r, tuple):
            if len(r) == 1:
                r = r[0]
            else:
                raise ValueError('Need a post function to handle tuple output')
        return r

    def __mars_tokenize__(self):
        return [self.pre, self.agg, self.post, self.name, self.kwds]


class ReductionPreStep(NamedTuple):
    input_key: str
    output_key: str
    columns: Union[List[str], None]
    func: Callable


class ReductionAggStep(NamedTuple):
    input_key: str
    map_func_name: Union[str, None]
    agg_func_name: Union[str, None]
    custom_reduction: Union[CustomReduction, None]
    output_key: str
    output_limit: int
    kwds: Dict[str, Any]


class ReductionPostStep(NamedTuple):
    input_keys: List[str]
    output_key: str
    func_name: str
    columns: Union[List[str], None]
    func: Callable


class ReductionSteps(NamedTuple):
    pre_funcs: List[ReductionPreStep]
    agg_funcs: List[ReductionAggStep]
    post_funcs: List[ReductionPostStep]


@functools.lru_cache(100)
def _compile_expr_function(py_src):
    from ... import tensor, dataframe
    result_store = dict()
    global_vars = globals()
    global_vars.update(dict(mt=tensor, md=dataframe, array=np.array, nan=np.nan))
    exec(py_src, globals(), result_store)  # noqa: W0122  # nosec  # pylint: disable=exec-used
    fun = result_store['expr_function']
    return fun


_func_name_converts = dict(
    greater='gt',
    greater_equal='ge',
    less='lt',
    less_equal='le',
    equal='eq',
    not_equal='ne',
    true_divide='truediv',
    floor_divide='floordiv',
    power='pow',
)
_func_compile_cache = dict()  # type: Dict[str, ReductionSteps]


class ReductionCompiler:
    def __init__(self, axis=0):
        self._axis = axis
        self._key_to_tileable = dict()
        self._output_tileables = []
        self._lambda_counter = 0
        self._custom_counter = 0
        self._func_cache = dict()

        self._compiled_funcs = []
        self._output_key_to_pre_steps = dict()
        self._output_key_to_pre_cols = dict()
        self._output_key_to_agg_steps = dict()
        self._output_key_to_post_steps = dict()
        self._output_key_to_post_cols = dict()

    @classmethod
    def _check_function_valid(cls, func):
        if isinstance(func, functools.partial):
            return cls._check_function_valid(func.func)
        elif isinstance(func, CustomReduction):
            return

        func_code = func.__code__
        for var_name in func_code.co_names:
            if isinstance(func.__globals__.get(var_name), (Base, Entity)):
                raise ValueError(f'Global variable {var_name} used by {func.__name__} '
                                 'cannot be a Mars object')
        for cell in func.__closure__ or ():
            if isinstance(cell.cell_contents, (Base, Entity)):
                raise ValueError(f'Cannot reference Mars objects inside {func.__name__}')

    def add_function(self, func, ndim, cols=None, func_name=None):
        cols = cols if cols is not None and self._axis == 0 else None

        func_name = func_name or getattr(func, '__name__', None)
        if func_name == '<lambda>' or func_name is None:
            func_name = f'<lambda_{self._lambda_counter}>'
            self._lambda_counter += 1
        if func_name == '<custom>' or func_name is None:
            func_name = f'<custom_{self._custom_counter}>'
            self._custom_counter += 1

        compile_result = self._compile_function(func, func_name, ndim=ndim)
        self._compiled_funcs.append(compile_result)

        for step in compile_result.pre_funcs:
            self._output_key_to_pre_steps[step.output_key] = step
            if step.output_key in self._output_key_to_pre_cols:
                existing_cols = self._output_key_to_pre_cols[step.output_key]
                if existing_cols is not None:
                    existing_col_set = set(existing_cols)
                    self._output_key_to_pre_cols[step.output_key].extend(
                        [c for c in cols if c not in existing_col_set])
            else:
                self._output_key_to_pre_cols[step.output_key] = list(cols) if cols is not None else None

        for step in compile_result.agg_funcs:
            self._output_key_to_agg_steps[step.output_key] = step

        for step in compile_result.post_funcs:
            self._output_key_to_post_steps[step.output_key] = step
            self._output_key_to_post_cols[step.output_key] = cols

    @enter_mode(build=True)
    def _compile_function(self, func, func_name=None, ndim=1) -> ReductionSteps:
        from . import DataFrameAll, DataFrameAny, DataFrameSum, DataFrameProd, \
            DataFrameCount, DataFrameMin, DataFrameMax, DataFrameSize, \
            DataFrameCustomReduction
        from ...tensor.arithmetic.core import TensorBinOp, TensorUnaryOp
        from ...tensor.base import TensorWhere
        from ..arithmetic.core import DataFrameBinOp, DataFrameUnaryOp
        from ..datasource.dataframe import DataFrameDataSource
        from ..indexing.where import DataFrameWhere
        from ..datasource.series import SeriesDataSource
        from ..initializer import DataFrame as MarsDataFrame, Series as MarsSeries

        func_token = tokenize(func, self._axis, func_name, ndim)
        if func_token in _func_compile_cache:
            return _func_compile_cache[func_token]
        custom_reduction = func if isinstance(func, CustomReduction) else None
        output_limit = getattr(func, 'output_limit', None) or 1

        atomic_agg_op_types = (DataFrameAll, DataFrameAny, DataFrameSum, DataFrameProd,
                               DataFrameCount, DataFrameMin, DataFrameMax, DataFrameSize,
                               DataFrameCustomReduction)

        if ndim == 1:
            mock_series = build_empty_series(np.dtype(float))
            mock_obj = MarsSeries(mock_series)
        else:
            mock_df = build_empty_df(pd.Series([np.dtype(float)] * 2, index=['A', 'B']))
            mock_obj = MarsDataFrame(mock_df)

        self._check_function_valid(func)
        func_ret = func(mock_obj)
        if not isinstance(func_ret, (Base, Entity)):
            raise ValueError(f'Custom function should return a Mars object, not {type(func_ret)}')
        if func_ret.ndim >= mock_obj.ndim:
            raise ValueError('Function not a reduction')

        agg_graph = func_ret.build_graph()
        agg_tileables = set(t for t in agg_graph if isinstance(t.op, atomic_agg_op_types))
        for t in agg_graph.dfs(list(agg_tileables), visit_predicate='all', reverse=True):
            if t not in agg_tileables and \
                    not isinstance(t.op, (DataFrameUnaryOp, DataFrameBinOp,
                                          TensorUnaryOp, TensorBinOp,
                                          TensorWhere, DataFrameWhere,
                                          DataFrameDataSource, SeriesDataSource)):
                raise ValueError(f'Cannot support operand {type(t.op)} in custom aggregation')
        for t in agg_graph.dfs(list(agg_tileables), visit_predicate='all'):
            if t not in agg_tileables and \
                    not isinstance(t.op, (DataFrameUnaryOp, DataFrameBinOp,
                                          TensorWhere, DataFrameWhere,
                                          TensorUnaryOp, TensorBinOp)):
                raise ValueError(f'Cannot support operand {type(t.op)} in custom aggregation')

        pre_funcs, agg_funcs, post_funcs = [], [], []
        visited_inputs = set()
        for t in agg_tileables:
            agg_input_key = t.inputs[0].key

            step_func_name = getattr(t.op, '_func_name')
            if step_func_name in ('count', 'size'):
                map_func_name, agg_func_name = step_func_name, 'sum'
            else:
                map_func_name, agg_func_name = step_func_name, step_func_name

            func_args = dict(skipna=t.op.skipna)
            if t.inputs[0].ndim > 1:
                func_args['axis'] = self._axis
            if t.op.numeric_only is not None:
                func_args['numeric_only'] = t.op.numeric_only
            if t.op.bool_only is not None:
                func_args['bool_only'] = t.op.bool_only

            agg_funcs.append(ReductionAggStep(
                agg_input_key, map_func_name, agg_func_name, custom_reduction, t.key, output_limit,
                {k: v for k, v in func_args.items() if v is not None}
            ))
            if agg_input_key not in visited_inputs:
                visited_inputs.add(agg_input_key)
                initial_inputs = list(t.inputs[0].build_graph().iter_indep())
                assert len(initial_inputs) == 1
                input_key = initial_inputs[0].key

                func_str, _ = self._generate_function_str(t.inputs[0])
                pre_funcs.append(ReductionPreStep(
                    input_key, agg_input_key, None, _compile_expr_function(func_str)
                ))
        func_str, input_keys = self._generate_function_str(func_ret)
        post_funcs.append(ReductionPostStep(
            input_keys, func_ret.key, func_name, None, _compile_expr_function(func_str)
        ))
        if len(_func_compile_cache) > 100:
            _func_compile_cache.pop(next(iter(_func_compile_cache.keys())))
        result = _func_compile_cache[func_token] = ReductionSteps(pre_funcs, agg_funcs, post_funcs)
        return result

    def _generate_function_str(self, out_tileable):
        from ...tensor.arithmetic.core import TensorBinOp, TensorUnaryOp
        from ...tensor.base import TensorWhere
        from ...tensor.datasource import Scalar
        from ..arithmetic.core import DataFrameBinOp, DataFrameUnaryOp, DataFrameUnaryUfunc
        from ..datasource.dataframe import DataFrameDataSource
        from ..datasource.series import SeriesDataSource
        from ..indexing.where import DataFrameWhere

        input_key_to_arg = OrderedDict()
        local_key_to_arg = dict()
        local_lines = []

        def _gen_expr_str(t):
            if t.key in local_key_to_arg:
                return

            if isinstance(t.op, (DataFrameDataSource, SeriesDataSource, DataFrameReductionOperand)):
                if t.key not in input_key_to_arg:
                    input_key_to_arg[t.key] = local_key_to_arg[t.key] = f'invar{len(input_key_to_arg)}'
            else:
                for inp in t.inputs:
                    _gen_expr_str(inp)

                var_name = local_key_to_arg[t.key] = f'var{len(local_key_to_arg)}'
                keys_to_vars = {inp.key: local_key_to_arg[inp.key] for inp in t.inputs}

                def _interpret_var(v):
                    if hasattr(v, 'key'):
                        return keys_to_vars[v.key]
                    return v

                func_name = func_name_raw = getattr(t.op, '_func_name', None)
                rfunc_name = getattr(t.op, '_rfunc_name', func_name)

                if func_name in _func_name_converts:
                    func_name = _func_name_converts[func_name]
                if rfunc_name in _func_name_converts:
                    rfunc_name = 'r' + _func_name_converts[rfunc_name]

                if isinstance(t.op, (DataFrameUnaryOp, TensorUnaryOp)):
                    val = _interpret_var(t.inputs[0])
                    if isinstance(t.op, DataFrameUnaryUfunc):
                        statements = [f'{var_name} = np.{func_name_raw}({val})']
                    else:
                        statements = [f'try:',
                                      f'    {var_name} = {val}.{func_name}()',
                                      f'except AttributeError:',
                                      f'    {var_name} = np.{func_name_raw}({val})']
                elif isinstance(t.op, (DataFrameBinOp, TensorBinOp)):
                    lhs, rhs = t.op.lhs, t.op.rhs
                    op_axis = 1 - self._axis if hasattr(lhs, 'ndim') and hasattr(rhs, 'ndim') \
                        and lhs.ndim != rhs.ndim else None
                    lhs = _interpret_var(lhs)
                    rhs = _interpret_var(rhs)
                    axis_expr = f'axis={op_axis!r}, ' if op_axis is not None else ''
                    if t.op.lhs is t.inputs[0]:
                        statements = [f'try:',
                                      f'    {var_name} = {lhs}.{func_name}({rhs}, {axis_expr})',
                                      f'except AttributeError:',
                                      f'    {var_name} = np.{func_name_raw}({lhs}, {rhs})']
                    else:
                        statements = [f'try:',
                                      f'    {var_name} = {rhs}.{rfunc_name}({lhs}, {axis_expr})',
                                      f'except AttributeError:',
                                      f'    {var_name} = np.{func_name_raw}({lhs}, {rhs})']
                elif isinstance(t.op, TensorWhere):
                    inp = _interpret_var(t.op.condition)
                    x = _interpret_var(t.op.x)
                    y = _interpret_var(t.op.y)
                    statements = [f'{var_name} = np.where({inp}, {x}, {y})']
                elif isinstance(t.op, DataFrameWhere):
                    func_name = 'mask' if t.op.replace_true else 'where'
                    inp = _interpret_var(t.op.input)
                    cond = _interpret_var(t.op.cond)
                    other = _interpret_var(t.op.other)
                    op_axis = t.op.axis
                    op_level = t.op.level
                    statements = [f'{var_name} = {inp}.{func_name}({cond}, {other}, '
                                  f'axis={op_axis!r}, level={op_level!r})']
                elif isinstance(t.op, Scalar):
                    data = _interpret_var(t.op.data)
                    statements = [f'{var_name} = {data}']
                else:
                    raise NotImplementedError(f'Does not support aggregating on {type(t.op)}')
                local_lines.extend(statements)

        _gen_expr_str(out_tileable)

        args_str = ', '.join(input_key_to_arg.values())
        lines_str = '\n    '.join(local_lines)
        return f"def expr_function({args_str}):\n" \
               f"    {lines_str}\n" \
               f"    return {local_key_to_arg[out_tileable.key]}", \
            list(input_key_to_arg.keys())

    def compile(self) -> ReductionSteps:
        pre_funcs, agg_funcs, post_funcs = [], [], []
        for key, step in self._output_key_to_pre_steps.items():
            cols = self._output_key_to_pre_cols[key]
            pre_funcs.append(ReductionPreStep(
                step.input_key, step.output_key, cols, step.func))

        for step in self._output_key_to_agg_steps.values():
            agg_funcs.append(step)

        for key, step in self._output_key_to_post_steps.items():
            cols = self._output_key_to_post_cols[key]
            post_funcs.append(ReductionPostStep(
                step.input_keys, step.output_key, step.func_name, cols, step.func))

        return ReductionSteps(pre_funcs, agg_funcs, post_funcs)
