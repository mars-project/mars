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

import functools
from collections import OrderedDict
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd

from ...core import OutputType, ENTITY_TYPE, is_build_mode, \
    is_kernel_mode, enter_mode, recursive_tile
from ...core.operand import OperandStage
from ...utils import tokenize
from ...serialization.serializables import BoolField, AnyField, \
    DataTypeField, Int32Field, StringField
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
    _method = StringField('method')

    _dtype = DataTypeField('dtype')
    _combine_size = Int32Field('combine_size')

    def __init__(self, axis=None, skipna=None, level=None, numeric_only=None, bool_only=None,
                 min_count=None, dtype=None, combine_size=None, gpu=None,
                 sparse=None, output_types=None, use_inf_as_na=None, method=None, **kw):
        super().__init__(_axis=axis, _skipna=skipna, _level=level, _numeric_only=numeric_only,
                         _bool_only=bool_only, _min_count=min_count, _dtype=dtype,
                         _combine_size=combine_size, _gpu=gpu, _sparse=sparse,
                         _output_types=output_types, _use_inf_as_na=use_inf_as_na,
                         _method=method, **kw)

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

    @property
    def is_atomic(self):
        return False

    @property
    def method(self):
        return self._method

    def get_reduction_args(self, axis=None):
        args = dict(skipna=self.skipna)
        if self.inputs[0].ndim > 1:
            args['axis'] = axis
        if self.numeric_only is not None:
            args['numeric_only'] = self.numeric_only
        if self.bool_only is not None:
            args['bool_only'] = self.bool_only
        return {k: v for k, v in args.items() if v is not None}


class DataFrameCumReductionOperand(DataFrameOperand):
    _axis = AnyField('axis')
    _skipna = BoolField('skipna')
    _use_inf_as_na = BoolField('use_inf_as_na')

    _dtype = DataTypeField('dtype')

    def __init__(self, axis=None, skipna=None, dtype=None, gpu=None, sparse=None,
                 output_types=None, use_inf_as_na=None, **kw):
        super().__init__(_axis=axis, _skipna=skipna, _dtype=dtype, _gpu=gpu, _sparse=sparse,
                         _output_types=output_types, _use_inf_as_na=use_inf_as_na, **kw)

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
    def get_reduction_callable(cls, op):
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

        out_df = yield from recursive_tile(in_df.agg(
            cls.get_reduction_callable(op), axis=op.axis or 0, _numeric_only=op.numeric_only,
            _bool_only=op.bool_only, _combine_size=op.combine_size, _output_type=output_type,
            _dtypes=dtypes, _index=index
        ))
        return [out_df]

    def _call_groupby_level(self, df, level):
        return df.groupby(level=level).agg(self.get_reduction_callable(self),
                                           method=self.method)

    def _call_dataframe(self, df):
        axis = getattr(self, 'axis', None) or 0
        level = getattr(self, 'level', None)
        skipna = getattr(self, 'skipna', None)
        numeric_only = getattr(self, 'numeric_only', None)
        bool_only = getattr(self, 'bool_only', None)
        self._axis = axis = validate_axis(axis, df)
        func_name = getattr(self, '_func_name')

        if level is not None and axis == 1:
            raise NotImplementedError('Not support specify level for axis==1')

        empty_df = build_df(df, ensure_string=True)
        if func_name == 'count':
            reduced_df = getattr(empty_df, func_name)(axis=axis, level=level, numeric_only=numeric_only)
        elif func_name == 'nunique':
            reduced_df = getattr(empty_df, func_name)(axis=axis)
        elif func_name in ('all', 'any'):
            reduced_df = getattr(empty_df, func_name)(axis=axis, level=level, bool_only=bool_only)
        elif func_name == 'size':
            reduced_df = pd.Series(np.zeros(df.shape[1 - axis]), index=empty_df.columns if axis == 0 else None)
        elif func_name == 'custom_reduction':
            reduced_df = getattr(self, 'custom_reduction').__call_agg__(empty_df)
        elif func_name == 'str_concat':
            reduced_df = empty_df.apply(lambda s: s.str.cat(**getattr(self, 'get_reduction_args')()), axis=axis)
        else:
            reduced_df = getattr(empty_df, func_name)(axis=axis, level=level, skipna=skipna,
                                                      numeric_only=numeric_only)

        if level is not None:
            return self._call_groupby_level(df[list(reduced_df.columns)], level)

        reduced_shape = (df.shape[0],) if axis == 1 else reduced_df.shape
        index_value = parse_index(reduced_df.index, store_data=True) \
            if axis == 0 else parse_index(pd.RangeIndex(-1))
        return self.new_series([df], shape=reduced_shape, dtype=reduced_df.dtype,
                               index_value=index_value)

    def _call_series(self, series):
        level = getattr(self, 'level', None)
        axis = getattr(self, 'axis', None)
        skipna = getattr(self, 'skipna', None)
        numeric_only = getattr(self, 'numeric_only', None)
        bool_only = getattr(self, 'bool_only', None)
        self._axis = axis = validate_axis(axis or 0, series)
        func_name = getattr(self, '_func_name')

        if level is not None:
            return self._call_groupby_level(series, level)

        empty_series = build_series(series, ensure_string=True)
        if func_name == 'count':
            reduced_series = empty_series.count(level=level)
        elif func_name == 'nunique':
            reduced_series = empty_series.nunique()
        elif func_name in ('all', 'any'):
            reduced_series = getattr(empty_series, func_name)(axis=axis, level=level, bool_only=bool_only)
        elif func_name == 'size':
            reduced_series = empty_series.size
        elif func_name == 'custom_reduction':
            reduced_series = getattr(self, 'custom_reduction').__call_agg__(empty_series)
        elif func_name == 'str_concat':
            reduced_series = pd.Series([empty_series.str.cat(**getattr(self, 'get_reduction_args')())])
        else:
            reduced_series = getattr(empty_series, func_name)(axis=axis, level=level, skipna=skipna,
                                                              numeric_only=numeric_only)

        return self.new_scalar([series], dtype=np.array(reduced_series).dtype)

    def __call__(self, a):
        if is_kernel_mode() and not getattr(self, 'is_atomic', False):
            return self.get_reduction_callable(self)(a)

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
        new_chunk_op.stage = OperandStage.combine
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
            new_chunk_op.stage = OperandStage.map
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
            new_chunk_op.stage = OperandStage.map
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
    name: Optional[str]
    output_limit: Optional[int]
    kwds: Dict

    # set to True when pre() already performs aggregation
    pre_with_agg = False

    def __init__(self, name=None, is_gpu=False):
        self.name = name or '<custom>'
        self.output_limit = 1
        self._is_gpu = is_gpu

    @property
    def __name__(self):
        return self.name

    def __call__(self, value):
        if is_build_mode():
            from .custom_reduction import build_custom_reduction_result
            return build_custom_reduction_result(value, self)
        return self.__call_agg__(value)

    def __call_agg__(self, value):
        r = self.pre(value)
        if not isinstance(r, tuple):
            r = (r,)
        # update output limit into actual size
        self.output_limit = len(r)

        # only perform aggregation when pre() does not perform aggregation
        if not self.pre_with_agg:
            r = self.agg(*r)
            if not isinstance(r, tuple):
                r = (r,)

        r = self.post(*r)
        return r

    def is_gpu(self):
        return self._is_gpu if not is_build_mode() else False

    def pre(self, value):  # noqa: R0201  # pylint: disable=no-self-use
        return value,

    def agg(self, *values):  # noqa: R0201  # pylint: disable=no-self-use
        raise NotImplementedError

    def post(self, *value):  # noqa: R0201  # pylint: disable=no-self-use
        assert len(value) == 1
        return value[0]

    def __mars_tokenize__(self):
        import cloudpickle
        return cloudpickle.dumps(self)


class ReductionPreStep(NamedTuple):
    input_key: str
    output_key: str
    columns: Optional[List[str]]
    func: Callable


class ReductionAggStep(NamedTuple):
    input_key: str
    map_func_name: Optional[str]
    agg_func_name: Optional[str]
    custom_reduction: Optional[CustomReduction]
    output_key: str
    output_limit: int
    kwds: Dict[str, Any]


class ReductionPostStep(NamedTuple):
    input_keys: List[str]
    output_key: str
    func_name: str
    columns: Optional[List[str]]
    func: Callable


class ReductionSteps(NamedTuple):
    pre_funcs: List[ReductionPreStep]
    agg_funcs: List[ReductionAggStep]
    post_funcs: List[ReductionPostStep]


# lookup table for numpy arithmetic operands in pandas
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
_func_name_to_op = dict(
    greater='>', gt='>',
    greater_equal='>=', ge='>',
    less='<', lt='<',
    less_equal='<=', le='<=',
    equal='==', eq='==',
    not_equal='!=', ne='!=',
    bitwise_and='&', __and__='&',
    bitwise_or='|', __or__='|',
    bitwise_xor='^', __xor__='^',
    add='+',
    subtract='-', sub='-',
    multiply='*', mul='*',
    true_divide='/', truediv='/',
    floor_divide='//', floordiv='//',
    power='**', pow='**',
    mod='%',
)
_func_compile_cache = dict()  # type: Dict[str, ReductionSteps]


class ReductionCompiler:
    def __init__(self, axis=0, store_source=False):
        self._axis = axis
        self._store_source = store_source

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
        func_vars = {n: func.__globals__.get(n) for n in func_code.co_names}
        if func.__closure__:
            func_vars.update({n: cell.cell_contents for
                              n, cell in zip(func_code.co_freevars, func.__closure__)})
        # external Mars objects shall not be referenced
        for var_name, val in func_vars.items():
            if isinstance(val, ENTITY_TYPE):
                raise ValueError(f'Variable {var_name} used by {func.__name__} '
                                 'cannot be a Mars object')

    @staticmethod
    def _update_col_dict(col_dict: Dict, key: str, cols: List):
        if key in col_dict:
            existing_cols = col_dict[key]
            if existing_cols is not None:
                existing_col_set = set(existing_cols)
                col_dict[key].extend(
                    [c for c in cols if c not in existing_col_set])
        else:
            col_dict[key] = list(cols) if cols is not None else None

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
            self._update_col_dict(self._output_key_to_pre_cols, step.output_key, cols)

        for step in compile_result.agg_funcs:
            self._output_key_to_agg_steps[step.output_key] = step

        for step in compile_result.post_funcs:
            self._output_key_to_post_steps[step.output_key] = step
            self._update_col_dict(self._output_key_to_post_cols, step.output_key, cols)

    @functools.lru_cache(100)
    def _compile_expr_function(self, py_src):
        from ... import tensor, dataframe
        result_store = dict()
        global_vars = globals()
        global_vars.update(dict(mt=tensor, md=dataframe, array=np.array, nan=np.nan))
        exec(py_src, global_vars, result_store)  # noqa: W0122  # nosec  # pylint: disable=exec-used
        fun = result_store['expr_function']
        if self._store_source:
            fun.__source__ = py_src
        return fun

    @staticmethod
    def _build_mock_return_object(func, input_dtype, ndim):
        from ..initializer import DataFrame as MarsDataFrame, Series as MarsSeries

        if ndim == 1:
            mock_series = build_empty_series(np.dtype(input_dtype))
            mock_obj = MarsSeries(mock_series)
        else:
            mock_df = build_empty_df(pd.Series([np.dtype(input_dtype)] * 2, index=['A', 'B']))
            mock_obj = MarsDataFrame(mock_df)

        # calc target tileable to generate DAG
        with enter_mode(kernel=True):
            return func(mock_obj)

    @enter_mode(build=True)
    def _compile_function(self, func, func_name=None, ndim=1) -> ReductionSteps:
        from ...tensor.arithmetic.core import TensorBinOp, TensorUnaryOp
        from ...tensor.base import TensorWhere
        from ..arithmetic.core import DataFrameBinOp, DataFrameUnaryOp
        from ..datasource.dataframe import DataFrameDataSource
        from ..indexing.where import DataFrameWhere
        from ..datasource.series import SeriesDataSource

        func_token = tokenize(func, self._axis, func_name, ndim)
        if func_token in _func_compile_cache:
            return _func_compile_cache[func_token]
        custom_reduction = func if isinstance(func, CustomReduction) else None

        self._check_function_valid(func)

        try:
            func_ret = self._build_mock_return_object(func, float, ndim=ndim)
        except (TypeError, AttributeError):
            # we may encounter lambda x: x.str.cat(...), use an object series to test
            func_ret = self._build_mock_return_object(func, object, ndim=1)
        output_limit = getattr(func, 'output_limit', None) or 1

        if not isinstance(func_ret, ENTITY_TYPE):
            raise ValueError(f'Custom function should return a Mars object, not {type(func_ret)}')
        if func_ret.ndim >= ndim:
            raise ValueError('Function not a reduction')

        agg_graph = func_ret.build_graph()
        agg_tileables = set(t for t in agg_graph if getattr(t.op, 'is_atomic', False))
        # check operands before aggregation
        for t in agg_graph.dfs(list(agg_tileables), visit_predicate='all', reverse=True):
            if t not in agg_tileables and \
                    not isinstance(t.op, (DataFrameUnaryOp, DataFrameBinOp,
                                          TensorUnaryOp, TensorBinOp,
                                          TensorWhere, DataFrameWhere,
                                          DataFrameDataSource, SeriesDataSource)):
                raise ValueError(f'Cannot support operand {type(t.op)} in aggregation')
        # check operands after aggregation
        for t in agg_graph.dfs(list(agg_tileables), visit_predicate='all'):
            if t not in agg_tileables and \
                    not isinstance(t.op, (DataFrameUnaryOp, DataFrameBinOp,
                                          TensorWhere, DataFrameWhere,
                                          TensorUnaryOp, TensorBinOp)):
                raise ValueError(f'Cannot support operand {type(t.op)} in aggregation')

        pre_funcs, agg_funcs, post_funcs = [], [], []
        visited_inputs = set()
        # collect aggregations and their inputs
        for t in agg_tileables:
            agg_input_key = t.inputs[0].key

            # collect agg names
            step_func_name = getattr(t.op, '_func_name')
            if step_func_name in ('count', 'size'):
                map_func_name, agg_func_name = step_func_name, 'sum'
            else:
                map_func_name, agg_func_name = step_func_name, step_func_name

            # build agg description
            agg_funcs.append(ReductionAggStep(
                agg_input_key, map_func_name, agg_func_name, custom_reduction, t.key, output_limit,
                t.op.get_reduction_args(axis=self._axis)
            ))
            # collect agg input and build function
            if agg_input_key not in visited_inputs:
                visited_inputs.add(agg_input_key)
                initial_inputs = list(t.inputs[0].build_graph().iter_indep())
                assert len(initial_inputs) == 1
                input_key = initial_inputs[0].key

                func_str, _ = self._generate_function_str(t.inputs[0])
                pre_funcs.append(ReductionPreStep(
                    input_key, agg_input_key, None, self._compile_expr_function(func_str)
                ))
        # collect function output after agg
        func_str, input_keys = self._generate_function_str(func_ret)
        post_funcs.append(ReductionPostStep(
            input_keys, func_ret.key, func_name, None, self._compile_expr_function(func_str)
        ))
        if len(_func_compile_cache) > 100:  # pragma: no cover
            _func_compile_cache.pop(next(iter(_func_compile_cache.keys())))
        result = _func_compile_cache[func_token] = ReductionSteps(pre_funcs, agg_funcs, post_funcs)
        return result

    def _generate_function_str(self, out_tileable):
        """
        Generate python code from tileable DAG
        """
        from ...tensor.arithmetic.core import TensorBinOp, TensorUnaryOp
        from ...tensor.base import TensorWhere
        from ...tensor.datasource import Scalar
        from ..arithmetic.core import DataFrameBinOp, DataFrameUnaryOp, DataFrameUnaryUfunc
        from ..datasource.dataframe import DataFrameDataSource
        from ..datasource.series import SeriesDataSource
        from ..indexing.where import DataFrameWhere

        input_key_to_var = OrderedDict()
        local_key_to_var = dict()
        ref_counts = dict()
        ref_visited = set()
        local_lines = []

        input_op_types = (DataFrameDataSource, SeriesDataSource, DataFrameReductionOperand)

        def _calc_ref_counts(t):
            # calculate object refcount for t, this reduces memory usage in functions
            if t.key in ref_visited:
                return
            ref_visited.add(t.key)
            for inp in t.inputs:
                _calc_ref_counts(inp)

                if not isinstance(inp.op, input_op_types):
                    if inp.key not in ref_counts:
                        ref_counts[inp.key] = 0
                    ref_counts[inp.key] += 1

        def _gen_expr_str(t):
            # generate code for t
            if t.key in local_key_to_var:
                return

            if isinstance(t.op, input_op_types):
                # tileable is an input arg, build a function variable
                if t.key not in input_key_to_var:  # pragma: no branch
                    input_key_to_var[t.key] = local_key_to_var[t.key] = f'invar{len(input_key_to_var)}'
            else:
                keys_to_del = []
                for inp in t.inputs:
                    _gen_expr_str(inp)

                    if inp.key in ref_counts:
                        ref_counts[inp.key] -= 1
                        if ref_counts[inp.key] == 0:
                            # the input is no longer referenced, a del statement will be produced
                            keys_to_del.append(inp.key)

                var_name = local_key_to_var[t.key] = f'var{len(local_key_to_var)}'
                keys_to_vars = {inp.key: local_key_to_var[inp.key] for inp in t.inputs}

                def _interpret_var(v):
                    # get representation for variables
                    if hasattr(v, 'key'):
                        return keys_to_vars[v.key]
                    return v

                func_name = func_name_raw = getattr(t.op, '_func_name', None)
                rfunc_name = getattr(t.op, '_rfunc_name', func_name)

                if func_name is None:
                    func_name = func_name_raw = getattr(t.op, '_bit_func_name', None)
                    rfunc_name = getattr(t.op, '_bit_rfunc_name', func_name)

                # handle function name differences between numpy and pandas arithmetic ops
                if func_name in _func_name_converts:
                    func_name = _func_name_converts[func_name]
                if rfunc_name in _func_name_converts:
                    rfunc_name = 'r' + _func_name_converts[rfunc_name]

                # build given different op types
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
                    op_str = _func_name_to_op[func_name]
                    if t.op.lhs is t.inputs[0]:
                        statements = [f'try:',
                                      f'    {var_name} = {lhs}.{func_name}({rhs}, {axis_expr})',
                                      f'except AttributeError:',
                                      f'    {var_name} = {lhs} {op_str} {rhs}']
                    else:
                        statements = [f'try:',
                                      f'    {var_name} = {rhs}.{rfunc_name}({lhs}, {axis_expr})',
                                      f'except AttributeError:',
                                      f'    {var_name} = {rhs} {op_str} {lhs}']
                elif isinstance(t.op, TensorWhere):
                    cond = _interpret_var(t.op.condition)
                    x = _interpret_var(t.op.x)
                    y = _interpret_var(t.op.y)
                    statements = [f'if not gpu:',
                                  f'    {var_name} = np.where({cond}, {x}, {y})',
                                  f'else:',  # there is a bug with cudf.where
                                  f'    {var_name} = {x}']
                elif isinstance(t.op, DataFrameWhere):
                    func_name = 'mask' if t.op.replace_true else 'where'
                    inp = _interpret_var(t.op.input)
                    cond = _interpret_var(t.op.cond)
                    other = _interpret_var(t.op.other)
                    statements = [f'if not gpu:',
                                  f'    {var_name} = {inp}.{func_name}({cond}, {other}, '
                                  f'axis={t.op.axis!r}, level={t.op.level!r})',
                                  f'else:',  # there is a bug with cudf.where
                                  f'    {var_name} = {inp}']
                elif isinstance(t.op, Scalar):
                    # for scalar inputs of other operands
                    data = _interpret_var(t.op.data)
                    statements = [f'{var_name} = {data}']
                else:  # pragma: no cover
                    raise NotImplementedError(f'Does not support aggregating on {type(t.op)}')

                # append del statements for used inputs
                for key in keys_to_del:
                    statements.append(f'del {local_key_to_var[key]}')

                local_lines.extend(statements)

        _calc_ref_counts(out_tileable)
        _gen_expr_str(out_tileable)

        args_str = ', '.join(input_key_to_var.values())
        lines_str = '\n    '.join(local_lines)
        return f"def expr_function({args_str}, gpu=False):\n" \
               f"    {lines_str}\n" \
               f"    return {local_key_to_var[out_tileable.key]}", \
            list(input_key_to_var.keys())

    def compile(self) -> ReductionSteps:
        pre_funcs, agg_funcs, post_funcs = [], [], []
        referred_cols = set()
        for key, step in self._output_key_to_pre_steps.items():
            cols = self._output_key_to_pre_cols[key]
            if cols:
                referred_cols.update(cols)
            pre_funcs.append(ReductionPreStep(
                step.input_key, step.output_key, cols, step.func))

        for step in self._output_key_to_agg_steps.values():
            agg_funcs.append(step)

        for key, step in self._output_key_to_post_steps.items():
            cols = self._output_key_to_post_cols[key]
            if cols and set(cols) == set(referred_cols):
                post_cols = None
            else:
                post_cols = cols

            post_funcs.append(ReductionPostStep(
                step.input_keys, step.output_key, step.func_name, post_cols, step.func))

        return ReductionSteps(pre_funcs, agg_funcs, post_funcs)
