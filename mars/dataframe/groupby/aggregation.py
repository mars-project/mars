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
import itertools
from collections import OrderedDict
from collections.abc import Iterable
from typing import List

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...config import options
from ...context import get_context, RunningMode
from ...core import Base, Entity, OutputType
from ...custom_log import redirect_custom_log
from ...operands import OperandStage
from ...serialize import Int32Field, AnyField, BoolField, StringField, ListField, DictField
from ...utils import enter_current_session, lazy_import
from ..core import GROUPBY_TYPE
from ..merge import DataFrameConcat
from ..operands import DataFrameOperand, DataFrameOperandMixin, DataFrameShuffleProxy
from ..reduction.core import CustomReduction, ReductionCompiler, ReductionSteps, \
    ReductionPreStep, ReductionAggStep, ReductionPostStep
from ..reduction.aggregation import mean_function, var_function, sem_function, \
    skew_function, kurt_function, is_funcs_aggregate
from ..utils import parse_index, build_concatenated_rows_frame
from .core import DataFrameGroupByOperand

cp = lazy_import('cupy', globals=globals(), rename='cp')
cudf = lazy_import('cudf', globals=globals())


_agg_functions = {
    'sum': lambda x: x.sum(),
    'prod': lambda x: x.prod(),
    'product': lambda x: x.product(),
    'min': lambda x: x.min(),
    'max': lambda x: x.max(),
    'all': lambda x: x.all(),
    'any': lambda x: x.any(),
    'count': lambda x: x.count(),
    'size': lambda x: x._reduction_size(),
    'mean': mean_function,
    'var': var_function,
    'std': lambda x, ddof=1: var_function(x, ddof=ddof) ** 0.5,
    'sem': sem_function,
    'skew': skew_function,
    'kurt': kurt_function,
    'kurtosis': kurt_function,
}
_series_col_name = 'col_name'


def _patch_groupby_kurt():
    try:
        from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
        if not hasattr(DataFrameGroupBy, 'kurt'):
            DataFrameGroupBy.kurt = lambda x: x.agg(pd.Series.kurt)
            SeriesGroupBy.kurt = lambda x: x.agg(pd.Series.kurt)

            def kurtosis(x, *args, **kwargs):
                return pd.Series.kurt(x, *args, **kwargs)

            DataFrameGroupBy.kurtosis = lambda x: x.agg(kurtosis)
            SeriesGroupBy.kurtosis = lambda x: x.agg(kurtosis)
    except (AttributeError, ImportError):
        pass


_patch_groupby_kurt()
del _patch_groupby_kurt


class DataFrameGroupByAgg(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY_AGG

    _func = AnyField('func')
    _raw_func = AnyField('raw_func')

    _groupby_params = DictField('groupby_params')

    _method = StringField('method')
    _use_inf_as_na = BoolField('use_inf_as_na')

    # for chunk
    _combine_size = Int32Field('combine_size')
    _pre_funcs = ListField('pre_funcs')
    _agg_funcs = ListField('agg_funcs')
    _post_funcs = ListField('post_funcs')
    # for chunk
    _tileable_op_key = StringField('tileable_op_key')

    def __init__(self, func=None, method=None, groupby_params=None, raw_func=None,
                 use_inf_as_na=None, combine_size=None, pre_funcs=None, agg_funcs=None,
                 post_funcs=None, stage=None, output_types=None, tileable_op_key=None, **kw):
        super().__init__(_func=func, _method=method, _groupby_params=groupby_params,
                         _combine_size=combine_size, _use_inf_as_na=use_inf_as_na,
                         _pre_funcs=pre_funcs, _agg_funcs=agg_funcs, _post_funcs=post_funcs,
                         _raw_func=raw_func, _stage=stage, _output_types=output_types,
                         _tileable_op_key=tileable_op_key, **kw)

    @property
    def func(self):
        return self._func

    @property
    def raw_func(self):
        return self._raw_func

    @property
    def groupby_params(self) -> dict:
        return self._groupby_params

    @property
    def method(self):
        return self._method

    @property
    def use_inf_as_na(self):
        return self._use_inf_as_na

    @property
    def combine_size(self) -> int:
        return self._combine_size

    @property
    def pre_funcs(self) -> List[ReductionPreStep]:
        return self._pre_funcs

    @property
    def agg_funcs(self) -> List[ReductionAggStep]:
        return self._agg_funcs

    @property
    def post_funcs(self) -> List[ReductionPostStep]:
        return self._post_funcs

    @property
    def tileable_op_key(self):
        return self._tileable_op_key

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs[1:])
        if len(self._inputs) > 1:
            by = []
            for v in self._groupby_params['by']:
                if isinstance(v, (Base, Entity)):
                    by.append(next(inputs_iter))
                else:
                    by.append(v)
            self._groupby_params['by'] = by

    def _normalize_funcs(self):
        self._raw_func = self._func

        if isinstance(self._func, dict):
            new_func = OrderedDict()
            for k, v in self._func.items():
                if isinstance(v, str) or callable(v):
                    new_func[k] = [v]
                else:
                    new_func[k] = v
            self._func = new_func
        elif isinstance(self._func, Iterable) and not isinstance(self._func, str):
            self._func = list(self._func)
        else:
            self._func = [self._func]

        custom_idx = 0
        if isinstance(self._func, list):
            custom_iter = (f for f in self._func if isinstance(f, CustomReduction))
        else:
            custom_iter = (f for f in self._func.values() if isinstance(f, CustomReduction))
        for r in custom_iter:
            if r.name == '<custom>':
                r.name = f'<custom_{custom_idx}>'
                custom_idx += 1

    def _get_inputs(self, inputs):
        if isinstance(self._groupby_params['by'], list):
            for v in self._groupby_params['by']:
                if isinstance(v, (Base, Entity)):
                    inputs.append(v)
        return inputs

    def _call_dataframe(self, groupby, input_df):
        grouped = groupby.op.build_mock_groupby()
        agg_df = grouped.aggregate(self.raw_func)

        shape = (np.nan, agg_df.shape[1])
        index_value = parse_index(agg_df.index, groupby.key, groupby.index_value.key)
        index_value.value.should_be_monotonic = True

        as_index = self.groupby_params.get('as_index')
        # make sure if as_index=False takes effect
        if not as_index:
            if isinstance(agg_df.index, pd.MultiIndex):
                # if MultiIndex, as_index=False definitely takes no effect
                self.groupby_params['as_index'] = True
            elif agg_df.index.name is not None:
                # if not MultiIndex and agg_df.index has a name
                # means as_index=False takes no effect
                self.groupby_params['as_index'] = True

        inputs = self._get_inputs([input_df])
        return self.new_dataframe(inputs, shape=shape, dtypes=agg_df.dtypes,
                                  index_value=index_value,
                                  columns_value=parse_index(agg_df.columns, store_data=True))

    def _call_series(self, groupby, in_series):
        agg_result = groupby.op.build_mock_groupby().aggregate(self.raw_func)

        index_value = parse_index(agg_result.index, groupby.key, groupby.index_value.key)
        index_value.value.should_be_monotonic = True

        inputs = self._get_inputs([in_series])
        # update value type
        if isinstance(agg_result, pd.DataFrame):
            return self.new_dataframe(inputs, shape=(np.nan, len(agg_result.columns)),
                                      dtypes=agg_result.dtypes, index_value=index_value,
                                      columns_value=parse_index(agg_result.columns, store_data=True))
        else:
            return self.new_series(inputs, shape=(np.nan,), dtype=agg_result.dtype,
                                   name=agg_result.name, index_value=index_value)

    def __call__(self, groupby):
        self._normalize_funcs()
        df = groupby
        while df.op.output_types[0] not in (OutputType.dataframe, OutputType.series):
            df = df.inputs[0]

        if self.raw_func == 'size':
            self.output_types = [OutputType.series]
        else:
            self.output_types = [OutputType.dataframe] \
                if groupby.op.output_types[0] == OutputType.dataframe_groupby else [OutputType.series]

        if self.output_types[0] == OutputType.dataframe:
            return self._call_dataframe(groupby, df)
        else:
            return self._call_series(groupby, df)

    @classmethod
    def _gen_shuffle_chunks(cls, op, in_df, chunks):
        # generate map chunks
        map_chunks = []
        chunk_shape = (in_df.chunk_shape[0], 1)
        for chunk in chunks:
            # no longer consider as_index=False for the intermediate phases,
            # will do reset_index at last if so
            map_op = DataFrameGroupByOperand(stage=OperandStage.map, shuffle_size=chunk_shape[0],
                                             output_types=[OutputType.dataframe_groupby])
            map_chunks.append(map_op.new_chunk([chunk], shape=(np.nan, np.nan), index=chunk.index,
                                               index_value=op.outputs[0].index_value))

        proxy_chunk = DataFrameShuffleProxy(output_types=[OutputType.dataframe]).new_chunk(map_chunks, shape=())

        # generate reduce chunks
        reduce_chunks = []
        for out_idx in itertools.product(*(range(s) for s in chunk_shape)):
            reduce_op = DataFrameGroupByOperand(
                stage=OperandStage.reduce, shuffle_key=','.join(str(idx) for idx in out_idx),
                output_types=[OutputType.dataframe_groupby])
            reduce_chunks.append(
                reduce_op.new_chunk([proxy_chunk], shape=(np.nan, np.nan), index=out_idx,
                                    index_value=op.outputs[0].index_value))

        return reduce_chunks

    @classmethod
    def _gen_map_chunks(cls, op: "DataFrameGroupByAgg", in_df, out_df, func_infos: ReductionSteps):
        agg_chunks = []
        for chunk in in_df.chunks:
            chunk_inputs = [chunk]
            agg_op = op.copy().reset_key()
            # force as_index=True for map phase
            agg_op.output_types = [OutputType.dataframe]
            agg_op._groupby_params = agg_op.groupby_params.copy()
            agg_op._groupby_params['as_index'] = True
            if isinstance(agg_op._groupby_params['by'], list):
                by = []
                for v in agg_op._groupby_params['by']:
                    if isinstance(v, (Base, Entity)):
                        by_chunk = v.cix[chunk.index[0], ]
                        chunk_inputs.append(by_chunk)
                        by.append(by_chunk)
                    else:
                        by.append(v)
                agg_op._groupby_params['by'] = by
            agg_op._stage = OperandStage.map
            agg_op._pre_funcs = func_infos.pre_funcs
            agg_op._agg_funcs = func_infos.agg_funcs
            new_index = chunk.index if len(chunk.index) == 2 else (chunk.index[0], 0)
            if op.output_types[0] == OutputType.dataframe:
                agg_chunk = agg_op.new_chunk(chunk_inputs, shape=out_df.shape, index=new_index,
                                             index_value=out_df.index_value,
                                             columns_value=out_df.columns_value)
            else:
                agg_chunk = agg_op.new_chunk(chunk_inputs, shape=(out_df.shape[0], 1), index=new_index,
                                             index_value=out_df.index_value)
            agg_chunks.append(agg_chunk)
        return agg_chunks

    @classmethod
    def _compile_funcs(cls, op: "DataFrameGroupByAgg", in_df) -> ReductionSteps:
        compiler = ReductionCompiler(store_source=True)
        if isinstance(op.func, list):
            func_iter = ((None, f) for f in op.func)
        else:
            func_iter = ((col, f) for col, funcs in op.func.items() for f in funcs)

        for col, f in func_iter:
            func_name = None
            if isinstance(f, str):
                f, func_name = _agg_functions[f], f

            func_cols = None
            if col is not None:
                func_cols = [col]
            compiler.add_function(f, in_df.ndim, cols=func_cols, func_name=func_name)
        return compiler.compile()

    @classmethod
    def _tile_with_shuffle(cls, op: "DataFrameGroupByAgg"):
        in_df = op.inputs[0]
        if len(in_df.shape) > 1:
            in_df = build_concatenated_rows_frame(in_df)
        out_df = op.outputs[0]

        index = out_df.index_value.to_pandas()
        level = 0 if not isinstance(index, pd.MultiIndex) else list(range(len(index.levels)))

        func_infos = cls._compile_funcs(op, in_df)

        # First, perform groupby and aggregation on each chunk.
        agg_chunks = cls._gen_map_chunks(op, in_df, out_df, func_infos)

        # Shuffle the aggregation chunk.
        reduce_chunks = cls._gen_shuffle_chunks(op, in_df, agg_chunks)

        # Combine groups
        agg_chunks = []
        for chunk in reduce_chunks:
            agg_op = op.copy().reset_key()
            agg_op._tileable_op_key = op.key
            agg_op._groupby_params = agg_op.groupby_params.copy()
            agg_op._groupby_params.pop('selection', None)
            # use levels instead of by for reducer
            agg_op._groupby_params.pop('by', None)
            agg_op._groupby_params['level'] = level
            agg_op._stage = OperandStage.agg
            agg_op._agg_funcs = func_infos.agg_funcs
            agg_op._post_funcs = func_infos.post_funcs
            if op.output_types[0] == OutputType.dataframe:
                agg_chunk = agg_op.new_chunk([chunk], shape=out_df.shape, index=chunk.index,
                                             index_value=out_df.index_value, dtypes=out_df.dtypes,
                                             columns_value=out_df.columns_value)
            else:
                agg_chunk = agg_op.new_chunk([chunk], shape=out_df.shape, index=(chunk.index[0],),
                                             dtype=out_df.dtype, index_value=out_df.index_value,
                                             name=out_df.name)
            agg_chunks.append(agg_chunk)

        new_op = op.copy()
        if op.output_types[0] == OutputType.dataframe:
            nsplits = ((np.nan,) * len(agg_chunks), (out_df.shape[1],))
        else:
            nsplits = ((np.nan,) * len(agg_chunks),)
        kw = out_df.params.copy()
        kw.update(dict(chunks=agg_chunks, nsplits=nsplits))
        return new_op.new_tileables([in_df], **kw)

    @classmethod
    def _tile_with_tree(cls, op: "DataFrameGroupByAgg"):
        in_df = op.inputs[0]
        if len(in_df.shape) > 1:
            in_df = build_concatenated_rows_frame(in_df)
        out_df = op.outputs[0]

        index = out_df.index_value.to_pandas()
        level = 0 if not isinstance(index, pd.MultiIndex) else list(range(len(index.levels)))

        func_infos = cls._compile_funcs(op, in_df)
        combine_size = op.combine_size
        chunks = cls._gen_map_chunks(op, in_df, out_df, func_infos)
        while len(chunks) > combine_size:
            new_chunks = []
            for idx, i in enumerate(range(0, len(chunks), combine_size)):
                chks = chunks[i: i + combine_size]
                if len(chks) == 1:
                    chk = chks[0]
                else:
                    concat_op = DataFrameConcat(output_types=[OutputType.dataframe])
                    # Change index for concatenate
                    for j, c in enumerate(chks):
                        c._index = (j, 0)
                    chk = concat_op.new_chunk(chks, dtypes=chks[0].dtypes)
                chunk_op = op.copy().reset_key()
                chunk_op._tileable_op_key = None
                chunk_op.output_types = [OutputType.dataframe]
                chunk_op._stage = OperandStage.combine
                chunk_op._groupby_params = chunk_op.groupby_params.copy()
                chunk_op._groupby_params.pop('selection', None)
                # use levels instead of by for agg
                chunk_op._groupby_params.pop('by', None)
                chunk_op._groupby_params['level'] = level
                chunk_op._agg_funcs = func_infos.agg_funcs

                new_shape = (np.nan, out_df.shape[1]) if len(out_df.shape) == 2 else (np.nan,)

                new_chunks.append(chunk_op.new_chunk([chk], index=(idx, 0), shape=new_shape,
                                                     index_value=chks[0].index_value,
                                                     columns_value=getattr(out_df, 'columns_value', None)))
            chunks = new_chunks

        concat_op = DataFrameConcat(output_types=[OutputType.dataframe])
        chk = concat_op.new_chunk(chunks, dtypes=chunks[0].dtypes)
        chunk_op = op.copy().reset_key()
        chunk_op._tileable_op_key = op.key
        chunk_op._stage = OperandStage.agg
        chunk_op._groupby_params = chunk_op.groupby_params.copy()
        chunk_op._groupby_params.pop('selection', None)
        # use levels instead of by for agg
        chunk_op._groupby_params.pop('by', None)
        chunk_op._groupby_params['level'] = level
        chunk_op._agg_funcs = func_infos.agg_funcs
        chunk_op._post_funcs = func_infos.post_funcs
        kw = out_df.params.copy()
        kw['index'] = (0, 0) if op.output_types[0] == OutputType.dataframe else (0,)
        chunk = chunk_op.new_chunk([chk], **kw)
        new_op = op.copy()
        if op.output_types[0] == OutputType.dataframe:
            nsplits = ((out_df.shape[0],), (out_df.shape[1],))
        else:
            nsplits = ((out_df.shape[0],),)

        kw = out_df.params.copy()
        kw.update(dict(chunks=[chunk], nsplits=nsplits))
        return new_op.new_tileables(op.inputs, **kw)

    @classmethod
    def tile(cls, op: "DataFrameGroupByAgg"):
        if op.method == 'auto':
            ctx = get_context()
            if ctx is not None and ctx.running_mode == RunningMode.distributed:  # pragma: no cover
                return cls._tile_with_shuffle(op)
            else:
                return cls._tile_with_tree(op)
        if op.method == 'shuffle':
            return cls._tile_with_shuffle(op)
        elif op.method == 'tree':
            return cls._tile_with_tree(op)
        else:  # pragma: no cover
            raise NotImplementedError

    @classmethod
    def _get_grouped(cls, op: "DataFrameGroupByAgg", df, ctx, copy=False, grouper=None):
        if copy:
            df = df.copy()

        params = op.groupby_params.copy()
        params.pop('as_index', None)
        selection = params.pop('selection', None)

        if grouper is not None:
            params['by'] = grouper
            params.pop('level', None)
        elif isinstance(params.get('by'), list):
            new_by = []
            for v in params['by']:
                if isinstance(v, Base):
                    new_by.append(ctx[v.key])
                else:
                    new_by.append(v)
            params['by'] = new_by

        if op.stage == OperandStage.agg:
            grouped = df.groupby(**params)
        else:
            # for the intermediate phases, do not sort
            params['sort'] = False
            grouped = df.groupby(**params)

        if selection:
            grouped = grouped[selection]
        return grouped

    @staticmethod
    def _pack_inputs(agg_funcs: List[ReductionAggStep], in_data):
        pos = 0
        out_dict = dict()
        for step in agg_funcs:
            if step.custom_reduction is None:
                out_dict[step.output_key] = in_data[pos]
            else:
                out_dict[step.output_key] = tuple(in_data[pos:pos + step.output_limit])
            pos += step.output_limit
        return out_dict

    @staticmethod
    def _do_custom_agg(op, custom_reduction, *input_objs):
        xdf = cudf if op.gpu else pd
        group_keys = functools.reduce(
            lambda x, y: x & y, (set(o.groups.keys()) for o in input_objs))

        results = []
        for group_key in group_keys:
            group_objs = [o.get_group(group_key) for o in input_objs]
            agg_done = False
            if op.stage == OperandStage.map:
                result = group_objs[0].transform(custom_reduction.pre)
                agg_done = custom_reduction.pre_with_agg
                if not isinstance(result, tuple):
                    result = (result,)
            else:
                result = group_objs

            if not agg_done:
                result = custom_reduction.agg(*result)
                if not isinstance(result, tuple):
                    result = (result,)

            if op.stage == OperandStage.agg:
                result = custom_reduction.post(*result)
                if not isinstance(result, tuple):
                    result = (result,)

            for r in result:
                r.index = xdf.Index([group_key], names=input_objs[0].grouper.keys)
            results.append(result)
        concat_result = tuple(xdf.concat(parts) for parts in zip(*results))
        return concat_result

    @staticmethod
    def _do_standard_agg(input_obj, agg_func, **_kwds):
        ndim = getattr(input_obj, 'ndim', None) or input_obj.obj.ndim
        if ndim == 2:
            result = input_obj.agg([agg_func])
            result.columns = result.columns.droplevel(-1)
            return result
        else:
            return input_obj.agg(agg_func)

    @staticmethod
    def _series_to_df(in_series, gpu):
        xdf = cudf if gpu else pd

        in_df = in_series.to_frame()
        if in_series.name is not None:
            in_df.columns = xdf.Index([in_series.name])
        return in_df

    @classmethod
    def _execute_map(cls, ctx, op: "DataFrameGroupByAgg"):
        xdf = cudf if op.gpu else pd

        in_data = ctx[op.inputs[0].key]
        if isinstance(in_data, xdf.Series) and op.output_types[0] == OutputType.dataframe:
            in_data = cls._series_to_df(in_data, op.gpu)

        # map according to map groups
        ret_map_groupbys = dict()
        grouped = cls._get_grouped(op, in_data, ctx)
        grouper = None

        for input_key, output_key, cols, func in op.pre_funcs:
            if input_key == output_key:
                ret_map_groupbys[output_key] = grouped if cols is None else grouped[cols]
            else:
                def _wrapped_func(col):
                    try:
                        return func(col)
                    except TypeError:
                        return col

                pre_df = in_data if cols is None else in_data[cols]
                try:
                    pre_df = func(pre_df)
                except TypeError:
                    pre_df = pre_df.transform(_wrapped_func)

                if grouper is None:
                    try:
                        grouper = grouped.grouper
                    except AttributeError:  # cudf does not have GroupBy.grouper
                        grouper = xdf.Series(grouped.grouping.keys, index=grouped.obj.index)

                ret_map_groupbys[output_key] = cls._get_grouped(op, pre_df, ctx, grouper=grouper)

        agg_dfs = []
        for input_key, map_func_name, _agg_func_name, custom_reduction, \
                _output_key, _output_limit, kwds in op.agg_funcs:
            input_obj = ret_map_groupbys[input_key]
            if map_func_name == 'custom_reduction':
                agg_dfs.extend(cls._do_custom_agg(op, custom_reduction, input_obj))
            else:
                agg_dfs.append(cls._do_standard_agg(input_obj, map_func_name, **kwds))
        ctx[op.outputs[0].key] = tuple(agg_dfs)

    @classmethod
    def _execute_combine(cls, ctx, op: "DataFrameGroupByAgg"):
        xdf = cudf if op.gpu else pd

        in_data_tuple = ctx[op.inputs[0].key]
        in_data_list = []
        for in_data in in_data_tuple:
            if isinstance(in_data, xdf.Series) and op.output_types[0] == OutputType.dataframe:
                in_data = cls._series_to_df(in_data, op.gpu)
            in_data_list.append(cls._get_grouped(op, in_data, ctx))
        in_data_tuple = tuple(in_data_list)
        in_data_dict = cls._pack_inputs(op.agg_funcs, in_data_tuple)

        combines = []
        for _input_key, _map_func_name, agg_func_name, custom_reduction, \
                output_key, _output_limit, kwds in op.agg_funcs:
            input_obj = in_data_dict[output_key]
            if agg_func_name == 'custom_reduction':
                combines.extend(cls._do_custom_agg(op, custom_reduction, *input_obj))
            else:
                combines.append(cls._do_standard_agg(input_obj, agg_func_name, **kwds))
            ctx[op.outputs[0].key] = tuple(combines)

    @classmethod
    def _execute_agg(cls, ctx, op: "DataFrameGroupByAgg"):
        xdf = cudf if op.gpu else pd
        out = op.outputs[0]
        col_value = out.columns_value.to_pandas() if hasattr(out, 'columns_value') else None

        in_data_tuple = ctx[op.inputs[0].key]
        in_data_list = []
        for in_data in in_data_tuple:
            if isinstance(in_data, xdf.Series) and op.output_types[0] == OutputType.dataframe:
                in_data = cls._series_to_df(in_data, op.gpu)
            in_data_list.append(in_data)
        in_data_tuple = tuple(in_data_list)
        in_data_dict = cls._pack_inputs(op.agg_funcs, in_data_tuple)

        for _input_key, _map_func_name, agg_func_name, custom_reduction, \
            output_key, _output_limit, kwds in op.agg_funcs:
            if agg_func_name == 'custom_reduction':
                input_obj = tuple(cls._get_grouped(op, o, ctx) for o in in_data_dict[output_key])
                in_data_dict[output_key] = cls._do_custom_agg(op, custom_reduction, *input_obj)[0]
            else:
                input_obj = cls._get_grouped(op, in_data_dict[output_key], ctx)
                in_data_dict[output_key] = cls._do_standard_agg(input_obj, agg_func_name, **kwds)

        aggs = []
        for input_keys, _output_key, func_name, cols, func in op.post_funcs:
            if cols is None:
                func_inputs = [in_data_dict[k] for k in input_keys]
            else:
                func_inputs = [in_data_dict[k][cols] for k in input_keys]

            agg_df = func(*func_inputs)
            if isinstance(agg_df, np.ndarray):
                agg_df = xdf.DataFrame(agg_df, index=func_inputs[0].index)

            new_cols = None
            if out.ndim == 2 and col_value is not None:
                if col_value.nlevels > agg_df.columns.nlevels:
                    new_cols = xdf.MultiIndex.from_product([agg_df.columns, [func_name]])
                elif agg_df.shape[-1] == 1 and func_name in col_value:
                    new_cols = xdf.Index([func_name])
            aggs.append((agg_df, new_cols))

        for agg_df, new_cols in aggs:
            if new_cols is not None:
                agg_df.columns = new_cols
        aggs = [item[0] for item in aggs]

        if out.ndim == 2:
            result = xdf.concat(aggs, axis=1)
            if not op.groupby_params.get('as_index', True) \
                    and col_value.nlevels == result.columns.nlevels:
                result.reset_index(inplace=True)
            result = result.reindex(col_value, axis=1)
        else:
            result = xdf.concat(aggs)
            if result.ndim == 2:
                result = result.iloc[:, 0]
            result.name = out.name
        ctx[op.outputs[0].key] = result

    @classmethod
    @redirect_custom_log
    @enter_current_session
    def execute(cls, ctx, op: "DataFrameGroupByAgg"):
        try:
            pd.set_option('mode.use_inf_as_na', op.use_inf_as_na)
            if op.stage == OperandStage.map:
                cls._execute_map(ctx, op)
            elif op.stage == OperandStage.combine:
                cls._execute_combine(ctx, op)
            elif op.stage == OperandStage.agg:
                cls._execute_agg(ctx, op)
            elif op.raw_func == 'size':
                xp = cp if op.gpu else np
                ctx[op.outputs[0].key] = xp.array(ctx[op.inputs[0].key].agg(op.raw_func)) \
                    .reshape(op.outputs[0].shape)
            else:
                xp = cp if op.gpu else np
                result = ctx[op.inputs[0].key].agg(op.raw_func)
                if op.output_types[0] == OutputType.tensor:
                    result = xp.array(result)
                ctx[op.outputs[0].key] = result
        finally:
            pd.reset_option('mode.use_inf_as_na')


def agg(groupby, func, method='auto', *args, **kwargs):
    """
    Aggregate using one or more operations on grouped data.

    Parameters
    ----------
    groupby : Mars Groupby
        Groupby data.
    func : str or list-like
        Aggregation functions.
    method : {'auto', 'shuffle', 'tree'}, default 'auto'
        'tree' method provide a better performance, 'shuffle' is recommended
        if aggregated result is very large, 'auto' will use 'shuffle' method
        in distributed mode and use 'tree' in local mode.

    Returns
    -------
    Series or DataFrame
        Aggregated result.
    """

    # When perform a computation on the grouped data, we won't shuffle
    # the data in the stage of groupby and do shuffle after aggregation.
    if not isinstance(groupby, GROUPBY_TYPE):
        raise TypeError(f'Input should be type of groupby, not {type(groupby)}')

    if method not in ['shuffle', 'tree', 'auto']:
        raise ValueError(f"Method {method} is not available, "
                         "please specify 'tree' or 'shuffle")

    if not is_funcs_aggregate(func):
        return groupby.transform(func, *args, _call_agg=True, **kwargs)

    use_inf_as_na = kwargs.pop('_use_inf_as_na', options.dataframe.mode.use_inf_as_na)
    agg_op = DataFrameGroupByAgg(func=func, method=method, raw_func=func,
                                 groupby_params=groupby.op.groupby_params,
                                 combine_size=options.combine_size,
                                 use_inf_as_na=use_inf_as_na)
    return agg_op(groupby)
