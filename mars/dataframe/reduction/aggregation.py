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

from collections import namedtuple, OrderedDict
from collections.abc import Iterable
from typing import Union, List, Dict

import numpy as np
import pandas as pd

from ... import opcodes
from ...config import options
from ...operands import OperandStage
from ...serialize import BoolField, AnyField, DictField
from ...utils import tokenize, ceildiv, lazy_import
from ..merge import DataFrameConcat
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import build_empty_df, build_empty_series, parse_index, validate_axis

cp = lazy_import('cupy', globals=globals(), rename='cp')
cudf = lazy_import('cudf', globals=globals())

_builtin_aggregation_functions = {'sum', 'prod', 'min', 'max', 'count', 'size', 'mean', 'var', 'std'}
_stage_info = namedtuple('_stage_info', ('map_groups', 'map_sources', 'combine_groups', 'combine_sources',
                                         'agg_sources', 'agg_columns', 'agg_funcs', 'key_to_funcs',
                                         'valid_columns'))
_series_col_name = 'col_name'


class DataFrameAggregate(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.AGGREGATE

    _func = AnyField('func')
    _raw_func = AnyField('raw_func')
    _axis = AnyField('axis')
    _use_inf_as_na = BoolField('use_inf_as_na')

    _map_groups = DictField('map_groups')
    _map_sources = DictField('map_sources')
    _combine_groups = DictField('combine_groups')
    _combine_sources = DictField('combine_sources')
    _agg_sources = DictField('agg_sources')
    _agg_columns = DictField('agg_columns')
    _agg_funcs = DictField('agg_funcs')
    _key_to_funcs = DictField('keys_to_funcs')

    def __init__(self, func=None, raw_func=None, axis=None, use_inf_as_na=None, map_groups=None,
                 map_sources=None, combine_groups=None, combine_sources=None, agg_sources=None,
                 agg_columns=None, agg_funcs=None, key_to_funcs=None, object_type=None, stage=None, **kw):
        super().__init__(_func=func, _raw_func=raw_func, _axis=axis, _use_inf_as_na=use_inf_as_na,
                         _map_groups=map_groups, _map_sources=map_sources, _combine_groups=combine_groups,
                         _combine_sources=combine_sources, _agg_sources=agg_sources,
                         _agg_columns=agg_columns, _agg_funcs=agg_funcs, _key_to_funcs=key_to_funcs,
                         _stage=stage, _object_type=object_type, **kw)

    @property
    def func(self):
        return self._func

    @property
    def raw_func(self):
        return self._raw_func

    @property
    def axis(self) -> int:
        return self._axis

    @property
    def use_inf_as_na(self) -> int:
        return self._use_inf_as_na

    @property
    def map_groups(self) -> Dict:
        return self._map_groups

    @property
    def map_sources(self) -> Dict:
        return self._map_sources

    @property
    def combine_groups(self) -> Dict:
        return self._combine_groups

    @property
    def combine_sources(self) -> Dict:
        return self._combine_sources

    @property
    def agg_sources(self) -> Dict:
        return self._agg_sources

    @property
    def agg_columns(self) -> Dict:
        return self._agg_columns

    @property
    def agg_funcs(self) -> Dict:
        return self._agg_funcs

    @property
    def key_to_funcs(self) -> Dict:
        return self._key_to_funcs

    def _calc_result_shape(self, df):
        if self.object_type == ObjectType.dataframe:
            empty_obj = build_empty_df(df.dtypes, index=pd.RangeIndex(0, 10))
        else:
            empty_obj = build_empty_series(df.dtype, index=pd.RangeIndex(0, 10), name=df.name)

        result_df = empty_obj.agg(self.func, axis=self.axis)

        if isinstance(result_df, pd.DataFrame):
            self._object_type = ObjectType.dataframe
            return result_df.dtypes, result_df.index
        elif isinstance(result_df, pd.Series):
            self._object_type = ObjectType.series
            return pd.Series([result_df.dtype], index=[result_df.name]), result_df.index
        else:
            self._object_type = ObjectType.scalar
            return np.array(result_df).dtype, None

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

    def __call__(self, df):
        dtypes, index = self._calc_result_shape(df)
        self._normalize_funcs()
        if self._object_type == ObjectType.dataframe:
            if self.axis == 0:
                new_shape = (len(index), len(dtypes))
                new_index = parse_index(index, store_data=True)
            else:
                new_shape = (df.shape[0], len(dtypes))
                new_index = df.index_value
            return self.new_dataframe([df], shape=new_shape, dtypes=dtypes, index_value=new_index,
                                      columns_value=parse_index(dtypes.index, store_data=True))
        elif self._object_type == ObjectType.series:
            if df.op.object_type == ObjectType.series:
                new_shape = (len(index),)
                new_index = parse_index(index, store_data=True)
            else:
                new_shape = (df.shape[1 - self.axis],)
                new_index = [df.columns_value, df.index_value][self.axis]
            return self.new_series([df], shape=new_shape, dtype=dtypes[0], name=dtypes.index[0],
                                   index_value=new_index)
        else:
            return self.new_scalar([df], dtype=dtypes)

    @staticmethod
    def _safe_append(d, key, val):
        if key not in d:
            d[key] = []
        if val not in d[key]:
            d[key].append(val)

    @classmethod
    def _gen_chunk_stage_info(cls, op_func: Union[List, Dict], chunk_cols=None):
        map_groups = OrderedDict()
        map_sources = OrderedDict()
        combine_groups = OrderedDict()
        combine_sources = OrderedDict()
        agg_sources = OrderedDict()
        agg_columns = OrderedDict()
        agg_funcs = OrderedDict()
        key_to_funcs = OrderedDict()
        valid_columns = []

        def _clean_dict(d):
            return OrderedDict((k, sorted(v) if v != [None] else None) for k, v in d.items())

        def _fun_to_str(fun):
            if isinstance(fun, str):
                return fun
            fun_str = tokenize(fun)
            key_to_funcs[fun_str] = fun
            return fun if isinstance(fun, str) else tokenize(fun)

        def _add_column_to_functions(col, fun_name, mappers, combiners, aggregator):
            sources = []
            for mapper, combiner in zip(mappers, combiners):
                mapper_str, combiner_str = _fun_to_str(mapper), _fun_to_str(combiner)
                cls._safe_append(map_groups, mapper_str, col)
                cls._safe_append(combine_groups, (mapper_str, combiner_str), col)
                sources.append((mapper_str, combiner_str))

            for mapper, combiner in zip(mappers, combiners):
                if callable(combiner):
                    combine_sources[(_fun_to_str(mapper), _fun_to_str(combiner))] = sources

            agg_sources[fun_name] = sources
            cls._safe_append(agg_columns, fun_name, col)
            agg_funcs[fun_name] = _fun_to_str(aggregator)

        chunk_cols = set(chunk_cols) if chunk_cols is not None else None
        if isinstance(op_func, list):
            op_func = {None: op_func}
        for col, funcs in op_func.items():
            if col is not None:
                if chunk_cols is not None and col not in chunk_cols:
                    continue
                valid_columns.append(col)
            for func in funcs:
                if func in {'sum', 'prod', 'min', 'max'}:
                    _add_column_to_functions(col, func, [func], [func], func)
                elif func in {'count', 'size'}:
                    _add_column_to_functions(col, func, [func], ['sum'], 'sum')
                elif func == 'mean':
                    def _mean(sum_data, count_data, axis=0):
                        return sum_data.sum(axis=axis) / count_data.sum(axis=axis)

                    _add_column_to_functions(col, func, ['sum', 'count'], ['sum', 'sum'], _mean)
                elif func in {'var', 'std'}:
                    def _reduce_var(sum_data, count_data, var_data, axis=0):
                        reduced_cnt = count_data.sum(axis=axis)
                        var_square = var_data * (count_data - 1)
                        avg = sum_data.sum(axis=axis) / reduced_cnt
                        avg_diff = (sum_data / count_data).subtract(avg, axis=1 - axis)
                        var_square = var_square.sum(axis=axis) + (count_data * avg_diff ** 2).sum(axis=axis)
                        return var_square / (reduced_cnt - 1)

                    def _reduce_std(*args, **kwargs):
                        return np.sqrt(_reduce_var(*args, **kwargs))

                    _add_column_to_functions(col, func, ['sum', 'count', 'var'],
                                             ['sum', 'sum', _reduce_var],
                                             _reduce_var if func == 'var' else _reduce_std)
                else:  # pragma: no cover
                    raise NotImplementedError

        return _stage_info(map_groups=_clean_dict(map_groups), map_sources=map_sources,
                           combine_groups=_clean_dict(combine_groups), combine_sources=combine_sources,
                           agg_sources=agg_sources, agg_columns=_clean_dict(agg_columns),
                           agg_funcs=agg_funcs, key_to_funcs=key_to_funcs, valid_columns=valid_columns or None)

    @classmethod
    def _gen_map_chunks(cls, op, in_df, out_df, stage_infos: List[_stage_info],
                        input_index_to_output: Dict[int, int]):
        axis = op.axis

        if axis == 0:
            agg_chunks_shape = (in_df.chunk_shape[0], len(stage_infos)) \
                               if len(in_df.chunk_shape) == 2 else (in_df.chunk_shape[0], 1)
        else:
            agg_chunks_shape = (len(stage_infos), in_df.chunk_shape[1])

        agg_chunks = np.empty(agg_chunks_shape, dtype=np.object)
        for chunk in in_df.chunks:
            input_index = chunk.index[1 - axis] if len(chunk.index) > 1 else 0
            if input_index not in input_index_to_output:
                continue
            map_op = op.copy().reset_key()
            new_axis_index = input_index_to_output[input_index]
            stage_info = stage_infos[new_axis_index]
            # force as_index=True for map phase
            map_op._object_type = ObjectType.dataframe
            map_op._stage = OperandStage.map
            map_op._map_groups = stage_info.map_groups
            map_op._map_sources = stage_info.map_sources
            map_op._combine_groups = stage_info.combine_groups
            map_op._key_to_funcs = stage_info.key_to_funcs

            if axis == 0:
                new_index = (chunk.index[0], new_axis_index) if len(chunk.index) == 2 else (chunk.index[0], 0)
            else:
                new_index = (new_axis_index, chunk.index[1])

            if op.object_type == ObjectType.dataframe:
                if axis == 0:
                    new_shape = (1, chunk.shape[1] if len(chunk.shape) > 1 else 1)
                else:
                    new_shape = (chunk.shape[1] if len(chunk.shape) > 1 else 1, 1)
                agg_chunk = map_op.new_chunk(
                    [chunk], shape=new_shape, index=new_index, index_value=chunk.index_value,
                    columns_value=chunk.columns_value)
            elif op.object_type == ObjectType.series:
                agg_chunk = map_op.new_chunk([chunk], shape=(out_df.shape[0], 1), index=new_index,
                                             index_value=out_df.index_value, name=out_df.name)
            else:  # scalar target
                agg_chunk = map_op.new_chunk([chunk], shape=(1, 1), index=new_index)
            agg_chunks[agg_chunk.index] = agg_chunk
        return agg_chunks

    @classmethod
    def _tile_single_chunk(cls, op: "DataFrameAggregate"):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        chunk_op = op.copy().reset_key()
        if op.object_type == ObjectType.dataframe:
            chunk = chunk_op.new_chunk(in_df.chunks, index=(0, 0), shape=out_df.shape,
                                       index_value=out_df.index_value, columns_value=out_df.columns_value,
                                       dtypes=out_df.dtypes)
        elif op.object_type == ObjectType.series:
            chunk = chunk_op.new_chunk(in_df.chunks, index=(0,), shape=out_df.shape, dtype=out_df.dtype,
                                       index_value=out_df.index_value, name=out_df.name)
        else:
            chunk = chunk_op.new_chunk(in_df.chunks, dtype=out_df.dtype, shape=())

        tileable_op = op.copy().reset_key()
        kw = out_df.params.copy()
        kw.update(dict(chunks=[chunk], nsplits=((x,) for x in out_df.shape)))
        return tileable_op.new_tileables([in_df], **kw)

    @classmethod
    def _tile_size(cls, op: "DataFrameAggregate"):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        chunks = []
        for c in in_df.chunks:
            chunk_op = op.copy().reset_key()
            chunks.append(chunk_op.new_chunk([c], index=c.index, shape=(1,) * len(in_df.shape),
                                             dtype=out_df.dtype))

        tileable_op = op.copy().reset_key()
        nsplits = tuple((1,) * s for s in in_df.chunk_shape)
        tileable = tileable_op.new_tileable(out_df.inputs, chunks=chunks, nsplits=nsplits,
                                            shape=in_df.chunk_shape, dtype=out_df.dtype)
        return [tileable.sum()._inplace_tile()]

    @classmethod
    def _tile_tree(cls, op: "DataFrameAggregate"):
        in_df = op.inputs[0]
        out_df = op.outputs[0]
        combine_size = options.combine_size
        axis = op.axis

        input_index_to_output = dict()
        output_index_to_input = []
        axis_stage_infos = []
        if len(in_df.chunk_shape) > 1:
            for col_idx in range(in_df.chunk_shape[1 - axis]):
                idx_chunk = in_df.cix[0, col_idx] if axis == 0 else in_df.cix[col_idx, 0]
                stage_info = cls._gen_chunk_stage_info(
                    op.func, idx_chunk.dtypes.index if axis == 0 else None)
                if not stage_info.map_groups:
                    continue
                input_index_to_output[col_idx] = len(axis_stage_infos)
                output_index_to_input.append(col_idx)
                axis_stage_infos.append(stage_info)
        else:
            stage_info = cls._gen_chunk_stage_info(op.func, [_series_col_name])
            input_index_to_output[0] = 0
            axis_stage_infos.append(stage_info)

        chunks = cls._gen_map_chunks(op, in_df, out_df, axis_stage_infos, input_index_to_output)
        while chunks.shape[axis] > combine_size:
            if axis == 0:
                new_chunks_shape = (ceildiv(chunks.shape[0], combine_size), chunks.shape[1])
            else:
                new_chunks_shape = (chunks.shape[0], ceildiv(chunks.shape[1], combine_size))

            new_chunks = np.empty(new_chunks_shape, dtype=np.object)
            for idx0, i in enumerate(range(0, chunks.shape[axis], combine_size)):
                for idx1 in range(chunks.shape[1 - axis]):
                    stage_info = axis_stage_infos[idx1]
                    if axis == 0:
                        chks = chunks[i: i + combine_size, idx1]
                        chunk_index = (idx0, idx1)
                    else:
                        chks = chunks[idx1, i: i + combine_size]
                        chunk_index = (idx1, idx0)

                    chks = chks.reshape((chks.shape[0],)).tolist()
                    if len(chks) == 1:
                        chk = chks[0]
                    else:
                        concat_op = DataFrameConcat(object_type=ObjectType.dataframe, axis=axis)
                        # Change index for concatenate
                        for j, c in enumerate(chks):
                            c._index = (j, 0) if axis == 0 else (0, j)
                        chk = concat_op.new_chunk(chks, dtypes=chks[0].dtypes)
                    chunk_op = op.copy().reset_key()
                    chunk_op._object_type = ObjectType.dataframe
                    chunk_op._stage = OperandStage.combine
                    chunk_op._combine_groups = stage_info.combine_groups
                    chunk_op._combine_sources = stage_info.combine_sources
                    chunk_op._key_to_funcs = stage_info.key_to_funcs

                    if axis == 0:
                        new_chunks[chunk_index] = chunk_op.new_chunk(
                            [chk], index=chunk_index, shape=(np.nan, chks[0].shape[1]),
                            index_value=chks[0].index_value)
                    else:
                        new_chunks[chunk_index] = chunk_op.new_chunk(
                            [chk], index=chunk_index, shape=(chks[0].shape[0], np.nan),
                            index_value=chks[0].columns_value)
            chunks = new_chunks

        agg_chunks = []
        for idx in range(chunks.shape[1 - axis]):
            stage_info = axis_stage_infos[idx]

            concat_op = DataFrameConcat(object_type=ObjectType.dataframe, axis=axis)
            if axis == 0:
                chks = chunks[:, idx]
            else:
                chks = chunks[idx, :]
            chks = chks.reshape((chks.shape[0],)).tolist()
            chk = concat_op.new_chunk(chks, dtypes=chks[0].dtypes)
            chunk_op = op.copy().reset_key()
            chunk_op._stage = OperandStage.agg
            chunk_op._combine_groups = stage_info.combine_groups
            chunk_op._agg_columns = stage_info.agg_columns
            chunk_op._agg_funcs = stage_info.agg_funcs
            chunk_op._agg_sources = stage_info.agg_sources
            chunk_op._key_to_funcs = stage_info.key_to_funcs

            kw = out_df.params.copy()
            if op.object_type == ObjectType.dataframe:
                if axis == 0:
                    src_col_chunk = in_df.cix[0, output_index_to_input[idx]]
                    if axis_stage_infos[idx].valid_columns is None:
                        columns_value = src_col_chunk.columns_value
                        shape_len = src_col_chunk.shape[1]
                    else:
                        columns_value = parse_index(pd.Index(axis_stage_infos[idx].valid_columns), store_data=True)
                        shape_len = len(axis_stage_infos[idx].valid_columns)
                    kw.update(dict(shape=(out_df.shape[0], shape_len), columns_value=columns_value,
                                   index=(0, idx), dtypes=out_df.dtypes[columns_value.to_pandas()]))
                else:
                    src_col_chunk = in_df.cix[output_index_to_input[idx], 0]
                    kw.update(dict(index=(idx, 0), index_value=src_col_chunk.index_value,
                                   shape=(src_col_chunk.shape[0], out_df.shape[1]),
                                   dtypes=out_df.dtypes))
            else:
                if op.object_type == ObjectType.series:
                    if in_df.op.object_type == ObjectType.series:
                        index_value, shape = out_df.index_value, out_df.shape
                    elif axis == 0:
                        src_chunk = in_df.cix[0, output_index_to_input[idx]]
                        index_value, shape = src_chunk.columns_value, (src_chunk.shape[1],)
                    else:
                        src_chunk = in_df.cix[output_index_to_input[idx], 0]
                        index_value, shape = src_chunk.index_value, (src_chunk.shape[0],)
                    kw.update(dict(name=out_df.name, dtype=out_df.dtype, index=(idx,),
                                   index_value=index_value, shape=shape))
                else:
                    kw.update(dict(index=(), shape=(), dtype=out_df.dtype))
            agg_chunks.append(chunk_op.new_chunk([chk], **kw))

        new_op = op.copy()
        if op.object_type == ObjectType.dataframe:
            if axis == 0:
                nsplits = ((out_df.shape[0],), tuple(c.shape[1] for c in agg_chunks))
            else:
                nsplits = (tuple(c.shape[0] for c in agg_chunks), (out_df.shape[1],))
            return new_op.new_tileables(op.inputs, chunks=agg_chunks, nsplits=nsplits, dtypes=out_df.dtypes,
                                        shape=out_df.shape, index_value=out_df.index_value,
                                        columns_value=out_df.columns_value)
        elif op.object_type == ObjectType.series:
            nsplits = (tuple(c.shape[0] for c in agg_chunks),)
            return new_op.new_tileables(op.inputs, chunks=agg_chunks, nsplits=nsplits, dtype=out_df.dtype,
                                        shape=out_df.shape, index_value=out_df.index_value, name=out_df.name)
        else:  # scalar
            return new_op.new_tileables(op.inputs, chunks=agg_chunks, dtype=out_df.dtype,
                                        shape=(), nsplits=())

    @classmethod
    def tile(cls, op: "DataFrameAggregate"):
        in_df = op.inputs[0]

        if len(in_df.chunks) == 1:
            return cls._tile_single_chunk(op)
        elif in_df.ndim == 2 and op.raw_func == 'size':
            return cls._tile_size(op)
        else:
            return cls._tile_tree(op)

    @staticmethod
    def _wrap_df(xdf, value, columns=None, transform=False):
        if isinstance(value, (np.generic, int, float, complex)):
            value = xdf.DataFrame([value], columns=columns)
        else:
            value = xdf.DataFrame(value, columns=columns)
        return value.T if transform else value

    @classmethod
    def _execute_map(cls, ctx, op: "DataFrameAggregate"):
        xdf = cudf if op.gpu else pd
        in_data = ctx[op.inputs[0].key]
        axis = op.axis
        axis_index = op.outputs[0].index[axis]

        # map according to map groups
        ret_map_dfs = dict()
        for map_func_str, cols in op.map_groups.items():
            if cols is None:
                src_df = in_data
            else:
                src_df = in_data[cols]
            map_func = map_func_str if map_func_str != 'size' else (lambda x: x.size)
            ret_map_dfs[map_func_str] = cls._wrap_df(xdf, src_df.agg(map_func, axis=axis),
                                                     columns=[axis_index], transform=axis == 0)

        ret_combine_dfs = OrderedDict()
        for func_strs, cols in op.combine_groups.items():
            if cols is None:
                ret_combine_dfs[func_strs] = ret_map_dfs[func_strs[0]]
            else:
                ret_combine_dfs[func_strs] = ret_map_dfs[func_strs[0]][cols]
        ctx[op.outputs[0].key] = tuple(ret_combine_dfs.values())

    @classmethod
    def _execute_combine(cls, ctx, op: "DataFrameAggregate"):
        xdf = cudf if op.gpu else pd
        in_data = ctx[op.inputs[0].key]
        in_data_dict = dict(zip(op.combine_groups.keys(), in_data))
        axis = op.axis
        axis_index = op.outputs[0].index[axis]

        combines = []
        for fun_strs, df in zip(op.combine_groups.keys(), in_data):
            if fun_strs[-1] in op.key_to_funcs:
                func = op.key_to_funcs[fun_strs[-1]]
                sources = [in_data_dict[fs] for fs in op.combine_sources[fun_strs]]
                result = cls._wrap_df(xdf, func(*sources, axis=axis), columns=[axis_index],
                                      transform=axis == 0)
            else:
                result = cls._wrap_df(xdf, df.agg(fun_strs[-1], axis=axis), columns=[axis_index],
                                      transform=axis == 0)
            combines.append(result)
        ctx[op.outputs[0].key] = tuple(combines)

    @classmethod
    def _execute_agg(cls, ctx, op: "DataFrameAggregate"):
        xdf = cudf if op.gpu else pd
        in_data = ctx[op.inputs[0].key]
        in_data_dict = dict(zip(op.combine_groups.keys(), in_data))
        axis = op.axis

        aggs = []
        for func_name, func_sources in op.agg_sources.items():
            func_str = op.agg_funcs[func_name]
            func_cols = op.agg_columns[func_name]
            if func_cols is None:
                func_inputs = [in_data_dict[src] for src in func_sources]
            else:
                func_inputs = [in_data_dict[src][func_cols] for src in func_sources]

            if func_str in op.key_to_funcs:
                func = op.key_to_funcs[func_str]
                agg_series = func(*func_inputs, axis=axis)
            else:
                agg_series = func_inputs[0].agg(func_str, axis=axis)

            agg_series.name = func_name
            aggs.append(cls._wrap_df(xdf, agg_series, transform=axis == 0))

        concat_df = xdf.concat(aggs, axis=axis)
        if op.object_type == ObjectType.series:
            if concat_df.shape[1] > 1:
                concat_df = concat_df.iloc[0, :]
            else:
                concat_df = concat_df.iloc[:, 0]
            concat_df.name = op.outputs[0].name

            concat_df = concat_df.astype(op.outputs[0].dtype, copy=False)
        elif op.object_type == ObjectType.scalar:
            concat_df = concat_df.iloc[0, 0].astype(op.outputs[0].dtype)
        else:
            if axis == 0:
                concat_df = concat_df.reindex(op.outputs[0].index_value.to_pandas())
            else:
                concat_df = concat_df[op.outputs[0].columns_value.to_pandas()]

            concat_df = concat_df.astype(op.outputs[0].dtypes, copy=False)
        ctx[op.outputs[0].key] = concat_df

    @classmethod
    def execute(cls, ctx, op: "DataFrameAggregate"):
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
                ctx[op.outputs[0].key] = xp.array(ctx[op.inputs[0].key].agg(op.raw_func, axis=op.axis)) \
                    .reshape(op.outputs[0].shape)
            else:
                ctx[op.outputs[0].key] = ctx[op.inputs[0].key].agg(op.raw_func, axis=op.axis)
        finally:
            pd.reset_option('mode.use_inf_as_na')


def _is_funcs_agg(func):
    to_check = []
    if isinstance(func, list):
        to_check.extend(func)
    elif isinstance(func, dict):
        for f in func.values():
            if isinstance(f, Iterable) and not isinstance(f, str):
                to_check.extend(f)
            else:
                to_check.append(f)
    else:
        to_check.append(func)

    for f in to_check:
        if f not in _builtin_aggregation_functions:
            return False
    return True


def aggregate(df, func, axis=0, **kw):
    if not _is_funcs_agg(func):
        return df.transform(func, axis=axis, _call_agg=True)

    axis = validate_axis(axis, df)
    use_inf_as_na = kw.pop('_use_inf_as_na', options.dataframe.mode.use_inf_as_na)
    if (df.op.object_type == ObjectType.series or axis == 1) and isinstance(func, dict):
        raise NotImplementedError('Currently cannot aggregate dicts over axis=1 on %s'
                                  % type(df).__name__)
    op = DataFrameAggregate(func=func, axis=axis, object_type=df.op.object_type,
                            use_inf_as_na=use_inf_as_na)
    return op(df)
