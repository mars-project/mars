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

import copy
import functools
import itertools
from collections import OrderedDict
from collections.abc import Iterable
from typing import List, Dict, Union

import numpy as np
import pandas as pd

from ... import opcodes, tensor as mars_tensor
from ...config import options
from ...core import OutputType, ENTITY_TYPE, enter_mode, recursive_tile
from ...core.custom_log import redirect_custom_log
from ...core.operand import OperandStage
from ...lib.version import parse as parse_version
from ...serialization.serializables import BoolField, AnyField, Int32Field, ListField, DictField
from ...utils import ceildiv, lazy_import, enter_current_session
from ..core import INDEX_CHUNK_TYPE
from ..merge import DataFrameConcat
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import build_df, build_empty_df, build_series, parse_index, validate_axis
from .core import CustomReduction, ReductionCompiler, ReductionSteps, ReductionPreStep, \
    ReductionAggStep, ReductionPostStep

cp = lazy_import('cupy', globals=globals(), rename='cp')
cudf = lazy_import('cudf', globals=globals())

_agg_size_as_series = parse_version(pd.__version__) >= parse_version('1.3.0')


def where_function(cond, var1, var2):
    if var1.ndim >= 1:
        return var1.where(cond, var2)
    else:
        if isinstance(var1, ENTITY_TYPE):
            return mars_tensor.where(cond, var1, var2)
        else:
            return np.where(cond, var1, var2).item()


_agg_functions = {
    'sum': lambda x, skipna=None: x.sum(skipna=skipna),
    'prod': lambda x, skipna=None: x.prod(skipna=skipna),
    'product': lambda x, skipna=None: x.product(skipna=skipna),
    'min': lambda x, skipna=None: x.min(skipna=skipna),
    'max': lambda x, skipna=None: x.max(skipna=skipna),
    'all': lambda x, skipna=None: x.all(skipna=skipna),
    'any': lambda x, skipna=None: x.any(skipna=skipna),
    'count': lambda x: x.count(),
    'size': lambda x: x._reduction_size(),
    'mean': lambda x, skipna=None: x.mean(skipna=skipna),
    'var': lambda x, skipna=None, ddof=1: x.var(skipna=skipna, ddof=ddof),
    'std': lambda x, skipna=None, ddof=1: x.std(skipna=skipna, ddof=ddof),
    'sem': lambda x, skipna=None, ddof=1: x.sem(skipna=skipna, ddof=ddof),
    'skew': lambda x, skipna=None, bias=False: x.skew(skipna=skipna, bias=bias),
    'kurt': lambda x, skipna=None, bias=False: x.kurt(skipna=skipna, bias=bias),
    'kurtosis': lambda x, skipna=None, bias=False: x.kurtosis(skipna=skipna, bias=bias),
}


class DataFrameAggregate(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.AGGREGATE

    _raw_func = AnyField('raw_func')
    _raw_func_kw = DictField('raw_func_kw')
    _func = AnyField('func')
    _func_rename = ListField('func_rename')
    _axis = AnyField('axis')
    _numeric_only = BoolField('numeric_only')
    _bool_only = BoolField('bool_only')
    _use_inf_as_na = BoolField('use_inf_as_na')

    _combine_size = Int32Field('combine_size')
    _pre_funcs = ListField('pre_funcs')
    _agg_funcs = ListField('agg_funcs')
    _post_funcs = ListField('post_funcs')

    def __init__(self, raw_func=None, raw_func_kw=None, func=None, func_rename=None,
                 axis=None, use_inf_as_na=None, numeric_only=None, bool_only=None,
                 combine_size=None, pre_funcs=None, agg_funcs=None, post_funcs=None,
                 output_types=None, stage=None, **kw):
        super().__init__(_raw_func=raw_func, _raw_func_kw=raw_func_kw, _func=func,
                         _func_rename=func_rename, _axis=axis, _use_inf_as_na=use_inf_as_na,
                         _numeric_only=numeric_only, _bool_only=bool_only,
                         _combine_size=combine_size, _pre_funcs=pre_funcs, _agg_funcs=agg_funcs,
                         _post_funcs=post_funcs, _output_types=output_types, stage=stage, **kw)

    @property
    def raw_func(self):
        return self._raw_func

    @property
    def raw_func_kw(self) -> Dict:
        return self._raw_func_kw

    @property
    def func(self) -> Union[List, Dict[str, List]]:
        return self._func

    @property
    def func_rename(self) -> List:
        return self._func_rename

    @property
    def axis(self) -> int:
        return self._axis

    @property
    def numeric_only(self) -> bool:
        return self._numeric_only

    @property
    def bool_only(self) -> bool:
        return self._bool_only

    @property
    def use_inf_as_na(self) -> int:
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

    @staticmethod
    def _filter_dtypes(op: "DataFrameAggregate", dtypes):
        if not op.numeric_only and not op.bool_only:
            return dtypes
        empty_df = build_empty_df(dtypes)
        return empty_df.select_dtypes([np.number, np.bool_] if op.numeric_only else [np.bool_]).dtypes

    def _calc_result_shape(self, df):
        if df.ndim == 2:
            if self._numeric_only:
                df = df.select_dtypes([np.number, np.bool_])
            elif self._bool_only:
                df = df.select_dtypes([np.bool_])

        if self.output_types[0] == OutputType.dataframe:
            test_obj = build_df(df, size=[2, 2], fill_value=[1, 2], ensure_string=True)
        else:
            test_obj = build_series(df, size=[2, 2], fill_value=[1, 2], name=df.name, ensure_string=True)

        result_df = test_obj.agg(self.raw_func, axis=self.axis, **self.raw_func_kw)

        if isinstance(result_df, pd.DataFrame):
            self.output_types = [OutputType.dataframe]
            return result_df.dtypes, result_df.index
        elif isinstance(result_df, pd.Series):
            self.output_types = [OutputType.series]
            return pd.Series([result_df.dtype], index=[result_df.name]), result_df.index
        else:
            self.output_types = [OutputType.scalar]
            return np.array(result_df).dtype, None

    def __call__(self, df, output_type=None, dtypes=None, index=None):
        normalize_reduction_funcs(self, ndim=df.ndim)
        if output_type is None or dtypes is None:
            with enter_mode(kernel=False, build=False):
                dtypes, index = self._calc_result_shape(df)
        else:
            self.output_types = [output_type]

        if self.output_types[0] == OutputType.dataframe:
            if self.axis == 0:
                new_shape = (len(index), len(dtypes))
                new_index = parse_index(index, store_data=True)
            else:
                new_shape = (df.shape[0], len(dtypes))
                new_index = df.index_value
            return self.new_dataframe([df], shape=new_shape, dtypes=dtypes, index_value=new_index,
                                      columns_value=parse_index(dtypes.index, store_data=True))
        elif self.output_types[0] == OutputType.series:
            if df.ndim == 1:
                new_shape = (len(index),)
                new_index = parse_index(index, store_data=True)
            elif self.axis == 0:
                new_shape = (len(index),)
                new_index = parse_index(index, store_data=True)
            else:
                new_shape = (df.shape[0],)
                new_index = df.index_value
            return self.new_series([df], shape=new_shape, dtype=dtypes[0], name=dtypes.index[0],
                                   index_value=new_index)
        elif self.output_types[0] == OutputType.tensor:
            return self.new_tileable([df], dtype=dtypes, shape=(np.nan,))
        else:
            return self.new_scalar([df], dtype=dtypes)

    @staticmethod
    def _safe_append(d, key, val):
        if key not in d:
            d[key] = []
        if val not in d[key]:
            d[key].append(val)

    @classmethod
    def _gen_map_chunks(cls, op, in_df, out_df, func_infos: List[ReductionSteps],
                        input_index_to_output: Dict[int, int]):
        axis = op.axis

        if axis == 0:
            agg_chunks_shape = (in_df.chunk_shape[0], len(func_infos)) \
                               if len(in_df.chunk_shape) == 2 else (in_df.chunk_shape[0], 1)
        else:
            agg_chunks_shape = (len(func_infos), in_df.chunk_shape[1])

        agg_chunks = np.empty(agg_chunks_shape, dtype=np.object)
        dtypes_cache = dict()
        for chunk in in_df.chunks:
            input_index = chunk.index[1 - axis] if len(chunk.index) > 1 else 0
            if input_index not in input_index_to_output:
                continue
            map_op = op.copy().reset_key()  # type: "DataFrameAggregate"
            new_axis_index = input_index_to_output[input_index]
            func_info = func_infos[new_axis_index]
            # force as_index=True for map phase
            map_op.output_types = [OutputType.dataframe] if chunk.ndim == 2 else [OutputType.series]
            map_op.stage = OperandStage.map
            map_op._pre_funcs = func_info.pre_funcs
            map_op._agg_funcs = func_info.agg_funcs

            if axis == 0:
                new_index = (chunk.index[0], new_axis_index) if len(chunk.index) == 2 else (chunk.index[0], 0)
            else:
                new_index = (new_axis_index, chunk.index[1])

            if map_op.output_types[0] == OutputType.dataframe:
                if axis == 0:
                    shape = (1, out_df.shape[-1])
                    if out_df.ndim == 2:
                        columns_value = out_df.columns_value
                        index_value = out_df.index_value
                    else:
                        columns_value = out_df.index_value
                        index_value = parse_index(pd.Index([0]), out_df.key)

                    try:
                        dtypes = dtypes_cache[chunk.index[1]]
                    except KeyError:
                        dtypes = chunk.dtypes.reindex(columns_value.to_pandas()).dropna()
                        dtypes_cache[chunk.index[1]] = dtypes

                    agg_chunk = map_op.new_chunk([chunk], shape=shape, index=new_index, dtypes=dtypes,
                                                 columns_value=columns_value, index_value=index_value)
                else:
                    shape = (out_df.shape[0], 1)
                    columns_value = parse_index(pd.Index([0]), out_df.key, store_data=True)
                    index_value = out_df.index_value

                    agg_chunk = map_op.new_chunk([chunk], shape=shape, index=new_index,
                                                 columns_value=columns_value, index_value=index_value)
            else:
                agg_chunk = map_op.new_chunk([chunk], shape=(1,), index=new_index)
            agg_chunks[agg_chunk.index] = agg_chunk
        return agg_chunks

    @classmethod
    def _tile_single_chunk(cls, op: "DataFrameAggregate"):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        chunk_op = op.copy().reset_key()
        if op.output_types[0] == OutputType.dataframe:
            chunk = chunk_op.new_chunk(in_df.chunks, index=(0, 0), shape=out_df.shape,
                                       index_value=out_df.index_value, columns_value=out_df.columns_value,
                                       dtypes=out_df.dtypes)
        elif op.output_types[0] == OutputType.series:
            chunk = chunk_op.new_chunk(in_df.chunks, index=(0,), shape=out_df.shape, dtype=out_df.dtype,
                                       index_value=out_df.index_value, name=out_df.name)
        elif op.output_types[0] == OutputType.tensor:
            chunk = chunk_op.new_chunk(in_df.chunks, index=(0,), dtype=out_df.dtype, shape=(np.nan,))
        else:
            chunk = chunk_op.new_chunk(in_df.chunks, dtype=out_df.dtype, index=(), shape=())

        tileable_op = op.copy().reset_key()
        kw = out_df.params.copy()
        kw.update(dict(chunks=[chunk], nsplits=tuple((x,) for x in out_df.shape)))
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
        ret = yield from recursive_tile(tileable.sum())
        return [ret]

    @staticmethod
    def _add_functions(op: "DataFrameAggregate", compiler: ReductionCompiler,
                       cols=None):
        if isinstance(op.func, list):
            func_iter = ((None, f) for f in op.func)
            cols_set = set(cols) if cols is not None else None
        else:
            assert cols is not None
            cols_set = set(cols) & set(op.func.keys())
            if len(cols_set) == 0:
                return False
            func_iter = ((col, f) for col, funcs in op.func.items() for f in funcs)

        func_renames = op.func_rename if op.func_rename is not None else itertools.repeat(None)
        for func_rename, (col, f) in zip(func_renames, func_iter):
            if cols_set is not None and col is not None and col not in cols_set:
                continue
            func_name = None
            if isinstance(f, str):
                f, func_name = _agg_functions[f], f
            if func_rename is not None:
                func_name = func_rename
            ndim = 1 if cols is None else 2
            func_cols = [col] if col is not None else None
            compiler.add_function(f, ndim, cols=func_cols, func_name=func_name)
        return True

    @classmethod
    def _tile_tree(cls, op: "DataFrameAggregate"):
        in_df = op.inputs[0]
        out_df = op.outputs[0]
        combine_size = op.combine_size
        axis = op.axis

        input_index_to_output = dict()
        output_index_to_input = []
        axis_func_infos = []
        dtypes_list = []
        if len(in_df.chunk_shape) > 1:
            for col_idx in range(in_df.chunk_shape[1 - axis]):
                compiler = ReductionCompiler(axis=op.axis)
                idx_chunk = in_df.cix[0, col_idx] if axis == 0 else in_df.cix[col_idx, 0]
                new_dtypes = cls._filter_dtypes(op, idx_chunk.dtypes)
                if not cls._add_functions(op, compiler, cols=list(new_dtypes.index)):
                    continue
                input_index_to_output[col_idx] = len(axis_func_infos)
                output_index_to_input.append(col_idx)
                axis_func_infos.append(compiler.compile())
                dtypes_list.append(new_dtypes)
        else:
            compiler = ReductionCompiler(axis=op.axis)
            cls._add_functions(op, compiler)
            input_index_to_output[0] = 0
            axis_func_infos.append(compiler.compile())

        chunks = cls._gen_map_chunks(op, in_df, out_df, axis_func_infos, input_index_to_output)
        while chunks.shape[axis] > combine_size:
            if axis == 0:
                new_chunks_shape = (ceildiv(chunks.shape[0], combine_size), chunks.shape[1])
            else:
                new_chunks_shape = (chunks.shape[0], ceildiv(chunks.shape[1], combine_size))

            new_chunks = np.empty(new_chunks_shape, dtype=np.object)
            for idx0, i in enumerate(range(0, chunks.shape[axis], combine_size)):
                for idx1 in range(chunks.shape[1 - axis]):
                    func_info = axis_func_infos[idx1]
                    if axis == 0:
                        chks = chunks[i: i + combine_size, idx1]
                        chunk_index = (idx0, idx1)
                        if chks[0].ndim == 1:
                            concat_shape = (len(chks),)
                            agg_shape = (1,)
                        else:
                            concat_shape = (len(chks), chks[0].shape[1])
                            agg_shape = (chks[0].shape[1], 1)
                    else:
                        chks = chunks[idx1, i: i + combine_size]
                        chunk_index = (idx1, idx0)
                        concat_shape = (chks[0].shape[0], len(chks))
                        agg_shape = (chks[0].shape[0], 1)

                    chks = chks.reshape((chks.shape[0],)).tolist()
                    if len(chks) == 1:
                        chk = chks[0]
                    else:
                        concat_op = DataFrameConcat(output_types=[OutputType.dataframe], axis=axis)
                        # Change index for concatenate
                        for j, c in enumerate(chks):
                            c._index = (j, 0) if axis == 0 else (0, j)
                        chk = concat_op.new_chunk(chks, dtypes=dtypes_list[idx1] if dtypes_list else None,
                                                  shape=concat_shape, index_value=chks[0].index_value)
                    chunk_op = op.copy().reset_key()
                    chunk_op.output_types = [OutputType.dataframe]
                    chunk_op.stage = OperandStage.combine
                    chunk_op._agg_funcs = func_info.agg_funcs

                    if axis == 0:
                        new_chunks[chunk_index] = chunk_op.new_chunk(
                            [chk], index=chunk_index, shape=agg_shape,
                            index_value=chks[0].index_value)
                    else:
                        new_chunks[chunk_index] = chunk_op.new_chunk(
                            [chk], index=chunk_index, shape=agg_shape,
                            index_value=chks[0].columns_value)
            chunks = new_chunks

        agg_chunks = []
        for idx in range(chunks.shape[1 - axis]):
            func_info = axis_func_infos[idx]

            concat_op = DataFrameConcat(output_types=[OutputType.dataframe], axis=axis)
            if axis == 0:
                chks = chunks[:, idx]
                if chks[0].ndim == 1:
                    concat_shape = (len(chks),)
                else:
                    concat_shape = (len(chks), chks[0].shape[1])
            else:
                chks = chunks[idx, :]
                concat_shape = (chks[0].shape[0], len(chks))
            chks = chks.reshape((chks.shape[0],)).tolist()
            chk = concat_op.new_chunk(chks, dtypes=dtypes_list[idx] if dtypes_list else None,
                                      shape=concat_shape, index_value=chks[0].index_value)
            chunk_op = op.copy().reset_key()
            chunk_op.stage = OperandStage.agg
            chunk_op._agg_funcs = func_info.agg_funcs
            chunk_op._post_funcs = func_info.post_funcs

            kw = out_df.params.copy()
            if op.output_types[0] == OutputType.dataframe:
                if axis == 0:
                    src_col_chunk = in_df.cix[0, output_index_to_input[idx]]
                    valid_cols = [c for pre in func_info.pre_funcs for c in pre.columns or ()]
                    if not valid_cols:
                        columns_value = src_col_chunk.columns_value
                        shape_len = src_col_chunk.shape[1]
                    else:
                        col_index = pd.Index(valid_cols).unique()
                        columns_value = parse_index(col_index, store_data=True)
                        shape_len = len(col_index)
                    kw.update(dict(shape=(out_df.shape[0], shape_len), columns_value=columns_value,
                                   index=(0, idx), dtypes=out_df.dtypes[columns_value.to_pandas()]))
                else:
                    src_col_chunk = in_df.cix[output_index_to_input[idx], 0]
                    kw.update(dict(index=(idx, 0), index_value=src_col_chunk.index_value,
                                   shape=(src_col_chunk.shape[0], out_df.shape[1]),
                                   dtypes=out_df.dtypes))
            else:
                if op.output_types[0] == OutputType.series:
                    if in_df.ndim == 1:
                        index_value, shape = out_df.index_value, out_df.shape
                    elif axis == 0:
                        out_dtypes = dtypes_list[idx]
                        index_value = parse_index(out_dtypes.index, store_data=True)
                        shape = (len(out_dtypes),)
                    else:
                        src_chunk = in_df.cix[output_index_to_input[idx], 0]
                        index_value, shape = src_chunk.index_value, (src_chunk.shape[0],)
                    kw.update(dict(name=out_df.name, dtype=out_df.dtype, index=(idx,),
                                   index_value=index_value, shape=shape))
                elif op.output_types[0] == OutputType.tensor:
                    kw.update(dict(index=(0,), shape=(np.nan,), dtype=out_df.dtype))
                else:
                    kw.update(dict(index=(), shape=(), dtype=out_df.dtype))
            agg_chunks.append(chunk_op.new_chunk([chk], **kw))

        new_op = op.copy()
        if op.output_types[0] == OutputType.dataframe:
            if axis == 0:
                nsplits = ((out_df.shape[0],), tuple(c.shape[1] for c in agg_chunks))
            else:
                nsplits = (tuple(c.shape[0] for c in agg_chunks), (out_df.shape[1],))
            return new_op.new_tileables(op.inputs, chunks=agg_chunks, nsplits=nsplits, dtypes=out_df.dtypes,
                                        shape=out_df.shape, index_value=out_df.index_value,
                                        columns_value=out_df.columns_value)
        elif op.output_types[0] == OutputType.series:
            nsplits = (tuple(c.shape[0] for c in agg_chunks),)
            return new_op.new_tileables(op.inputs, chunks=agg_chunks, nsplits=nsplits, dtype=out_df.dtype,
                                        shape=out_df.shape, index_value=out_df.index_value, name=out_df.name)
        elif op.output_types[0] == OutputType.tensor:  # unique
            return new_op.new_tileables(op.inputs, chunks=agg_chunks, dtype=out_df.dtype,
                                        shape=out_df.shape, nsplits=((np.nan,),))
        else:  # scalar
            return new_op.new_tileables(op.inputs, chunks=agg_chunks, dtype=out_df.dtype,
                                        shape=(), nsplits=())

    @classmethod
    def tile(cls, op: "DataFrameAggregate"):
        in_df = op.inputs[0]

        if len(in_df.chunks) == 1:
            return cls._tile_single_chunk(op)
        elif not _agg_size_as_series and in_df.ndim == 2 and op.raw_func == 'size':
            return (yield from cls._tile_size(op))
        else:
            return cls._tile_tree(op)

    @classmethod
    def _wrap_df(cls, op, value, index=None):
        xdf = cudf if op.gpu else pd
        axis = op.axis
        ndim = op.inputs[0].ndim

        if ndim == 2:
            dtype = None
            if isinstance(value, (np.generic, int, float, complex)):
                value = xdf.DataFrame([value], columns=index)
            elif not isinstance(value, xdf.DataFrame):
                new_index = None if not op.gpu else getattr(value, 'index', None)
                dtype = getattr(value, 'dtype', None)
                value = xdf.DataFrame(value, columns=index, index=new_index)
            else:
                return value

            value = value.T if axis == 0 else value
            if dtype == np.dtype('O') and getattr(op.outputs[0], 'dtypes', None) is not None:
                value = value.astype(op.outputs[0].dtypes)
            return value
        else:
            if isinstance(value, (np.generic, int, float, complex)):
                value = xdf.Series([value], index=index)
            elif isinstance(value, np.ndarray):
                # assert value.ndim == 0
                value = xdf.Series(value.tolist(), index=index)
            return value

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

    @classmethod
    def _do_predefined_agg(cls, op: "DataFrameAggregate", input_obj, func_name, kwds):
        if func_name == 'size':
            return input_obj.agg(lambda x: x.size, axis=op.axis)
        elif func_name == 'str_concat':
            ret = input_obj.agg(lambda x: x.str.cat(**kwds), axis=op.axis)
            if isinstance(ret, str):
                ret = pd.Series([ret])
            return ret
        else:
            if op.gpu:
                if kwds.pop('numeric_only', None):
                    raise NotImplementedError('numeric_only not implemented under cudf')
            return getattr(input_obj, func_name)(**kwds)

    @classmethod
    def _execute_map(cls, ctx, op: "DataFrameAggregate"):
        in_data = ctx[op.inputs[0].key]
        axis_index = op.outputs[0].index[op.axis]

        if in_data.ndim == 2:
            if op.numeric_only:
                in_data = in_data.select_dtypes([np.number, np.bool_])
            elif op.bool_only:
                in_data = in_data.select_dtypes([np.bool_])

        # map according to map groups
        ret_map_dfs = dict()
        in_cols_set = set(in_data.columns) if in_data.ndim == 2 else None
        for input_key, output_key, cols, func in op.pre_funcs:
            if cols and in_cols_set == set(cols):
                cols = None

            src_df = in_data if cols is None else in_data[cols]
            if input_key == output_key:
                ret_map_dfs[output_key] = src_df
            else:
                ret_map_dfs[output_key] = func(src_df, gpu=op.is_gpu())

        agg_dfs = []
        for input_key, map_func_name, _agg_func_name, custom_reduction, \
                _output_key, _output_limit, kwds in op.agg_funcs:
            input_obj = ret_map_dfs[input_key]
            if map_func_name == 'custom_reduction':
                pre_result = custom_reduction.pre(input_obj)
                if not isinstance(pre_result, tuple):
                    pre_result = (pre_result,)

                if custom_reduction.pre_with_agg:
                    # when custom_reduction.pre already aggregates, skip
                    agg_result = pre_result
                else:
                    agg_result = custom_reduction.agg(*pre_result)
                    if not isinstance(agg_result, tuple):
                        agg_result = (agg_result,)

                agg_dfs.extend([cls._wrap_df(op, r, index=[axis_index]) for r in agg_result])
            else:
                agg_dfs.append(cls._wrap_df(op, cls._do_predefined_agg(op, input_obj, map_func_name, kwds),
                                            index=[axis_index]))
        ctx[op.outputs[0].key] = tuple(agg_dfs)

    @classmethod
    def _execute_combine(cls, ctx, op: "DataFrameAggregate"):
        in_data = ctx[op.inputs[0].key]
        in_data_dict = cls._pack_inputs(op.agg_funcs, in_data)
        axis = op.axis
        axis_index = op.outputs[0].index[axis]

        combines = []
        for _input_key, _map_func_name, agg_func_name, custom_reduction, \
                output_key, _output_limit, kwds in op.agg_funcs:
            input_obj = in_data_dict[output_key]
            if agg_func_name == 'custom_reduction':
                agg_result = custom_reduction.agg(*input_obj)
                if not isinstance(agg_result, tuple):
                    agg_result = (agg_result,)
                combines.extend([cls._wrap_df(op, r, index=[axis_index])
                                 for r in agg_result])
            else:
                combines.append(cls._wrap_df(op, cls._do_predefined_agg(op, input_obj, agg_func_name, kwds),
                                             index=[axis_index]))
        ctx[op.outputs[0].key] = tuple(combines)

    @classmethod
    def _execute_agg(cls, ctx, op: "DataFrameAggregate"):
        xdf = cudf if op.gpu else pd
        xp = cp if op.gpu else np

        out = op.outputs[0]
        in_data = ctx[op.inputs[0].key]
        in_data_dict = cls._pack_inputs(op.agg_funcs, in_data)
        axis = op.axis

        # perform agg
        for _input_key, _map_func_name, agg_func_name, custom_reduction, \
                output_key, _output_limit, kwds in op.agg_funcs:
            input_obj = in_data_dict[output_key]
            if agg_func_name == 'custom_reduction':
                agg_result = custom_reduction.agg(*input_obj)
                if not isinstance(agg_result, tuple):
                    agg_result = (agg_result,)
                in_data_dict[output_key] = custom_reduction.post(*agg_result)
            else:
                in_data_dict[output_key] = cls._do_predefined_agg(op, input_obj, agg_func_name, kwds)

        aggs = []
        # perform post op
        for input_keys, _output_key, func_name, cols, func in op.post_funcs:
            if cols is None:
                func_inputs = [in_data_dict[k] for k in input_keys]
            else:
                func_inputs = [in_data_dict[k][cols] for k in input_keys]

            agg_series = func(*func_inputs, gpu=op.is_gpu())
            agg_series_ndim = getattr(agg_series, 'ndim', 0)

            ser_index = None
            if agg_series_ndim < out.ndim:
                ser_index = [func_name]
            aggs.append(cls._wrap_df(op, agg_series, index=ser_index))

        # concatenate to produce final result
        concat_df = xdf.concat(aggs, axis=axis)
        if op.output_types[0] == OutputType.series:
            if concat_df.ndim > 1:
                if op.inputs[0].ndim == 2:
                    if axis == 0:
                        concat_df = concat_df.iloc[0, :]
                    else:
                        concat_df = concat_df.iloc[:, 0]
                else:
                    concat_df = concat_df.iloc[:, 0]
            concat_df.name = op.outputs[0].name

            concat_df = concat_df.astype(op.outputs[0].dtype, copy=False)
        elif op.output_types[0] == OutputType.scalar:
            concat_df = concat_df.iloc[0]
            try:
                concat_df = concat_df.astype(op.outputs[0].dtype)
            except AttributeError:
                # concat_df may be a string and has no `astype` method
                pass
        elif op.output_types[0] == OutputType.tensor:
            concat_df = xp.array(concat_df).astype(dtype=out.dtype)
        else:
            if axis == 0:
                concat_df = concat_df.reindex(op.outputs[0].index_value.to_pandas())
            else:
                concat_df = concat_df[op.outputs[0].columns_value.to_pandas()]

            concat_df = concat_df.astype(op.outputs[0].dtypes, copy=False)
        ctx[op.outputs[0].key] = concat_df

    @classmethod
    @redirect_custom_log
    @enter_current_session
    def execute(cls, ctx, op: "DataFrameAggregate"):
        try:
            pd.set_option('mode.use_inf_as_na', op.use_inf_as_na)
            if op.stage == OperandStage.map:
                cls._execute_map(ctx, op)
            elif op.stage == OperandStage.combine:
                cls._execute_combine(ctx, op)
            elif op.stage == OperandStage.agg:
                cls._execute_agg(ctx, op)
            elif not _agg_size_as_series and op.raw_func == 'size':
                xp = cp if op.gpu else np
                ctx[op.outputs[0].key] = xp.array(ctx[op.inputs[0].key].agg(op.raw_func, axis=op.axis)) \
                    .reshape(op.outputs[0].shape)
            else:
                xp = cp if op.gpu else np
                in_obj = op.inputs[0]
                if isinstance(in_obj, INDEX_CHUNK_TYPE):
                    result = op.func[0](ctx[in_obj.key])
                else:
                    result = ctx[in_obj.key].agg(op.raw_func, axis=op.axis)

                if op.output_types[0] == OutputType.tensor:
                    result = xp.array(result)
                ctx[op.outputs[0].key] = result
        finally:
            pd.reset_option('mode.use_inf_as_na')


def is_funcs_aggregate(func, func_kw=None, ndim=2):
    func_kw = func_kw or dict()
    if ndim == 1 and func is None:
        func, func_kw = func_kw, dict()

    to_check = []
    if func is not None:
        if isinstance(func, list):
            to_check.extend(func)
        elif isinstance(func, dict):
            if ndim == 2:
                for f in func.values():
                    if isinstance(f, Iterable) and not isinstance(f, str):
                        to_check.extend(f)
                    else:
                        to_check.append(f)
            else:
                if any(isinstance(v, tuple) for v in func.values()):
                    raise TypeError('nested renamer is not supported')
                to_check.extend(func.values())
        else:
            to_check.append(func)
    else:
        for v in func_kw.values():
            if not isinstance(v, tuple) or len(v) != 2 \
                    or (not isinstance(v[1], str) and not callable(v[1])):
                raise TypeError("Must provide 'func' or tuples of (column, aggfunc).")
            else:
                to_check.append(v[1])

    compiler = ReductionCompiler()
    for f in to_check:
        if f in _agg_functions:
            continue
        elif callable(f):
            try:
                if ndim == 2:
                    compiler.add_function(f, 2, cols=['A', 'B'])
                else:
                    compiler.add_function(f, 1)
            except ValueError:
                return False
        else:
            return False
    return True


def normalize_reduction_funcs(op, ndim=None):
    raw_func = op.raw_func
    if ndim == 1 and raw_func is None:
        raw_func = op.raw_func_kw

    if raw_func is not None:
        if isinstance(raw_func, dict):
            if ndim == 2:
                new_func = OrderedDict()
                for k, v in raw_func.items():
                    if isinstance(v, str) or callable(v):
                        new_func[k] = [v]
                    else:
                        new_func[k] = v
                op._func = new_func
            else:
                op._func = list(raw_func.values())
                op._func_rename = list(raw_func.keys())
        elif isinstance(raw_func, Iterable) and not isinstance(raw_func, str):
            op._func = list(raw_func)
        else:
            op._func = [raw_func]
    else:
        new_func = OrderedDict()
        new_func_names = OrderedDict()
        for k, v in op.raw_func_kw.items():
            try:
                col_funcs = new_func[v[0]]
                col_func_names = new_func_names[v[0]]
            except KeyError:
                col_funcs = new_func[v[0]] = []
                col_func_names = new_func_names[v[0]] = []
            col_funcs.append(v[1])
            col_func_names.append(k)
        op._func = new_func
        op._func_rename = functools.reduce(lambda a, b: a + b, new_func_names.values(), [])

    custom_idx = 0
    if isinstance(op._func, list):
        custom_iter = (f for f in op._func if isinstance(f, CustomReduction))
    else:
        custom_iter = (f for f in op._func.values() if isinstance(f, CustomReduction))
    for r in custom_iter:
        if r.name == '<custom>':
            r.name = f'<custom_{custom_idx}>'
            custom_idx += 1


def aggregate(df, func=None, axis=0, **kw):
    axis = validate_axis(axis, df)
    use_inf_as_na = kw.pop('_use_inf_as_na', options.dataframe.mode.use_inf_as_na)
    if df.ndim == 2 and isinstance(func, dict) \
            and (df.op.output_types[0] == OutputType.series or axis == 1):
        raise NotImplementedError('Currently cannot aggregate dicts over axis=1 on %s'
                                  % type(df).__name__)
    combine_size = kw.pop('_combine_size', None) or options.combine_size
    numeric_only = kw.pop('_numeric_only', None)
    bool_only = kw.pop('_bool_only', None)

    output_type = kw.pop('_output_type', None)
    dtypes = kw.pop('_dtypes', None)
    index = kw.pop('_index', None)

    if not is_funcs_aggregate(func, func_kw=kw, ndim=df.ndim):
        return df.transform(func, axis=axis, _call_agg=True)

    op = DataFrameAggregate(raw_func=copy.deepcopy(func), raw_func_kw=copy.deepcopy(kw), axis=axis,
                            output_types=df.op.output_types, combine_size=combine_size,
                            numeric_only=numeric_only, bool_only=bool_only, use_inf_as_na=use_inf_as_na)

    return op(df, output_type=output_type, dtypes=dtypes, index=index)
