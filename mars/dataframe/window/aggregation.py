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
from typing import Dict

import numpy as np
import pandas as pd

from ...operands import OperandStage
from ...serialize import AnyField, BoolField, Int32Field, Int64Field, \
    DictField, StringField
from ...utils import tokenize
from ..core import DATAFRAME_TYPE
from ..merge import DataFrameConcat
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import build_empty_df, parse_index, build_empty_series

_stage_info = namedtuple('_stage_info', ('map_groups', 'map_sources', 'combine_sources',
                                         'combine_columns', 'combine_funcs', 'key_to_funcs',
                                         'valid_columns', 'min_periods_func_name'))


class BaseDataFrameExpandingAgg(DataFrameOperand, DataFrameOperandMixin):
    _min_periods = Int64Field('min_periods')
    _axis = Int32Field('axis')
    _func = AnyField('func')

    # always treat count as valid. this behavior is cancelled in pandas 1.0
    _count_always_valid = BoolField('count_always_valid')
    # True if function name is treated as new index
    _append_index = BoolField('append_index')

    # chunk params
    _output_agg = BoolField('output_agg')

    _map_groups = DictField('map_groups')
    _map_sources = DictField('map_sources')
    _combine_sources = DictField('combine_sources')
    _combine_columns = DictField('combine_columns')
    _combine_funcs = DictField('combine_funcs')
    _key_to_funcs = DictField('keys_to_funcs')

    _min_periods_func_name = StringField('min_periods_func_name')

    def __init__(self, min_periods=None, axis=None, func=None, count_always_valid=None,
                 append_index=None, output_agg=False, map_groups=None, map_sources=None,
                 combine_sources=None, combine_columns=None, combine_funcs=None,
                 key_to_funcs=None, min_periods_func_name=None, stage=None, **kw):
        super().__init__(_min_periods=min_periods, _axis=axis, _func=func, _stage=stage,
                         _count_always_valid=count_always_valid, _append_index=append_index,
                         _output_agg=output_agg, _map_groups=map_groups, _map_sources=map_sources,
                         _combine_sources=combine_sources, _combine_columns=combine_columns,
                         _combine_funcs=combine_funcs, _key_to_funcs=key_to_funcs,
                         _min_periods_func_name=min_periods_func_name, **kw)

    @property
    def min_periods(self) -> int:
        return self._min_periods

    @property
    def axis(self) -> int:
        return self._axis

    @property
    def func(self):
        return self._func

    @property
    def count_always_valid(self):
        return self._count_always_valid

    @property
    def append_index(self):
        return self._append_index

    @property
    def output_agg(self):
        return self._output_agg

    @property
    def map_groups(self) -> Dict:
        return self._map_groups

    @property
    def map_sources(self) -> Dict:
        return self._map_sources

    @property
    def combine_sources(self) -> Dict:
        return self._combine_sources

    @property
    def combine_columns(self) -> Dict:
        return self._combine_columns

    @property
    def combine_funcs(self) -> Dict:
        return self._combine_funcs

    @property
    def key_to_funcs(self) -> Dict:
        return self._key_to_funcs

    @property
    def min_periods_func_name(self) -> str:
        return self._min_periods_func_name

    @property
    def output_limit(self):
        return 2 if self.output_agg else 1

    def __call__(self, expanding):
        inp = expanding.input
        raw_func = self.func
        self._normalize_funcs()

        if isinstance(inp, DATAFRAME_TYPE):
            pd_index = inp.index_value.to_pandas()

            empty_df = build_empty_df(inp.dtypes, index=pd_index[:1])
            for c, t in empty_df.dtypes.items():
                if t == np.dtype('O'):
                    empty_df[c] = 'O'

            test_df = expanding(empty_df).agg(raw_func)
            if self._axis == 0:
                index_value = inp.index_value
            else:
                index_value = parse_index(test_df.index,
                                          expanding.params, inp,
                                          store_data=False)
            self._object_type = ObjectType.dataframe
            self._append_index = test_df.columns.nlevels != empty_df.columns.nlevels
            return self.new_dataframe(
                [inp], shape=(inp.shape[0], test_df.shape[1]),
                dtypes=test_df.dtypes, index_value=index_value,
                columns_value=parse_index(test_df.columns, store_data=True))
        else:
            pd_index = inp.index_value.to_pandas()
            empty_series = build_empty_series(inp.dtype, index=pd_index[:0], name=inp.name)
            test_obj = expanding(empty_series).agg(raw_func)
            if isinstance(test_obj, pd.DataFrame):
                self._object_type = ObjectType.dataframe
                return self.new_dataframe([inp], shape=(inp.shape[0], test_obj.shape[1]),
                                          dtypes=test_obj.dtypes, index_value=inp.index_value,
                                          columns_value=parse_index(test_obj.dtypes.index, store_data=True))
            else:
                self._object_type = ObjectType.series
                return self.new_series([inp], shape=inp.shape, dtype=test_obj.dtype,
                                       index_value=inp.index_value, name=test_obj.name)

    def _normalize_funcs(self):
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

    @staticmethod
    def _safe_append(d, key, val):
        if key not in d:
            d[key] = []
        if val not in d[key]:
            d[key].append(val)

    @classmethod
    def _get_stage_functions(cls, op: "BaseDataFrameExpandingAgg", func):
        raise NotImplementedError

    @classmethod
    def _gen_chunk_stage_info(cls, op: "BaseDataFrameExpandingAgg", chunk_cols=None, min_periods=1):
        map_groups = OrderedDict()
        map_sources = OrderedDict()
        combine_sources = OrderedDict()
        combine_columns = OrderedDict()
        combine_funcs = OrderedDict()
        key_to_funcs = OrderedDict()
        valid_columns = []
        min_periods_func_name = None

        def _clean_dict(d):
            return OrderedDict((k, sorted(v) if v != [None] else None) for k, v in d.items())

        def _fun_to_str(fun):
            if isinstance(fun, str):
                return fun
            fun_str = tokenize(fun)
            key_to_funcs[fun_str] = fun
            return fun if isinstance(fun, str) else tokenize(fun)

        def _add_column_to_functions(col, fun_name, mappers, aggregator):
            sources = []
            for mapper in mappers:
                mapper_str = _fun_to_str(mapper)
                cls._safe_append(map_groups, mapper_str, col)
                sources.append(mapper_str)

            combine_sources[fun_name] = sources
            cls._safe_append(combine_columns, fun_name, col)
            combine_funcs[fun_name] = _fun_to_str(aggregator)

        chunk_cols = set(chunk_cols) if chunk_cols is not None else None
        if isinstance(op.func, list):
            op_func = {None: op.func}
        else:
            op_func = op.func

        for col, funcs in op_func.items():
            if col is not None:
                if chunk_cols is not None and col not in chunk_cols:
                    continue
                valid_columns.append(col)

            if min_periods > 1:
                min_periods_func_name = tokenize(chunk_cols, 'min_periods')
                _add_column_to_functions(col, min_periods_func_name,
                                         *cls._get_stage_functions(op, '_data_count'))

            for func in funcs:
                mapper_funcs, combine_func = cls._get_stage_functions(op, func)
                _add_column_to_functions(col, func, mapper_funcs, combine_func)

        return _stage_info(map_groups=_clean_dict(map_groups), map_sources=map_sources,
                           combine_sources=combine_sources, combine_columns=_clean_dict(combine_columns),
                           combine_funcs=combine_funcs, key_to_funcs=key_to_funcs,
                           valid_columns=valid_columns or None, min_periods_func_name=min_periods_func_name)

    @classmethod
    def _remap_dtypes(cls, in_df, out_df):
        if in_df.ndim == 1:
            if out_df.ndim == 2:
                return {0: (0, out_df.dtypes)}, (in_df.nsplits[0], (len(out_df.dtypes),))
            return None, in_df.nsplits

        axis = out_df.op.axis
        chunk_idx_to_dtypes = dict()
        out_dtypes = out_df.dtypes
        new_dtypes_sizes = []
        for c in in_df.cix[0, :]:
            columns = c.columns_value.to_pandas()
            try:
                dtypes = out_dtypes.loc[columns].dropna()
            except KeyError:
                dtypes = out_dtypes.reindex(columns).dropna()

            if len(dtypes):
                chunk_idx_to_dtypes[c.index[1]] = (len(chunk_idx_to_dtypes), dtypes)
                new_dtypes_sizes.append(len(dtypes))
        new_nsplits = list(in_df.nsplits)
        new_nsplits[1 - axis] = tuple(new_dtypes_sizes)
        return chunk_idx_to_dtypes, tuple(new_nsplits)

    @classmethod
    def _tile_single(cls, op: "BaseDataFrameExpandingAgg"):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        chunk_idx_to_dtypes, new_nsplits = cls._remap_dtypes(in_df, out_df)

        chunks = []
        for c in in_df.chunks:
            try:
                if out_df.ndim == 2:
                    new_axis_idx, new_dtypes = chunk_idx_to_dtypes[c.index[1] if c.ndim > 1 else 0]
                else:
                    new_axis_idx, new_dtypes = None, None
            except KeyError:
                continue

            chunk_op = op.copy().reset_key()

            if out_df.ndim == 2:
                chunks.append(chunk_op.new_chunk(
                    [in_df.chunks[0]], dtypes=new_dtypes, index=(c.index[0], new_axis_idx),
                    shape=(c.shape[0], len(new_dtypes)), index_value=c.index_value,
                    columns_value=parse_index(new_dtypes.index, store_data=True)))
            else:
                params = c.params.copy()
                params['dtype'] = out_df.dtype
                chunks.append(chunk_op.new_chunk([in_df.chunks[0]], **params))

        tileable_op = op.copy().reset_key()
        params = out_df.params.copy()
        params['chunks'] = chunks
        if new_nsplits:
            params['nsplits'] = new_nsplits
        return tileable_op.new_tileables([in_df], **params)

    @classmethod
    def tile(cls, op: "BaseDataFrameExpandingAgg"):
        axis = op.axis

        in_df = op.inputs[0]
        out_df = op.outputs[0]

        if in_df.chunk_shape[op.axis] == 1:
            return cls._tile_single(op)

        dtypes_mapping, new_nsplits = cls._remap_dtypes(in_df, out_df)
        new_chunk_shape = tuple(len(split) for split in new_nsplits)

        data_chunks = []
        summary_chunks = np.empty(new_chunk_shape, dtype=np.object)
        stage_info_dict = dict()
        for c in in_df.chunks:
            try:
                if out_df.ndim == 2:
                    new_axis_idx, new_dtypes = dtypes_mapping[c.index[1] if c.ndim > 1 else 0]
                else:
                    new_axis_idx, new_dtypes = None, None
            except KeyError:
                continue

            new_index = (c.index[0], new_axis_idx)

            try:
                stage_info = stage_info_dict[new_index[1]]
            except KeyError:
                cols = c.dtypes.index if c.ndim == 2 else None
                stage_info = stage_info_dict[new_index[1]] \
                    = cls._gen_chunk_stage_info(op, cols, min_periods=op.min_periods)

            chunk_op = op.copy().reset_key()
            chunk_op._output_agg = c.index[axis] != in_df.chunk_shape[axis] - 1
            chunk_op._stage = OperandStage.map
            chunk_op._map_sources = stage_info.map_sources
            chunk_op._map_groups = stage_info.map_groups
            chunk_op._key_to_funcs = stage_info.key_to_funcs

            if out_df.ndim == 2:
                kw0 = dict(dtypes=new_dtypes, index=new_index,
                           shape=(c.shape[0], len(new_dtypes)),
                           index_value=c.index_value,
                           columns_value=parse_index(new_dtypes.index, store_data=True))
                kw1 = kw0.copy()
                kw1['shape'] = (1, len(new_dtypes)) if axis == 0 else (c.shape[0], 1)
            else:
                kw0 = dict(dtype=out_df.dtype, index=c.index, shape=c.shape,
                           name=c.name, index_value=c.index_value)
                kw1 = kw0.copy()
                kw1['shape'] = (1,)
            out_chunks = chunk_op.new_chunks([c], [kw0, kw1])
            data_chunks.append(out_chunks[0])
            if chunk_op.output_agg:
                summary_chunks[new_index] = out_chunks[1]

        chunks = []
        for c in data_chunks:
            stage_info = stage_info_dict[c.index[1] if c.ndim > 1 else None]

            chunk_op = op.copy().reset_key()
            chunk_op._output_agg = False
            chunk_op._stage = OperandStage.combine
            chunk_op._map_groups = stage_info.map_groups
            chunk_op._combine_sources = stage_info.combine_sources
            chunk_op._combine_columns = stage_info.combine_columns
            chunk_op._combine_funcs = stage_info.combine_funcs
            chunk_op._key_to_funcs = stage_info.key_to_funcs
            chunk_op._min_periods_func_name = stage_info.min_periods_func_name

            params = c.params.copy()
            if c.ndim == 2:
                summary_inputs = list(summary_chunks[:c.index[0], c.index[1]])
            else:
                summary_inputs = list(summary_chunks[:c.index[0]])

            if len(summary_inputs) > 1:
                concat_op = DataFrameConcat(object_type=out_df.op.object_type, axis=op.axis)
                concat_summary = concat_op.new_chunk(summary_inputs)
                chunks.append(chunk_op.new_chunk([c, concat_summary], **params))
            elif len(summary_inputs) == 1:
                chunks.append(chunk_op.new_chunk([c, summary_inputs[0]], **params))
            else:
                chunks.append(chunk_op.new_chunk([c], **params))

        df_op = op.copy().reset_key()
        params = out_df.params.copy()
        params.update(dict(chunks=chunks, nsplits=new_nsplits))
        return df_op.new_tileables([in_df], **params)

    @classmethod
    def _execute_map_function(cls, op: "BaseDataFrameExpandingAgg", func, in_data):
        raise NotImplementedError

    @classmethod
    def _execute_map(cls, ctx, op: "BaseDataFrameExpandingAgg"):
        in_data = ctx[op.inputs[0].key]

        # map according to map groups
        map_results = []
        summary_results = []
        for map_func_str, cols in op.map_groups.items():
            if cols is None:
                src_df = in_data
            else:
                src_df = in_data[cols]

            result, summary = cls._execute_map_function(op, map_func_str, src_df)
            map_results.append(result)
            if op.output_agg:
                summary_results.append(summary)

        if op.output_agg:
            summary_results.append(pd.Series([len(in_data)], index=summary_results[0].index))

        ctx[op.outputs[0].key] = tuple(map_results)
        if op.output_agg:
            ctx[op.outputs[1].key] = tuple(summary_results)

    @classmethod
    def _append_func_name_index(cls, op: "BaseDataFrameExpandingAgg", df, func_name):
        if not op.append_index:
            return

        col_frame = df.columns.to_frame().copy()
        col_frame[len(col_frame.columns)] = func_name
        df.columns = pd.MultiIndex.from_frame(
            col_frame, names=tuple(df.columns.names) + (None,))

    @classmethod
    def _execute_combine_function(cls, op: "BaseDataFrameExpandingAgg", func,
                                  pred_inputs, local_inputs, func_cols):
        raise NotImplementedError

    @classmethod
    def _execute_combine(cls, ctx, op: "BaseDataFrameExpandingAgg"):
        out_df = op.outputs[0]
        local_data = ctx[op.inputs[0].key]
        local_data_dict = dict(zip(op.map_groups.keys(), local_data))

        func_to_aggs = OrderedDict()

        if len(op.inputs) == 1:
            pred_record_count = 0
            for func_name, func_sources in op.combine_sources.items():
                func_str = op.combine_funcs[func_name]
                func_cols = op.combine_columns[func_name]
                if func_cols is None:
                    local_inputs = [local_data_dict[src] for src in func_sources]
                else:
                    local_inputs = [local_data_dict[src][func_cols] for src in func_sources]

                func = op.key_to_funcs[func_str]
                func_to_aggs[func_name] = cls._execute_combine_function(
                    op, func, None, local_inputs, func_cols)
        else:
            pred_data = ctx[op.inputs[1].key]
            pred_record_count = pred_data[-1].sum()
            pred_data_dict = dict(zip(op.map_groups.keys(), pred_data))

            for func_name, func_sources in op.combine_sources.items():
                func_str = op.combine_funcs[func_name]
                func_cols = op.combine_columns[func_name]
                if func_cols is None:
                    local_inputs = [local_data_dict[src] for src in func_sources]
                    pred_inputs = [pred_data_dict[src] for src in func_sources]
                else:
                    local_inputs = [local_data_dict[src][func_cols] for src in func_sources]
                    pred_inputs = [pred_data_dict[src][func_cols] for src in func_sources]

                func = op.key_to_funcs[func_str]
                func_to_aggs[func_name] = cls._execute_combine_function(
                    op, func, pred_inputs, local_inputs, func_cols)

        if op.min_periods_func_name is not None:
            valid_counts = func_to_aggs.pop(op.min_periods_func_name)
            invalid_poses = valid_counts < op.min_periods
            for func_name in func_to_aggs.keys():
                if func_name == 'count':
                    if not op.count_always_valid and pred_record_count < op.min_periods - 1:
                        func_to_aggs[func_name].iloc[:op.min_periods - pred_record_count - 1] = np.nan
                else:
                    func_to_aggs[func_name][invalid_poses] = np.nan

        for func_name, agg_df in func_to_aggs.items():
            if out_df.ndim == 2 and agg_df.ndim == 1:
                agg_df.name = func_name
                agg_df = func_to_aggs[func_name] = pd.DataFrame(agg_df)
            cls._append_func_name_index(op, agg_df, func_name)

        if len(func_to_aggs) == 1:
            val = list(func_to_aggs.values())[0]
        else:
            out_df = op.outputs[0]
            val = pd.concat(list(func_to_aggs.values()), axis=1 - op.axis)

        if out_df.ndim > 1:
            val = val.reindex(out_df.columns_value.to_pandas(), axis=1 - op.axis, copy=False)
        else:
            val.name = out_df.name
        ctx[op.outputs[0].key] = val

    @classmethod
    def _execute_raw_function(cls, op: "BaseDataFrameExpandingAgg", in_data):
        raise NotImplementedError

    @classmethod
    def execute(cls, ctx, op: "BaseDataFrameExpandingAgg"):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        elif op.stage == OperandStage.combine:
            cls._execute_combine(ctx, op)
        else:
            in_data = ctx[op.inputs[0].key]
            out_df = op.outputs[0]

            r = cls._execute_raw_function(op, in_data)
            if out_df.ndim == 1:
                r = r.iloc[:, 0]
                r.name = out_df.name
            ctx[op.outputs[0].key] = r
