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
from collections import OrderedDict, namedtuple
from collections.abc import Iterable
from functools import partial

import cloudpickle
import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...config import options
from ...operands import OperandStage
from ...serialize import ValueType, BoolField, AnyField, StringField, BytesField, ListField
from ..merge import DataFrameConcat
from ..operands import DataFrameOperand, DataFrameOperandMixin, DataFrameShuffleProxy, ObjectType
from ..core import GROUPBY_TYPE
from ..utils import build_empty_df, parse_index, build_concated_rows_frame, tokenize
from .core import DataFrameGroupByOperand


_available_aggregation_functions = {'sum', 'prod', 'min', 'max', 'count',
                                    'mean', 'var', 'std'}

_stage_infos = namedtuple('stage_infos', ('intermediate_cols', 'agg_cols',
                                          'map_func', 'map_output_column_to_func',
                                          'combine_func', 'combine_output_column_to_func',
                                          'agg_func', 'agg_output_column_to_func'))


class DataFrameGroupByAgg(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY_AGG

    _func = AnyField('func')
    _raw_func = AnyField('raw_func')
    _by = AnyField('by')
    _as_index = BoolField('as_index')
    _sort = BoolField('sort')
    _method = StringField('method')
    # for chunk
    # store the intermediate aggregated columns for the result
    _agg_columns = ListField('agg_columns', ValueType.string)
    # store output columns -> function to apply on DataFrameGroupBy
    _output_column_to_func = BytesField('output_column_to_funcs',
                                        on_serialize=cloudpickle.dumps,
                                        on_deserialize=cloudpickle.loads)

    def __init__(self, func=None, by=None, as_index=None, sort=None, method=None,
                 raw_func=None, agg_columns=None, output_column_to_func=None, stage=None, **kw):
        super().__init__(_func=func, _by=by, _as_index=as_index, _sort=sort, _method=method,
                         _agg_columns=agg_columns, _output_column_to_func=output_column_to_func,
                         _raw_func=raw_func, _stage=stage, _object_type=ObjectType.dataframe, **kw)

    @property
    def func(self):
        return self._func

    @property
    def raw_func(self):
        return self._raw_func

    @property
    def by(self):
        return self._by

    @property
    def as_index(self):
        return self._as_index

    @property
    def sort(self):
        return self._sort

    @property
    def method(self):
        return self._method

    @property
    def agg_columns(self):
        return self._agg_columns

    @property
    def output_column_to_func(self):
        return self._output_column_to_func

    @property
    def stage(self):
        return self._stage

    def _normalize_keyword_aggregations(self, empty_df):
        raw_func = self._raw_func
        if isinstance(raw_func, dict):
            func = OrderedDict()
            for col, function in raw_func.items():
                if isinstance(function, Iterable) and not isinstance(function, str):
                    func[col] = list(function)
                else:
                    func[col] = [function]
            self._func = func
        else:
            # force as_index=True
            agg_df = empty_df.groupby(self.by, as_index=True).aggregate(self.func)

            if not isinstance(agg_df.columns, pd.MultiIndex):
                # 1 func, the columns of agg_df is the columns to aggregate
                self._func = OrderedDict((c, [raw_func]) for c in agg_df.columns)
            else:
                func = OrderedDict()
                for c, f in agg_df.columns:
                    self._safe_append(func, c, f)
                self._func = func

    def __call__(self, df):
        empty_df = build_empty_df(df.dtypes)
        agg_df = empty_df.groupby(self.by, as_index=self._as_index).aggregate(self.func)

        shape = (np.nan, agg_df.shape[1])
        index_value = parse_index(agg_df.index, df.key)
        index_value.value.should_be_monotonic = True

        # convert func to dict always
        self._normalize_keyword_aggregations(empty_df)

        # make sure if as_index=False takes effect
        if not self._as_index:
            if isinstance(agg_df.index, pd.MultiIndex):
                # if MultiIndex, as_index=False definitely takes no effect
                self._as_index = True
            elif agg_df.index.name is not None:
                # if not MultiIndex and agg_df.index has a name
                # means as_index=False takes no effect
                self._as_index = True

        return self.new_dataframe([df], shape=shape, dtypes=agg_df.dtypes,
                                  index_value=index_value,
                                  columns_value=parse_index(agg_df.columns, store_data=True))

    @staticmethod
    def _safe_append(d, key, val):
        if key not in d:
            d[key] = []
        d[key].append(val)

    @classmethod
    def _gen_stages_columns_and_funcs(cls, func):
        intermediate_cols = []
        intermediate_cols_set = set()
        agg_cols = []
        map_func = OrderedDict()
        map_output_column_to_func = dict()
        combine_func = OrderedDict()
        combine_output_column_to_func = dict()
        agg_func = OrderedDict()
        agg_output_column_to_func = dict()

        for col, functions in func.items():
            for f in functions:
                new_col = tokenize(col, f)
                agg_cols.append(new_col)
                if f in {'sum', 'prod', 'min', 'max'}:
                    if new_col not in intermediate_cols_set:
                        intermediate_cols.append(new_col)
                        intermediate_cols_set.add(new_col)

                        # function identical for all stages
                        cls._safe_append(map_func, col, f)
                        cls._safe_append(combine_func, new_col, f)

                    cls._safe_append(agg_func, new_col, f)
                elif f == 'count':
                    if new_col not in intermediate_cols_set:
                        intermediate_cols.append(new_col)
                        intermediate_cols_set.add(new_col)

                        # do count for map
                        cls._safe_append(map_func, col, f)
                        # do sum for combine and agg
                        cls._safe_append(combine_func, new_col, 'sum')
                    cls._safe_append(agg_func, new_col, 'sum')
                elif f in {'mean', 'var', 'std'}:
                    # handle special funcs that have intermediate results
                    sum_col = tokenize(col, 'sum')
                    if sum_col not in intermediate_cols_set:
                        intermediate_cols.append(sum_col)
                        intermediate_cols_set.add(sum_col)

                        cls._safe_append(map_func, col, 'sum')
                        cls._safe_append(combine_func, sum_col, 'sum')
                    count_col = tokenize(col, 'count')
                    if count_col not in intermediate_cols_set:
                        intermediate_cols.append(count_col)
                        intermediate_cols_set.add(count_col)

                        cls._safe_append(map_func, col, 'count')
                        cls._safe_append(combine_func, count_col, 'sum')
                    if f == 'mean':
                        def _mean(df, columns):
                            return df[columns[0]].sum() / df[columns[1]].sum()

                        cls._safe_append(agg_func, new_col, None)
                        agg_output_column_to_func[new_col] = \
                            partial(_mean, columns=[sum_col, count_col])
                    else:  # var, std
                        # calculate var for map
                        var_col = tokenize(col, 'var')

                        def _reduce_var(df, columns):
                            cnt = df[columns[1]]
                            reduced_cnt = cnt.sum()
                            data = df[columns[0]]
                            var_square = df[columns[2]] * (cnt - 1)
                            avg = data.sum() / reduced_cnt
                            avg_diff = data / cnt - avg
                            var_square = (var_square.sum() + (cnt * avg_diff ** 2).sum())
                            return var_square / (reduced_cnt - 1)

                        def _reduce_std(df, columns):
                            return np.sqrt(_reduce_var(df, columns))

                        if var_col not in intermediate_cols_set:
                            intermediate_cols.append(var_col)
                            intermediate_cols_set.add(var_col)

                            cls._safe_append(map_func, col, 'var')
                            cls._safe_append(combine_func, var_col, None)
                            combine_output_column_to_func[var_col] = \
                                partial(_reduce_var, columns=[sum_col, count_col, var_col])
                        cls._safe_append(agg_func, new_col, None)
                        fun = _reduce_var if f == 'var' else _reduce_std
                        agg_output_column_to_func[new_col] = \
                           partial(fun, columns=[sum_col, count_col, var_col])
                else:  # pragma: no cover
                    raise NotImplementedError

        return _stage_infos(intermediate_cols=intermediate_cols, agg_cols=agg_cols,
                            map_func=map_func,
                            map_output_column_to_func=map_output_column_to_func,
                            combine_func=combine_func,
                            combine_output_column_to_func=combine_output_column_to_func,
                            agg_func=agg_func,
                            agg_output_column_to_func=agg_output_column_to_func)

    @classmethod
    def _gen_shuffle_chunks(cls, op, in_df, chunks):
        # generate map chunks
        map_chunks = []
        chunk_shape = (in_df.chunk_shape[0], 1)
        for chunk in chunks:
            # no longer consider as_index=False for the intermediate phases,
            # will do reset_index at last if so
            map_op = DataFrameGroupByOperand(stage=OperandStage.map, shuffle_size=chunk_shape[0])
            map_chunks.append(map_op.new_chunk([chunk], shape=(np.nan, np.nan), index=chunk.index,
                                               index_value=op.outputs[0].index_value))

        proxy_chunk = DataFrameShuffleProxy(object_type=ObjectType.dataframe).new_chunk(map_chunks, shape=())

        # generate reduce chunks
        reduce_chunks = []
        for out_idx in itertools.product(*(range(s) for s in chunk_shape)):
            reduce_op = DataFrameGroupByOperand(
                stage=OperandStage.reduce, shuffle_key=','.join(str(idx) for idx in out_idx))
            reduce_chunks.append(
                reduce_op.new_chunk([proxy_chunk], shape=(np.nan, np.nan), index=out_idx,
                                    index_value=op.outputs[0].index_value))

        return reduce_chunks

    @classmethod
    def _gen_map_chunks(cls, op, in_df, out_df, stage_infos: _stage_infos):
        agg_chunks = []
        for chunk in in_df.chunks:
            agg_op = op.copy().reset_key()
            # force as_index=True for map phase
            agg_op._as_index = True
            agg_op._stage = OperandStage.map
            agg_op._func = stage_infos.map_func
            agg_op._output_column_to_func = stage_infos.map_output_column_to_func
            columns_value = parse_index(pd.Index(stage_infos.intermediate_cols), store_data=True)
            agg_chunk = agg_op.new_chunk([chunk], shape=out_df.shape, index=chunk.index,
                                         index_value=out_df.index_value,
                                         columns_value=columns_value)
            agg_chunks.append(agg_chunk)
        return agg_chunks

    @classmethod
    def _tile_with_shuffle(cls, op):
        in_df = build_concated_rows_frame(op.inputs[0])
        out_df = op.outputs[0]

        stage_infos = cls._gen_stages_columns_and_funcs(op.func)

        # First, perform groupby and aggregation on each chunk.
        agg_chunks = cls._gen_map_chunks(op, in_df, out_df, stage_infos)

        # Shuffle the aggregation chunk.
        reduce_chunks = cls._gen_shuffle_chunks(op, in_df, agg_chunks)

        # Combine groups
        agg_chunks = []
        for chunk in reduce_chunks:
            agg_op = op.copy().reset_key()
            agg_op._stage = OperandStage.agg
            agg_op._func = stage_infos.agg_func
            agg_op._output_column_to_func = stage_infos.agg_output_column_to_func
            agg_op._agg_columns = stage_infos.agg_cols
            agg_chunk = agg_op.new_chunk([chunk], shape=out_df.shape, index=chunk.index,
                                         index_value=out_df.index_value,
                                         columns_value=out_df.columns_value)
            agg_chunks.append(agg_chunk)

        new_op = op.copy()
        return new_op.new_dataframes([in_df], shape=out_df.shape, index_value=out_df.index_value,
                                     columns_value=out_df.columns_value, chunks=agg_chunks,
                                     nsplits=((np.nan,) * len(agg_chunks), (out_df.shape[1],)))

    @classmethod
    def _tile_with_tree(cls, op):
        in_df = build_concated_rows_frame(op.inputs[0])
        out_df = op.outputs[0]

        stage_infos = cls._gen_stages_columns_and_funcs(op.func)
        combine_size = options.combine_size
        chunks = cls._gen_map_chunks(op, in_df, out_df, stage_infos)
        while len(chunks) > combine_size:
            new_chunks = []
            for idx, i in enumerate(range(0, len(chunks), combine_size)):
                chks = chunks[i: i + combine_size]
                if len(chks) == 1:
                    chk = chks[0]
                else:
                    concat_op = DataFrameConcat(object_type=ObjectType.dataframe)
                    # Change index for concatenate
                    for j, c in enumerate(chks):
                        c._index = (j, 0)
                    chk = concat_op.new_chunk(chks, dtypes=chks[0].dtypes)
                chunk_op = op.copy().reset_key()
                chunk_op._stage = OperandStage.combine
                chunk_op._func = stage_infos.combine_func
                chunk_op._output_column_to_func = stage_infos.combine_output_column_to_func
                columns_value = parse_index(pd.Index(stage_infos.intermediate_cols), store_data=True)
                new_chunks.append(chunk_op.new_chunk([chk], index=(idx, 0), shape=(np.nan, out_df.shape[1]),
                                                     index_value=chks[0].index_value,
                                                     columns_value=columns_value))
            chunks = new_chunks

        concat_op = DataFrameConcat(object_type=ObjectType.dataframe)
        chk = concat_op.new_chunk(chunks, dtypes=chunks[0].dtypes)
        chunk_op = op.copy().reset_key()
        chunk_op._stage = OperandStage.agg
        chunk_op._func = stage_infos.agg_func
        chunk_op._output_column_to_func = stage_infos.agg_output_column_to_func
        chunk_op._agg_columns = stage_infos.agg_cols
        chunk = chunk_op.new_chunk([chk], index=(0, 0), shape=out_df.shape, index_value=out_df.index_value,
                                   columns_value=out_df.columns_value, dtypes=out_df.dtypes)
        new_op = op.copy()
        nsplits = ((out_df.shape[0],), (out_df.shape[1],))
        return new_op.new_tileables(op.inputs, chunks=[chunk], nsplits=nsplits,
                                    dtypes=out_df.dtypes, shape=out_df.shape,
                                    index_value=out_df.index_value,
                                    columns_value=out_df.columns_value)

    @classmethod
    def tile(cls, op):
        if op.method == 'shuffle':
            return cls._tile_with_shuffle(op)
        elif op.method == 'tree':
            return cls._tile_with_tree(op)
        else:  # pragma: no cover
            raise NotImplementedError

    @classmethod
    def _get_grouped(cls, op, df, copy=False):
        if copy:
            df = df.copy()
        if op.stage == OperandStage.agg:
            return df.groupby(op.by, sort=op.sort)
        else:
            # for the intermediate phases, do not sort
            return df.groupby(op.by, sort=False)

    @classmethod
    def _is_raw_one_func(cls, op):
        raw_func = op.raw_func
        func = None
        if isinstance(raw_func, str):
            func = raw_func
        elif isinstance(raw_func, list) and len(raw_func) == 1:
            func = raw_func[0]
        if func is None:
            return False
        return func in {'min', 'max', 'prod', 'sum', 'count'}

    @classmethod
    def execute(cls, ctx, op):
        df = ctx[op.inputs[0].key]
        out = op.outputs[0]

        grouped = cls._get_grouped(op, df)
        if op.stage == OperandStage.agg:
            # use intermediate columns
            columns = op.agg_columns
        else:
            columns = op.outputs[0].columns_value.to_pandas().tolist()

        func = OrderedDict()
        processed_cols = []
        col_iter = iter(columns)
        for col, funcs in op.func.items():
            for f in funcs:
                c = next(col_iter)
                if f is not None:
                    cls._safe_append(func, col, f)
                    processed_cols.append(c)

        if cls._is_raw_one_func(op):
            # do some optimization if the raw func is a str or list whose length is 1
            func = next(iter(func.values()))[0]
            result = grouped.agg(func)
            if result.empty and not df.empty:
                # fix for py35, due to buffer read-only
                grouped = cls._get_grouped(op, df, copy=True)
                result = grouped.agg(func)
        else:
            # agg the funcs that can be done
            try:
                result = grouped.agg(func)
            except ValueError:  # pragma: no cover
                # fail due to buffer read-only
                # force to get grouped again by copy
                grouped = cls._get_grouped(op, df, copy=True)
                result = grouped.agg(func)
        result.columns = processed_cols

        if len(op.output_column_to_func) > 0:
            # process the functions that require operating on the grouped data
            for out_col, f in op.output_column_to_func.items():
                if len(df) > 0:
                    result[out_col] = grouped.apply(f)
                else:
                    result[out_col] = []

            # sort columns as origin
            result = result[columns]

        if len(result) == 0:
            # empty data, set index manually
            result.index = out.index_value.to_pandas()

        if op.stage == OperandStage.agg:
            if not op.as_index:
                result.reset_index(inplace=True)
            result.columns = out.columns_value.to_pandas()

        ctx[out.key] = result


def _check_if_func_available(func):
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
        if f not in _available_aggregation_functions:
            raise NotImplementedError(
                'Aggregation function {} has not been implemented, '
                'available functions include: {}'.format(
                    f, _available_aggregation_functions))


def agg(groupby, func, method='tree'):
    """
    Aggregate using one or more operations on grouped data.
    :param groupby: Groupby data.
    :param func: Aggregation functions.
    :param method: 'shuffle' or 'tree', 'tree' method provide a better performance, 'shuffle' is recommended
    if aggregated result is very large.
    :return: Aggregated result.
    """

    # When perform a computation on the grouped data, we won't shuffle
    # the data in the stage of groupby and do shuffle after aggregation.
    if not isinstance(groupby, GROUPBY_TYPE):
        raise TypeError('Input should be type of groupby, not %s' % type(groupby))

    if method not in ['shuffle', 'tree']:
        raise ValueError("Method %s is not available, "
                         "please specify 'tree' or 'shuffle" % method)

    _check_if_func_available(func)

    in_df = groupby.inputs[0]
    agg_op = DataFrameGroupByAgg(func=func, by=groupby.op.by, method=method, raw_func=func,
                                 as_index=groupby.op.as_index, sort=groupby.op.sort)
    return agg_op(in_df)
