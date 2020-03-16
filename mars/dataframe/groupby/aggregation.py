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

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...config import options
from ...operands import OperandStage
from ...serialize import ValueType, BoolField, AnyField, StringField, \
    ListField, DictField
from ..merge import DataFrameConcat
from ..operands import DataFrameOperand, DataFrameOperandMixin, \
    DataFrameShuffleProxy, ObjectType
from ..core import GROUPBY_TYPE
from ..utils import build_empty_df, build_empty_series, parse_index, \
    build_concatenated_rows_frame, tokenize
from .core import DataFrameGroupByOperand


_available_aggregation_functions = {'sum', 'prod', 'min', 'max', 'count',
                                    'mean', 'var', 'std'}

_stage_infos = namedtuple('stage_infos', ('intermediate_cols', 'agg_cols',
                                          'map_func', 'map_output_column_to_func',
                                          'combine_func', 'combine_output_column_to_func',
                                          'agg_func', 'agg_output_column_to_func'))

_series_col_name = 'col_name'


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
    _output_column_to_func = DictField('output_column_to_func')
    # store original name of the series
    _series_name = StringField('series_name')

    def __init__(self, func=None, by=None, as_index=None, sort=None, method=None, raw_func=None,
                 agg_columns=None, output_column_to_func=None, series_name=None, stage=None,
                 object_type=None, **kw):
        super().__init__(_func=func, _by=by, _as_index=as_index, _sort=sort, _method=method,
                         _agg_columns=agg_columns, _output_column_to_func=output_column_to_func,
                         _raw_func=raw_func, _series_name=series_name, _stage=stage,
                         _object_type=object_type, **kw)

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
    def series_name(self):
        return self._series_name

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

            if isinstance(agg_df, pd.Series):
                self._func = OrderedDict([(_series_col_name, [raw_func])])
            else:
                if isinstance(empty_df, pd.Series):
                    func = OrderedDict()
                    for f in agg_df.columns:
                        self._safe_append(func, _series_col_name, f)
                    self._func = func
                elif not isinstance(agg_df.columns, pd.MultiIndex):
                    # 1 func, the columns of agg_df is the columns to aggregate
                    self._func = OrderedDict((c, [raw_func]) for c in agg_df.columns)
                else:
                    func = OrderedDict()
                    for c, f in agg_df.columns:
                        self._safe_append(func, c, f)
                    self._func = func

    def _call_dataframe(self, df):
        empty_df = build_empty_df(df.dtypes)
        agg_df = empty_df.groupby(self.by, as_index=self.as_index).aggregate(self.func)

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

    def _call_series(self, series):
        empty_series = build_empty_series(series.dtype)
        agg_result = empty_series.groupby(self.by, as_index=self.as_index).aggregate(self.func)

        index_value = parse_index(agg_result.index, series.key)
        index_value.value.should_be_monotonic = True

        # convert func to dict always
        self._normalize_keyword_aggregations(empty_series)

        self._series_name = series.name
        # update value type
        if isinstance(agg_result, pd.DataFrame):
            self._object_type = ObjectType.dataframe
            return self.new_dataframe([series], shape=(np.nan, len(agg_result.columns)),
                                      dtypes=agg_result.dtypes, index_value=index_value,
                                      columns_value=parse_index(agg_result.columns, store_data=True))
        else:
            self._object_type = ObjectType.series
            return self.new_series([series], shape=(np.nan,), dtype=agg_result.dtype,
                                   index_value=index_value)

    def __call__(self, df):
        if df.op.object_type == ObjectType.dataframe:
            return self._call_dataframe(df)
        else:
            return self._call_series(df)

    @staticmethod
    def _safe_append(d, key, val):
        if key not in d:
            d[key] = []
        d[key].append(val)

    @classmethod
    def _append_func(cls, func_dict, callable_func_dict, col, func, src_cols):
        if callable(func):
            callable_func_dict[col] = partial(func, columns=list(src_cols))
            func = None
        cls._safe_append(func_dict, col, func)

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

        def _add_column_to_functions(col, fun_name, mappers, combiners, aggregator):
            final_col = tokenize(col, fun_name)
            agg_cols.append(final_col)
            mapper_to_cols = OrderedDict([(mapper, tokenize(col, mapper)) for mapper in mappers])
            for mapper, combiner in zip(mappers, combiners):
                mapper_col = mapper_to_cols[mapper]
                if mapper_col not in intermediate_cols_set:
                    intermediate_cols.append(mapper_col)
                    intermediate_cols_set.add(mapper_col)

                    cls._append_func(map_func, map_output_column_to_func,
                                     col, mapper, (mapper_col,))
                    cls._append_func(combine_func, combine_output_column_to_func,
                                     mapper_col, combiner, mapper_to_cols.values())

            cls._append_func(agg_func, agg_output_column_to_func,
                             final_col, aggregator, mapper_to_cols.values())

        for col, functions in func.items():
            for f in functions:
                if f in {'sum', 'prod', 'min', 'max'}:
                    _add_column_to_functions(col, f, [f], [f], f)
                elif f == 'count':
                    _add_column_to_functions(col, f, [f], ['sum'], 'sum')
                elif f == 'mean':
                    def _mean(df, columns):
                        return df[columns[0]].sum() / df[columns[1]].sum()

                    _add_column_to_functions(col, f, ['sum', 'count'], ['sum', 'sum'], _mean)
                elif f in {'var', 'std'}:
                    def _reduce_var(df, columns):
                        cnt = df[columns[1]]
                        reduced_cnt = cnt.sum()
                        data = df[columns[0]]
                        var_square = df[columns[2]] * (cnt - 1)
                        avg = data.sum() / reduced_cnt
                        avg_diff = data / cnt - avg
                        var_square = var_square.sum() + (cnt * avg_diff ** 2).sum()
                        return var_square / (reduced_cnt - 1)

                    def _reduce_std(df, columns):
                        return np.sqrt(_reduce_var(df, columns))

                    _add_column_to_functions(col, f, ['sum', 'count', 'var'],
                                             ['sum', 'sum', _reduce_var],
                                             _reduce_var if f == 'var' else _reduce_std)
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
            map_op = DataFrameGroupByOperand(stage=OperandStage.map, shuffle_size=chunk_shape[0],
                                             object_type=ObjectType.dataframe_groupby)
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
            agg_op._object_type = ObjectType.dataframe
            agg_op._as_index = True
            agg_op._stage = OperandStage.map
            agg_op._func = stage_infos.map_func
            agg_op._output_column_to_func = stage_infos.map_output_column_to_func
            columns_value = parse_index(pd.Index(stage_infos.intermediate_cols), store_data=True)
            new_index = chunk.index if len(chunk.index) == 2 else (chunk.index[0], 0)
            if op.object_type == ObjectType.dataframe:
                agg_chunk = agg_op.new_chunk([chunk], shape=out_df.shape, index=new_index,
                                             index_value=out_df.index_value,
                                             columns_value=columns_value)
            else:
                agg_chunk = agg_op.new_chunk([chunk], shape=(out_df.shape[0], 1), index=new_index,
                                             index_value=out_df.index_value, columns_value=columns_value)
            agg_chunks.append(agg_chunk)
        return agg_chunks

    @classmethod
    def _tile_with_shuffle(cls, op):
        in_df = op.inputs[0]
        if len(in_df.shape) > 1:
            in_df = build_concatenated_rows_frame(in_df)
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
            if op.object_type == ObjectType.dataframe:
                agg_chunk = agg_op.new_chunk([chunk], shape=out_df.shape, index=chunk.index,
                                             index_value=out_df.index_value, dtypes=out_df.dtypes,
                                             columns_value=out_df.columns_value)
            else:
                agg_chunk = agg_op.new_chunk([chunk], shape=out_df.shape, index=chunk.index, dtype=out_df.dtype,
                                             index_value=out_df.index_value, name=out_df.name)
            agg_chunks.append(agg_chunk)

        new_op = op.copy()
        if op.object_type == ObjectType.dataframe:
            nsplits = ((np.nan,) * len(agg_chunks), (out_df.shape[1],))
        else:
            nsplits = ((np.nan,) * len(agg_chunks),)
        kw = out_df.params.copy()
        kw.update(dict(chunks=agg_chunks, nsplits=nsplits))
        return new_op.new_tileables([in_df], **kw)

    @classmethod
    def _tile_with_tree(cls, op):
        in_df = op.inputs[0]
        if len(in_df.shape) > 1:
            in_df = build_concatenated_rows_frame(in_df)
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
                chunk_op._object_type = ObjectType.dataframe
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
        kw = out_df.params.copy()
        kw['index'] = (0, 0) if op.object_type == ObjectType.dataframe else (0,)
        chunk = chunk_op.new_chunk([chk], **kw)
        new_op = op.copy()
        if op.object_type == ObjectType.dataframe:
            nsplits = ((out_df.shape[0],), (out_df.shape[1],))
        else:
            nsplits = ((out_df.shape[0],),)

        kw = out_df.params.copy()
        kw.update(dict(chunks=[chunk], nsplits=nsplits))
        return new_op.new_tileables(op.inputs, **kw)

    @classmethod
    def tile(cls, op: "DataFrameGroupByAgg"):
        if op.method == 'shuffle':
            return cls._tile_with_shuffle(op)
        elif op.method == 'tree':
            return cls._tile_with_tree(op)
        else:  # pragma: no cover
            raise NotImplementedError

    @classmethod
    def _get_grouped(cls, op: "DataFrameGroupByAgg", df, copy=False):
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

        if isinstance(df, pd.Series) and op.object_type == ObjectType.dataframe:
            df = pd.DataFrame(df, columns=[_series_col_name])

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
            try:
                result = grouped.agg(func)
                if result.empty and not df.empty:
                    # fix for py35, due to buffer read-only
                    raise ValueError
            except ValueError:
                # fail due to buffer read-only
                # force to get grouped again by copy
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

        if op.object_type == ObjectType.series:
            result = result.iloc[:, 0]
            result.name = op.series_name
        else:
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
                                 as_index=groupby.op.as_index, sort=groupby.op.sort,
                                 object_type=in_df.op.object_type)
    return agg_op(in_df)
