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

import itertools
from functools import partial

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...core import ENTITY_TYPE, Entity, OutputType
from ...core.operand import OperandStage, MapReduceOperand
from ...lib.groupby_wrapper import wrapped_groupby
from ...serialization.serializables import BoolField, Int32Field, AnyField
from ...utils import lazy_import
from ..align import align_dataframe_series, align_series_series
from ..initializer import Series as asseries
from ..core import SERIES_TYPE, SERIES_CHUNK_TYPE
from ..utils import (
    build_concatenated_rows_frame,
    hash_dataframe_on,
    build_df,
    build_series,
    parse_index,
    is_cudf,
)
from ..operands import DataFrameOperandMixin, DataFrameShuffleProxy


cudf = lazy_import("cudf")


class DataFrameGroupByOperand(MapReduceOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY

    _by = AnyField("by", on_serialize=lambda x: x.data if isinstance(x, Entity) else x)
    _level = AnyField("level", nullable=False)
    _as_index = BoolField("as_index", nullable=False)
    _sort = BoolField("sort", nullable=False)
    _group_keys = BoolField("group_keys", nullable=False)

    _shuffle_size = Int32Field("shuffle_size")

    def __init__(
        self,
        by=None,
        level=None,
        as_index=None,
        sort=None,
        group_keys=None,
        shuffle_size=None,
        output_types=None,
        **kw
    ):
        super().__init__(
            _by=by,
            _level=level,
            _as_index=as_index,
            _sort=sort,
            _group_keys=group_keys,
            _shuffle_size=shuffle_size,
            _output_types=output_types,
            **kw
        )
        if output_types:
            if self.stage in (OperandStage.map, OperandStage.reduce):
                if output_types[0] in (
                    OutputType.dataframe,
                    OutputType.dataframe_groupby,
                ):
                    output_types = [OutputType.dataframe]
                else:
                    output_types = [OutputType.series]
            else:
                if output_types[0] in (
                    OutputType.dataframe,
                    OutputType.dataframe_groupby,
                ):
                    output_types = [OutputType.dataframe_groupby]
                elif output_types[0] == OutputType.series:
                    output_types = [OutputType.series_groupby]
            self.output_types = output_types

    @property
    def by(self):
        return self._by

    @property
    def level(self):
        return self._level

    @property
    def as_index(self):
        return self._as_index

    @property
    def sort(self):
        return self._sort

    @property
    def group_keys(self):
        return self._group_keys

    @property
    def shuffle_size(self):
        return self._shuffle_size

    @property
    def is_dataframe_obj(self):
        return self.output_types[0] in (
            OutputType.dataframe_groupby,
            OutputType.dataframe,
        )

    @property
    def groupby_params(self):
        return dict(
            by=self.by,
            level=self.level,
            as_index=self.as_index,
            sort=self.sort,
            group_keys=self.group_keys,
        )

    def build_mock_groupby(self, **kwargs):
        in_df = self.inputs[0]
        if self.is_dataframe_obj:
            mock_obj = build_df(
                in_df, size=[2, 2], fill_value=[1, 2], ensure_string=True
            )
        else:
            mock_obj = build_series(
                in_df,
                size=[2, 2],
                fill_value=[1, 2],
                name=in_df.name,
                ensure_string=True,
            )

        new_kw = self.groupby_params
        new_kw.update(kwargs)
        if new_kw.get("level"):
            new_kw["level"] = 0
        if isinstance(new_kw["by"], list):
            new_by = []
            for v in new_kw["by"]:
                if isinstance(v, ENTITY_TYPE):
                    build_fun = build_df if v.ndim == 2 else build_series
                    mock_by = pd.concat(
                        [
                            build_fun(v, size=2, fill_value=1, name=v.name),
                            build_fun(v, size=2, fill_value=2, name=v.name),
                        ]
                    )
                    new_by.append(mock_by)
                else:
                    new_by.append(v)
            new_kw["by"] = new_by
        return mock_obj.groupby(**new_kw)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs[1:])
        if len(inputs) > 1:
            by = []
            for k in self._by:
                if isinstance(k, (SERIES_TYPE, SERIES_CHUNK_TYPE)):
                    by.append(next(inputs_iter))
                else:
                    by.append(k)
            self._by = by

    def __call__(self, df):
        params = df.params.copy()
        params["index_value"] = parse_index(None, df.key, df.index_value.key)
        if df.ndim == 2:
            if isinstance(self.by, list):
                index, types = [], []
                for k in self.by:
                    if isinstance(k, SERIES_TYPE):
                        index.append(k.name)
                        types.append(k.dtype)
                    elif k in df.dtypes:
                        index.append(k)
                        types.append(df.dtypes[k])
                    else:
                        raise KeyError(k)
                params["key_dtypes"] = pd.Series(types, index=index)

        inputs = [df]
        if isinstance(self.by, list):
            for k in self.by:
                if isinstance(k, SERIES_TYPE):
                    inputs.append(k)

        return self.new_tileable(inputs, **params)

    @classmethod
    def _align_input_and_by(cls, op, inp, by):
        align_method = (
            partial(align_dataframe_series, axis="index")
            if op.is_dataframe_obj
            else align_series_series
        )
        nsplits, _, inp_chunks, by_chunks = align_method(inp, by)

        inp_params = inp.params
        inp_params["chunks"] = inp_chunks
        inp_params["nsplits"] = nsplits
        inp = inp.op.copy().new_tileable(op.inputs, kws=[inp_params])

        by_params = by.params
        by_params["chunks"] = by_chunks
        if len(nsplits) == 2:
            by_nsplits = nsplits[:1]
        else:
            by_nsplits = nsplits
        by_params["nsplits"] = by_nsplits
        by = by.op.copy().new_tileable(by.op.inputs, kws=[by_params])

        return inp, by

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        by = op.by

        series_in_by = False
        new_inputs = []
        if len(op.inputs) > 1:
            # by series
            new_by = []
            for k in by:
                if isinstance(k, SERIES_TYPE):
                    in_df, k = cls._align_input_and_by(op, in_df, k)
                    if len(new_inputs) == 0:
                        new_inputs.append(in_df)
                    new_inputs.append(k)
                    series_in_by = True
                new_by.append(k)
            by = new_by
        else:
            new_inputs = op.inputs

        is_dataframe_obj = op.is_dataframe_obj
        if is_dataframe_obj:
            in_df = build_concatenated_rows_frame(in_df)
            output_type = OutputType.dataframe
            chunk_shape = (in_df.chunk_shape[0], 1)
        else:
            output_type = OutputType.series
            chunk_shape = (in_df.chunk_shape[0],)

        # generate map chunks
        map_chunks = []
        for chunk in in_df.chunks:
            map_op = op.copy().reset_key()
            map_op.stage = OperandStage.map
            map_op._shuffle_size = chunk_shape[0]
            map_op._output_types = [output_type]
            chunk_inputs = [chunk]
            if len(op.inputs) > 1:
                chunk_by = []
                for k in by:
                    if isinstance(k, SERIES_TYPE):
                        by_chunk = k.cix[
                            chunk.index[0],
                        ]
                        chunk_by.append(by_chunk)
                        chunk_inputs.append(by_chunk)
                    else:
                        chunk_by.append(k)
                map_op._by = chunk_by
            map_chunks.append(
                map_op.new_chunk(
                    chunk_inputs,
                    shape=(np.nan, np.nan),
                    index=chunk.index,
                )
            )

        proxy_chunk = DataFrameShuffleProxy(output_types=[output_type]).new_chunk(
            map_chunks, shape=()
        )

        # generate reduce chunks
        reduce_chunks = []
        out_indices = list(itertools.product(*(range(s) for s in chunk_shape)))
        for ordinal, out_idx in enumerate(out_indices):
            reduce_op = op.copy().reset_key()
            reduce_op._by = None
            reduce_op._output_types = [output_type]
            reduce_op.stage = OperandStage.reduce
            reduce_op.reducer_ordinal = ordinal
            reduce_op.n_reducers = len(out_indices)
            reduce_chunks.append(
                reduce_op.new_chunk(
                    [proxy_chunk], shape=(np.nan, np.nan), index=out_idx
                )
            )

        # generate groupby chunks
        out_chunks = []
        for chunk in reduce_chunks:
            groupby_op = op.copy().reset_key()
            if series_in_by:
                # set by to None, cuz data of by will be passed from map to reduce to groupby
                groupby_op._by = None
            if is_dataframe_obj:
                new_shape = (np.nan, in_df.shape[1])
            else:
                new_shape = (np.nan,)
            params = dict(shape=new_shape, index=chunk.index)
            if op.is_dataframe_obj:
                params.update(
                    dict(
                        dtypes=in_df.dtypes,
                        columns_value=in_df.columns_value,
                        index_value=parse_index(None, chunk.key, proxy_chunk.key),
                    )
                )
            else:
                params.update(
                    dict(
                        name=in_df.name,
                        dtype=in_df.dtype,
                        index_value=parse_index(None, chunk.key, proxy_chunk.key),
                    )
                )
            out_chunks.append(groupby_op.new_chunk([chunk], **params))

        new_op = op.copy()
        params = op.outputs[0].params.copy()
        if is_dataframe_obj:
            params["nsplits"] = ((np.nan,) * len(out_chunks), (in_df.shape[1],))
        else:
            params["nsplits"] = ((np.nan,) * len(out_chunks),)
        params["chunks"] = out_chunks
        return new_op.new_tileables(new_inputs, **params)

    @classmethod
    def execute_map(cls, ctx, op):
        is_dataframe_obj = op.is_dataframe_obj
        by = op.by
        chunk = op.outputs[0]
        df = ctx[op.inputs[0].key]

        deliver_by = False  # output by for the upcoming process
        if isinstance(by, list):
            new_by = []
            for v in by:
                if isinstance(v, ENTITY_TYPE):
                    deliver_by = True
                    new_by.append(ctx[v.key])
                else:
                    new_by.append(v)
            by = new_by

        if isinstance(by, list) or callable(by):
            on = by
        else:
            on = None

        if isinstance(df, tuple):
            filters = hash_dataframe_on(df[0], on, op.shuffle_size, level=op.level)
        else:
            filters = hash_dataframe_on(df, on, op.shuffle_size, level=op.level)

        def _take_index(src, f):
            result = src.iloc[f]
            if src.index.names:
                result.index.names = src.index.names
            if isinstance(src.index, pd.MultiIndex):
                result.index = result.index.remove_unused_levels()
            if is_cudf(result):  # pragma: no cover
                result = result.copy()
            return result

        for index_idx, index_filter in enumerate(filters):
            if is_dataframe_obj:
                reducer_index = (index_idx, chunk.index[1])
            else:
                reducer_index = (index_idx,)

            if deliver_by:
                filtered_by = []
                for v in by:
                    if isinstance(v, pd.Series):
                        filtered_by.append(_take_index(v, index_filter))
                    else:
                        filtered_by.append(v)
                if isinstance(df, tuple):
                    ctx[
                        chunk.key, reducer_index
                    ] = ctx.get_current_chunk().index, tuple(
                        _take_index(x, index_filter) for x in df
                    ) + (
                        filtered_by,
                        deliver_by,
                    )
                else:
                    ctx[chunk.key, reducer_index] = ctx.get_current_chunk().index, (
                        _take_index(df, index_filter),
                        filtered_by,
                        deliver_by,
                    )
            else:
                if isinstance(df, tuple):
                    ctx[chunk.key, reducer_index] = (
                        ctx.get_current_chunk().index,
                        tuple(_take_index(x, index_filter) for x in df) + (deliver_by,),
                    )
                else:
                    ctx[chunk.key, reducer_index] = (
                        ctx.get_current_chunk().index,
                        _take_index(df, index_filter),
                    )

    @classmethod
    def execute_reduce(cls, ctx, op: "DataFrameGroupByOperand"):
        xdf = cudf if op.gpu else pd
        chunk = op.outputs[0]
        input_idx_to_df = dict(op.iter_mapper_data(ctx))
        row_idxes = sorted(input_idx_to_df.keys())

        res = []
        for row_idx in row_idxes:
            row_df = input_idx_to_df.get(row_idx, None)
            if row_df is not None:
                res.append(row_df)
        by = None
        if isinstance(res[0], tuple):
            # By is series
            deliver_by = res[0][-1]
            r = []
            part_len = len(res[0])
            part_len -= 1 if not deliver_by else 2
            for n in range(part_len):
                r.append(xdf.concat([it[n] for it in res], axis=0))
            r = tuple(r)

            if deliver_by:
                by = [None] * len(res[0][-2])
                for it in res:
                    for i, v in enumerate(it[1]):
                        if isinstance(v, pd.Series):
                            if by[i] is None:
                                by[i] = v
                            else:
                                by[i] = pd.concat([by[i], v], axis=0)
                        else:
                            by[i] = v
        else:
            r = pd.concat(res, axis=0)

        if chunk.index_value is not None:
            if isinstance(r, tuple):
                for s in r:
                    s.index.name = chunk.index_value.name
            else:
                r.index.name = chunk.index_value.name
        if by is None:
            ctx[chunk.key] = r
        elif isinstance(r, tuple):
            ctx[chunk.key] = r + (by,)
        else:
            ctx[chunk.key] = (r, by)

    @classmethod
    def execute(cls, ctx, op: "DataFrameGroupByOperand"):
        if op.stage == OperandStage.map:
            cls.execute_map(ctx, op)
        elif op.stage == OperandStage.reduce:
            cls.execute_reduce(ctx, op)
        else:
            inp = ctx[op.inputs[0].key]
            if isinstance(inp, tuple):
                # df, by
                df, by = inp
            else:
                df = inp
                by = op.by
            ctx[op.outputs[0].key] = wrapped_groupby(
                df,
                by=by,
                level=op.level,
                as_index=op.as_index,
                sort=op.sort,
                group_keys=op.group_keys,
            )


def groupby(df, by=None, level=None, as_index=True, sort=True, group_keys=True):
    if not as_index and df.op.output_types[0] == OutputType.series:
        raise TypeError("as_index=False only valid with DataFrame")

    output_types = (
        [OutputType.dataframe_groupby] if df.ndim == 2 else [OutputType.series_groupby]
    )
    if isinstance(by, (SERIES_TYPE, pd.Series)):
        if isinstance(by, pd.Series):
            by = asseries(by)
        by = [by]
    elif df.ndim > 1 and by is not None and not isinstance(by, list):
        by = [by]
    op = DataFrameGroupByOperand(
        by=by,
        level=level,
        as_index=as_index,
        sort=sort,
        group_keys=group_keys,
        output_types=output_types,
    )
    return op(df)
