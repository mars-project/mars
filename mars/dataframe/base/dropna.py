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

import numpy as np
import pandas as pd

from ... import opcodes
from ...config import options
from ...serialize import AnyField, BoolField, StringField, Int32Field
from ..align import align_dataframe_series
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import parse_index, validate_axis


class DataFrameDropNA(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.DROP_NA

    _axis = AnyField('axis')
    _how = StringField('how')
    _thresh = Int32Field('thresh')
    _subset = AnyField('subset')
    _use_inf_as_na = BoolField('use_inf_as_na')

    # when True, dropna will be called on the input,
    # otherwise non-nan counts will be used
    _drop_directly = BoolField('drop_directly')
    # size of subset, used when how == 'any'
    _subset_size = Int32Field('subset_size')

    def __init__(self, axis=None, how=None, thresh=None, subset=None, use_inf_as_na=None,
                 drop_directly=None, subset_size=None, sparse=None, object_type=None, **kw):
        super().__init__(_axis=axis, _how=how, _thresh=thresh, _subset=subset,
                         _use_inf_as_na=use_inf_as_na, _drop_directly=drop_directly,
                         _subset_size=subset_size, _sparse=sparse, _object_type=object_type, **kw)

    @property
    def axis(self) -> int:
        return self._axis

    @property
    def how(self) -> str:
        return self._how

    @property
    def thresh(self) -> int:
        return self._thresh

    @property
    def subset(self) -> list:
        return self._subset

    @property
    def use_inf_as_na(self) -> bool:
        return self._use_inf_as_na

    @property
    def drop_directly(self) -> bool:
        return self._drop_directly

    @property
    def subset_size(self) -> int:
        return self._subset_size

    def __call__(self, df):
        new_shape = list(df.shape)
        new_shape[0] = np.nan

        params = df.params.copy()
        params['index_value'] = parse_index(None, df.key, df.index_value.key)
        params['shape'] = tuple(new_shape)
        return self.new_tileable([df], **params)

    @classmethod
    def _tile_drop_directly(cls, op: "DataFrameDropNA"):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        chunks = []
        for c in in_df.chunks:
            new_shape = list(c.shape)
            new_shape[0] = np.nan

            params = c.params.copy()
            params['index_value'] = parse_index(None, c.key, c.index_value.key)
            params['shape'] = tuple(new_shape)

            new_op = op.copy().reset_key()
            new_op._drop_directly = True
            chunks.append(new_op.new_chunk([c], **params))

        new_nsplits = list(in_df.nsplits)
        new_nsplits[0] = (np.nan,) * len(in_df.nsplits[0])

        new_op = op.copy().reset_key()
        params = out_df.params.copy()
        params.update(dict(chunks=chunks, nsplits=new_nsplits))
        return new_op.new_tileables([in_df], **params)

    @classmethod
    def tile(cls, op: "DataFrameDropNA"):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        if len(in_df.chunk_shape) == 1 or in_df.chunk_shape[1] == 1:
            return cls._tile_drop_directly(op)

        subset_df = in_df
        if op.subset:
            subset_df = in_df[op.subset]._inplace_tile()
        count_series = subset_df.agg('count', axis=1, _use_inf_as_na=op.use_inf_as_na)._inplace_tile()

        nsplits, out_shape, left_chunks, right_chunks = align_dataframe_series(in_df, count_series, axis=0)
        out_chunk_indexes = itertools.product(*(range(s) for s in out_shape))

        out_chunks = []
        for out_idx, df_chunk in zip(out_chunk_indexes, left_chunks):
            series_chunk = right_chunks[out_idx[0]]
            kw = dict(shape=(nsplits[0][out_idx[0]], nsplits[1][out_idx[1]]),
                      index_value=df_chunk.index_value, columns_value=df_chunk.columns_value)

            new_op = op.copy().reset_key()
            new_op._drop_directly = False
            new_op._subset_size = len(op.subset) if op.subset else len(in_df.dtypes)
            out_chunks.append(new_op.new_chunk([df_chunk, series_chunk], index=out_idx, **kw))

        new_op = op.copy().reset_key()
        params = out_df.params.copy()
        params.update(dict(nsplits=tuple(tuple(ns) for ns in nsplits), chunks=out_chunks))
        return new_op.new_tileables(op.inputs, **params)

    @classmethod
    def execute(cls, ctx, op: "DataFrameDropNA"):
        try:
            pd.set_option('mode.use_inf_as_na', op.use_inf_as_na)

            in_data = ctx[op.inputs[0].key]
            if op.drop_directly:
                if isinstance(in_data, pd.DataFrame):
                    result = in_data.dropna(axis=op.axis, how=op.how, thresh=op.thresh,
                                            subset=op.subset)
                else:
                    result = in_data.dropna(axis=op.axis, how=op.how)
                ctx[op.outputs[0].key] = result
                return

            in_counts = ctx[op.inputs[1].key]
            if op.how == 'all':
                in_counts = in_counts[in_counts > 0]
            else:
                thresh = op.subset_size if op.thresh is None else op.thresh
                in_counts = in_counts[in_counts >= thresh]

            ctx[op.outputs[0].key] = in_data.reindex(in_counts.index)
        finally:
            pd.reset_option('mode.use_inf_as_na')


def df_dropna(df, axis=0, how='any', thresh=None, subset=None, inplace=False):
    axis = validate_axis(axis, df)
    if axis != 0:
        raise NotImplementedError('Does not support dropna on DataFrame when axis=1')

    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameDropNA(axis=axis, how=how, thresh=thresh, subset=subset,
                         object_type=ObjectType.dataframe, use_inf_as_na=use_inf_as_na)
    out_df = op(df)
    if inplace:
        df.data = out_df.data
    else:
        return out_df


def series_dropna(series, axis=0, inplace=False, how=None):
    axis = validate_axis(axis, series)
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameDropNA(axis=axis, how=how, object_type=ObjectType.series,
                         use_inf_as_na=use_inf_as_na)
    out_series = op(series)
    if inplace:
        series.data = out_series.data
    else:
        return out_series
