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

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ... import tensor as mt
from ...serialize import ValueType, KeyField, ListField
from ...utils import recursive_tile
from ..core import SERIES_TYPE
from ..initializer import DataFrame, Series
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import parse_index, build_empty_df


class DataFrameDescribe(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DESCRIBE

    _input = KeyField('input')
    _percentiles = ListField('percentiles', ValueType.float64)
    _include = ListField('include')
    _exclude = ListField('exclude')

    def __init__(self, percentiles=None, include=None, exclude=None,
                 object_type=None, **kw):
        super().__init__(_percentiles=percentiles, _include=include,
                         _exclude=exclude, _object_type=object_type, **kw)

    @property
    def input(self):
        return self._input

    @property
    def percentiles(self):
        return self._percentiles

    @property
    def include(self):
        return self._include

    @property
    def exclude(self):
        return self._exclude

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, df_or_series):
        if isinstance(df_or_series, SERIES_TYPE):
            self._object_type = ObjectType.series
            if not np.issubdtype(df_or_series.dtype, np.number):
                raise NotImplementedError('non-numeric type is not supported for now')
            test_series = pd.Series([], dtype=df_or_series.dtype).describe(
                percentiles=self._percentiles, include=self._include, exclude=self._exclude)
            return self.new_series([df_or_series], shape=(len(test_series),),
                                   dtype=test_series.dtype,
                                   index_value=parse_index(test_series.index, store_data=True))
        else:
            self._object_type = ObjectType.dataframe
            test_inp_df = build_empty_df(df_or_series.dtypes)
            test_df = test_inp_df.describe(
                percentiles=self._percentiles, include=self._include, exclude=self._exclude)
            for dtype in test_df.dtypes:
                if not np.issubdtype(dtype, np.number):
                    raise NotImplementedError('non-numeric type is not supported for now')
            return self.new_dataframe([df_or_series], shape=test_df.shape, dtypes=test_df.dtypes,
                                      index_value=parse_index(test_df.index, store_data=True),
                                      columns_value=parse_index(test_df.columns, store_data=True))

    @classmethod
    def tile(cls, op):
        inp = op.input

        if len(inp.chunks) == 1:
            return cls._tile_one_chunk(op)

        if isinstance(inp, SERIES_TYPE):
            return cls._tile_series(op)
        else:
            return cls._tile_dataframe(op)

    @classmethod
    def _tile_one_chunk(cls, op):
        out = op.outputs[0]

        chunk_op = op.copy().reset_key()
        chunk_params = out.params.copy()
        chunk_params['index'] = (0,) * out.ndim
        out_chunk = chunk_op.new_chunk([op.input.chunks[0]], kws=[chunk_params])

        new_op = op.copy()
        params = out.params.copy()
        params['chunks'] = [out_chunk]
        params['nsplits'] = tuple((s,) for s in out.shape)
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def _tile_series(cls, op):
        series = Series(op.input)
        out = op.outputs[0]
        index = out.index_value.to_pandas()
        # ['count', 'mean', 'std', 'min', {percentiles}, 'max']
        names = index.tolist()

        values = [None] * 6
        for i, agg in enumerate(names[:4]):
            values[i] = mt.atleast_1d(getattr(series, agg)())
        values[-1] = mt.atleast_1d(getattr(series, names[-1])())
        values[4] = series.quantile(op.percentiles).to_tensor()

        t = mt.concatenate(values).rechunk(len(names))
        ret = Series(t, index=index, name=series.name)
        return [recursive_tile(ret)]

    @classmethod
    def _tile_dataframe(cls, op):
        df = DataFrame(op.input)
        out = op.outputs[0]
        index = out.index_value.to_pandas()
        dtypes = out.dtypes
        columns = dtypes.index.tolist()
        # ['count', 'mean', 'std', 'min', {percentiles}, 'max']
        names = index.tolist()

        df = df[columns]

        values = [None] * 6
        for i, agg in enumerate(names[:4]):
            values[i] = getattr(df, agg)().to_tensor()[None, :]
        values[-1] = getattr(df, names[-1])().to_tensor()[None, :]
        values[4] = df.quantile(op.percentiles).to_tensor()

        t = mt.concatenate(values).rechunk((len(index), len(columns)))
        ret = DataFrame(t, index=index, columns=columns)
        return [recursive_tile(ret)]

    @classmethod
    def execute(cls, ctx, op):
        df_or_series = ctx[op.input.key]

        ctx[op.outputs[0].key] = df_or_series.describe(
            percentiles=op.percentiles, include=op.include, exclude=op.exclude)


def describe(df_or_series, percentiles=None, include=None, exclude=None):
    if percentiles is not None:
        for p in percentiles:
            if p < 0 or p > 1:
                raise ValueError('percentiles should all be in the interval [0, 1]. '
                                 'Try [{0:.3f}] instead.'.format(p / 100))
    if percentiles is None:
        percentiles = [0.25, 0.5, 0.75]
    if not percentiles:
        percentiles = [0.5]

    op = DataFrameDescribe(percentiles=percentiles, include=include, exclude=exclude)
    return op(df_or_series)
