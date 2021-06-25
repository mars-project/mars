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

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ... import tensor as mt
from ...core import recursive_tile
from ...core.operand import OperandStage
from ...serialization.serializables import FieldTypes, KeyField, ListField, AnyField
from ...utils import lazy_import
from ..core import SERIES_TYPE
from ..initializer import DataFrame, Series
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index, build_empty_df


cudf = lazy_import('cudf')


class DataFrameDescribe(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DESCRIBE

    _input = KeyField('input')
    _percentiles = ListField('percentiles', FieldTypes.float64)
    _include = AnyField('include')
    _exclude = AnyField('exclude')

    def __init__(self, percentiles=None, include=None, exclude=None,
                 output_types=None, **kw):
        super().__init__(_percentiles=percentiles, _include=include,
                         _exclude=exclude, _output_types=output_types, **kw)

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
        if self.stage != OperandStage.agg:
            self._input = self._inputs[0]

    def __call__(self, df_or_series):
        if isinstance(df_or_series, SERIES_TYPE):
            if not np.issubdtype(df_or_series.dtype, np.number):
                raise NotImplementedError('non-numeric type is not supported for now')
            test_series = pd.Series([], dtype=df_or_series.dtype).describe(
                percentiles=self._percentiles, include=self._include, exclude=self._exclude)
            return self.new_series([df_or_series], shape=(len(test_series),),
                                   dtype=test_series.dtype,
                                   index_value=parse_index(test_series.index, store_data=True))
        else:
            test_inp_df = build_empty_df(df_or_series.dtypes)
            test_df = test_inp_df.describe(
                percentiles=self._percentiles, include=self._include, exclude=self._exclude)
            if len(self.percentiles) == 0:
                # specify percentiles=False
                # Note: unlike pandas that False is illegal value for percentiles,
                # Mars DataFrame allows user to specify percentiles=False
                # to skip computation about percentiles
                test_df.drop(['50%'], axis=0, inplace=True)
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
            result = yield from cls._tile_series(op)
        else:
            result = yield from cls._tile_dataframe(op)
        return result

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
        ret = yield from recursive_tile(ret)
        return [ret]

    @classmethod
    def _tile_dataframe(cls, op):
        df = DataFrame(op.input)
        out = op.outputs[0]
        dtypes = out.dtypes
        columns = dtypes.index.tolist()

        if df.chunk_shape[1] > 1:
            df = df.rechunk({1: df.shape[1]})

        # check dtypes if selected all fields
        # to reduce graph scale
        if df.dtypes.index.tolist() != columns:
            df = df[columns]

        # perform aggregation together
        aggregation = yield from recursive_tile(
            df.agg(['count', 'mean', 'std', 'min', 'max']))
        # calculate percentiles
        percentiles = None
        if len(op.percentiles) > 0:
            percentiles = yield from recursive_tile(
                df.quantile(op.percentiles))

        chunk_op = DataFrameDescribe(output_types=op.output_types,
                                     stage=OperandStage.agg,
                                     percentiles=op.percentiles)
        chunk_params = out.params.copy()
        chunk_params['index'] = (0, 0)
        in_chunks = aggregation.chunks
        if percentiles is not None:
            in_chunks += percentiles.chunks
        out_chunk = chunk_op.new_chunk(in_chunks, kws=[chunk_params])

        new_op = op.copy()
        params = out.params.copy()
        params['chunks'] = [out_chunk]
        params['nsplits'] = tuple((s,) for s in out.shape)
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def execute(cls, ctx, op):
        out = op.outputs[0]
        if op.stage is None:  # 1 chunk
            df_or_series = ctx[op.input.key]

            ctx[out.key] = df_or_series.describe(
                percentiles=op.percentiles, include=op.include, exclude=op.exclude)
        else:
            assert op.stage == OperandStage.agg

            inputs = [ctx[inp.key] for inp in op.inputs]
            xdf = pd if isinstance(inputs[0], (pd.DataFrame, pd.Series, pd.Index)) \
                        or cudf is None else cudf

            if len(inputs) == 1:
                df = inputs[0]
            else:
                assert len(inputs) > 1
                aggregations = inputs[0]
                percentiles = xdf.concat(inputs[1:], axis=0)
                df = xdf.concat([aggregations.iloc[:-1], percentiles,
                                 aggregations.iloc[-1:]], axis=0)
            # ['count', 'mean', 'std', 'min', {percentiles}, 'max']
            df.index = out.index_value.to_pandas()
            ctx[out.key] = df


def describe(df_or_series, percentiles=None, include=None, exclude=None):
    if percentiles is False:
        percentiles = []
    elif percentiles is None:
        percentiles = [0.25, 0.5, 0.75]
    else:
        percentiles = list(percentiles)
        if percentiles is not None:
            for p in percentiles:
                if p < 0 or p > 1:
                    raise ValueError('percentiles should all be in the interval [0, 1]. '
                                     'Try [{0:.3f}] instead.'.format(p / 100))
        # median should always be included
        if 0.5 not in percentiles:
            percentiles.append(0.5)
        percentiles = np.asarray(percentiles)

        # sort and check for duplicates
        unique_pcts = np.unique(percentiles)
        if len(unique_pcts) < len(percentiles):
            raise ValueError("percentiles cannot contain duplicates")
        percentiles = unique_pcts.tolist()

    op = DataFrameDescribe(percentiles=percentiles, include=include, exclude=exclude)
    return op(df_or_series)
