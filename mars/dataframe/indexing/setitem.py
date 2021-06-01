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
from pandas.api.types import is_list_like

from ... import opcodes
from ...core import OutputType, recursive_tile
from ...serialization.serializables import KeyField, AnyField
from ...tensor.core import TENSOR_TYPE
from ..core import SERIES_TYPE, DataFrame
from ..initializer import Series as asseries
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index


class DataFrameSetitem(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.INDEXSETVALUE

    _target = KeyField('target')
    _indexes = AnyField('indexes')
    _value = AnyField('value')

    def __init__(self, target=None, indexes=None, value=None, output_types=None, **kw):
        super().__init__(_target=target, _indexes=indexes,
                         _value=value, _output_types=output_types, **kw)
        if self.output_types is None:
            self.output_types = [OutputType.dataframe]

    @property
    def target(self):
        return self._target

    @property
    def indexes(self):
        return self._indexes

    @property
    def value(self):
        return self._value

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._target = self._inputs[0]
        if len(inputs) > 1:
            self._value = self._inputs[-1]

    @staticmethod
    def _is_scalar_tensor(t):
        return isinstance(t, TENSOR_TYPE) and t.ndim == 0

    def __call__(self, target: DataFrame, value):
        raw_target = target

        inputs = [target]
        if np.isscalar(value):
            value_dtype = np.array(value).dtype
        elif self._is_scalar_tensor(value):
            inputs.append(value)
            value_dtype = value.dtype
        else:
            if isinstance(value, (pd.Series, SERIES_TYPE)):
                value = asseries(value)
                value_dtype = value.dtype
            elif is_list_like(value) or isinstance(value, TENSOR_TYPE):
                value = asseries(value, index=target.index)
                value_dtype = value.dtype
            else:  # pragma: no cover
                raise TypeError('Wrong value type, could be one of scalar, Series or tensor')

            if target.shape[0] == 0:
                # target empty, reindex target first
                target = target.reindex(value.index)
                inputs[0] = target
            elif value.index_value.key != target.index_value.key:
                # need reindex when target df is not empty and index different
                value = value.reindex(target.index)
            inputs.append(value)

        index_value = target.index_value
        dtypes = target.dtypes.copy(deep=True)
        dtypes.loc[self._indexes] = value_dtype
        columns_value = parse_index(dtypes.index, store_data=True)
        ret = self.new_dataframe(inputs, shape=(target.shape[0], len(dtypes)),
                                 dtypes=dtypes, index_value=index_value,
                                 columns_value=columns_value)
        raw_target.data = ret.data

    @classmethod
    def tile(cls, op):
        out = op.outputs[0]
        target = op.target
        value = op.value
        col = op.indexes
        columns = target.columns_value.to_pandas()
        is_value_scalar = np.isscalar(value) or cls._is_scalar_tensor(value)

        if not is_value_scalar:
            # check if all chunk's index_value are identical
            target_chunk_index_values = [c.index_value for c in target.chunks
                                         if c.index[1] == 0]
            value_chunk_index_values = [v.index_value for v in value.chunks]
            is_identical = len(target_chunk_index_values) == len(target_chunk_index_values) and \
                all(c.key == v.key for c, v in zip(target_chunk_index_values, value_chunk_index_values))
            if not is_identical:
                # do rechunk
                if any(np.isnan(s) for s in target.nsplits[0]) or \
                        any(np.isnan(s) for s in value.nsplits[0]):  # pragma: no cover
                    yield

                value = yield from recursive_tile(
                    value.rechunk({0: target.nsplits[0]}))

        out_chunks = []
        nsplits = [list(ns) for ns in target.nsplits]
        if col not in columns:
            nsplits[1][-1] += 1
            column_chunk_shape = target.chunk_shape[1]
            # append to the last chunk on columns axis direction
            for c in target.chunks:
                if c.index[-1] != column_chunk_shape - 1:
                    # not effected, just output
                    out_chunks.append(c)
                else:
                    chunk_op = op.copy().reset_key()
                    if pd.api.types.is_scalar(value):
                        chunk_inputs = [c]
                    elif is_value_scalar:
                        chunk_inputs = [c, value.chunks[0]]
                    else:
                        value_chunk = value.cix[c.index[0], ]
                        chunk_inputs = [c, value_chunk]

                    dtypes = c.dtypes.copy(deep=True)
                    dtypes.loc[out.dtypes.index[-1]] = out.dtypes.iloc[-1]
                    chunk = chunk_op.new_chunk(chunk_inputs,
                                               shape=(c.shape[0], c.shape[1] + 1),
                                               dtypes=dtypes,
                                               index_value=c.index_value,
                                               columns_value=parse_index(dtypes.index, store_data=True),
                                               index=c.index)
                    out_chunks.append(chunk)
        else:
            # replace exist column
            for c in target.chunks:
                if col in c.dtypes:
                    chunk_inputs = [c]
                    if not np.isscalar(value):
                        chunk_inputs.append(value.cix[c.index[0], ])
                    chunk_op = op.copy().reset_key()
                    chunk = chunk_op.new_chunk(chunk_inputs,
                                               shape=c.shape,
                                               dtypes=c.dtypes,
                                               index_value=c.index_value,
                                               columns_value=c.columns_value,
                                               index=c.index)
                    out_chunks.append(chunk)
                else:
                    out_chunks.append(c)

        params = out.params
        params['nsplits'] = tuple(tuple(ns) for ns in nsplits)
        params['chunks'] = out_chunks
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def execute(cls, ctx, op):
        target = ctx[op.target.key].copy()
        value = ctx[op.value.key] if not np.isscalar(op.value) else op.value
        target[op.indexes] = value
        ctx[op.outputs[0].key] = target


def dataframe_setitem(df, col, value):
    op = DataFrameSetitem(target=df, indexes=col, value=value)
    return op(df, value)
