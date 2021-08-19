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
from pandas.api.types import is_list_like

from ... import opcodes
from ...core import OutputType, recursive_tile
from ...serialization.serializables import KeyField, AnyField
from ...tensor.core import TENSOR_TYPE
from ..core import DATAFRAME_TYPE, SERIES_TYPE, DataFrame
from ..initializer import DataFrame as asframe, Series as asseries
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
            elif isinstance(value, (pd.DataFrame, DATAFRAME_TYPE)):
                if len(self.indexes) != value.shape[1]:  # pragma: no cover
                    raise ValueError('Columns must be same length as key')

                value = asframe(value)
                value_dtype = pd.Series(list(value.dtypes), index=self._indexes)
            elif is_list_like(value) or isinstance(value, TENSOR_TYPE):
                # convert to numpy to get actual dim and shape
                if is_list_like(value):
                    value = np.array(value)

                if value.ndim == 1:
                    value = asseries(value, index=target.index)
                    value_dtype = value.dtype
                else:
                    if len(self.indexes) != value.shape[1]:  # pragma: no cover
                        raise ValueError('Columns must be same length as key')

                    value = asframe(value, index=target.index)
                    value_dtype = pd.Series(list(value.dtypes), index=self._indexes)
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

        try:
            dtypes.loc[self._indexes] = value_dtype
        except KeyError:
            # when some index not exist, try update one by one
            if isinstance(value_dtype, pd.Series):
                for idx in self._indexes:
                    dtypes.loc[idx] = value_dtype.loc[idx]
            else:
                for idx in self._indexes:
                    dtypes.loc[idx] = value_dtype

        columns_value = parse_index(dtypes.index, store_data=True)
        ret = self.new_dataframe(inputs, shape=(target.shape[0], len(dtypes)),
                                 dtypes=dtypes, index_value=index_value,
                                 columns_value=columns_value)
        raw_target.data = ret.data

    @classmethod
    def tile(cls, op: "DataFrameSetitem"):
        out = op.outputs[0]
        target = op.target
        value = op.value
        indexes = op.indexes
        columns = target.columns_value.to_pandas()
        is_value_scalar = np.isscalar(value) or cls._is_scalar_tensor(value)
        has_multiple_cols = getattr(out.dtypes[indexes], 'ndim', 0) > 0
        target_index_to_value = dict()

        if has_multiple_cols:
            append_cols = [c for c in indexes if c not in columns]
        else:
            append_cols = [indexes] if indexes not in columns else []

        if not is_value_scalar:
            rechunk_arg = {}

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

                rechunk_arg[0] = target.nsplits[0]

            if isinstance(value, DATAFRAME_TYPE):
                # rechunk in column dim given distribution of indexes in target chunks
                col_nsplits = []
                for head_chunk in target.cix[0, :]:
                    new_indexes = [vc for vc in indexes if vc in head_chunk.dtypes]
                    if new_indexes:
                        target_index_to_value[head_chunk.index[1]] = len(col_nsplits)
                        col_nsplits.append(len(new_indexes))

                if not col_nsplits:
                    # purely new chunks, just update last column of chunks
                    target_index_to_value[target.chunk_shape[1] - 1] = 0
                    col_nsplits = [len(append_cols)]
                else:
                    col_nsplits[-1] += len(append_cols)
                rechunk_arg[1] = col_nsplits

            if rechunk_arg:
                value = yield from recursive_tile(value.rechunk(rechunk_arg))

        out_chunks = []
        nsplits = [list(ns) for ns in target.nsplits]

        nsplits[1][-1] += len(append_cols)

        column_chunk_shape = target.chunk_shape[1]
        for c in target.chunks:
            result_chunk = c

            if has_multiple_cols:
                new_indexes = [vc for vc in indexes if vc in c.dtypes]
            else:
                new_indexes = [indexes] if indexes in c.dtypes else []

            if c.index[-1] == column_chunk_shape - 1:
                new_indexes.extend(append_cols)

            if new_indexes:
                # update needed on current chunk
                chunk_op = op.copy().reset_key()
                chunk_op._indexes = new_indexes if has_multiple_cols else new_indexes[0]

                if pd.api.types.is_scalar(value):
                    chunk_inputs = [c]
                elif is_value_scalar:
                    chunk_inputs = [c, value.chunks[0]]
                else:
                    # get proper chunk from value chunks
                    if has_multiple_cols:
                        value_chunk = value.cix[c.index[0], target_index_to_value[c.index[1]]]
                    else:
                        value_chunk = value.cix[c.index[0], ]

                    chunk_inputs = [c, value_chunk]

                dtypes, shape, columns_value = c.dtypes, c.shape, c.columns_value

                if append_cols and c.index[-1] == column_chunk_shape - 1:
                    # some columns appended at the last column of chunks
                    shape = (shape[0], shape[1] + len(append_cols))
                    dtypes = pd.concat([dtypes, out.dtypes.iloc[-len(append_cols):]])
                    columns_value = parse_index(dtypes.index, store_data=True)

                result_chunk = chunk_op.new_chunk(chunk_inputs,
                                                  shape=shape,
                                                  dtypes=dtypes,
                                                  index_value=c.index_value,
                                                  columns_value=columns_value,
                                                  index=c.index)
            out_chunks.append(result_chunk)

        params = out.params
        params['nsplits'] = tuple(tuple(ns) for ns in nsplits)
        params['chunks'] = out_chunks
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def estimate_size(cls, ctx: dict, op: "DataFrameSetitem"):
        result_size = ctx[op.target.key][0]
        ctx[op.outputs[0].key] = (result_size, result_size)

    @classmethod
    def execute(cls, ctx, op: "DataFrameSetitem"):
        target = ctx[op.target.key].copy()
        value = ctx[op.value.key] if not np.isscalar(op.value) else op.value
        target[op.indexes] = value
        ctx[op.outputs[0].key] = target


def dataframe_setitem(df, col, value):
    op = DataFrameSetitem(target=df, indexes=col, value=value)
    return op(df, value)
