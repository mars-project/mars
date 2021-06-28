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

from ... import opcodes
from ...core import recursive_tile
from ...serialization.serializables import AnyField, BoolField, Int64Field
from ...tensor.core import TENSOR_TYPE, TENSOR_CHUNK_TYPE
from ..core import SERIES_TYPE, SERIES_CHUNK_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import build_empty_df, parse_index


class DataFrameInsert(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.INSERT

    _loc = Int64Field('loc')
    _column = AnyField('column')
    _value = AnyField('value')
    _allow_duplicates = BoolField('allow_duplicates')

    def __init__(self, loc=None, column=None, value=None, allow_duplicates=None, **kw):
        super().__init__(_loc=loc, _column=column, _value=value,
                         _allow_duplicates=allow_duplicates, **kw)

    @property
    def loc(self) -> int:
        return self._loc

    @property
    def column(self):
        return self._column

    @property
    def value(self):
        return self._value

    @property
    def allow_duplicates(self):
        return self._allow_duplicates

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if len(inputs) > 1:
            self._value = self._inputs[-1]

    def __call__(self, df):
        inputs = [df]
        if isinstance(self.value, (SERIES_TYPE, TENSOR_TYPE)):
            value_dtype = self.value.dtype
            inputs.append(self.value)
        else:
            value_dtype = pd.Series(self.value).dtype

        empty_df = build_empty_df(df.dtypes)
        empty_df.insert(loc=self.loc, column=self.column, allow_duplicates=self.allow_duplicates,
                        value=pd.Series([], dtype=value_dtype))

        params = df.params
        params['columns_value'] = parse_index(empty_df.columns, store_data=True)
        params['dtypes'] = empty_df.dtypes
        params['shape'] = (df.shape[0], df.shape[1] + 1)
        return self.new_dataframe(inputs, **params)

    @classmethod
    def tile(cls, op: 'DataFrameInsert'):
        inp = op.inputs[0]
        value = op.value
        if isinstance(value, (SERIES_TYPE, TENSOR_TYPE)):
            value = yield from recursive_tile(
                value.rechunk({0: inp.nsplits[0]}))
        out = op.outputs[0]

        chunk_bounds = np.cumsum((0,) + inp.nsplits[1])
        chunk_bounds[-1] += 1

        chunks = []
        new_split = list(inp.nsplits[1])
        chunk_dtypes = None
        chunk_columns_value = None
        for c in inp.chunks:
            left_bound = int(chunk_bounds[c.index[1]])
            right_bound = int(chunk_bounds[c.index[1] + 1])
            if left_bound > op.loc or right_bound <= op.loc:
                chunks.append(c)
                continue

            if chunk_dtypes is None:
                new_split[c.index[1]] = inp.nsplits[1][c.index[1]] + 1

                if isinstance(value, (SERIES_TYPE, TENSOR_TYPE)):
                    value_dtype = value.dtype
                else:
                    value_dtype = pd.Series(value).dtype

                empty_df = build_empty_df(c.dtypes)
                empty_df.insert(
                    loc=op.loc - left_bound, column=op.column, allow_duplicates=op.allow_duplicates,
                    value=pd.Series([], dtype=value_dtype))

                chunk_dtypes = empty_df.dtypes
                chunk_columns_value = parse_index(chunk_dtypes.index, store_data=True)

            params = c.params
            params['columns_value'] = chunk_columns_value
            params['dtypes'] = chunk_dtypes
            params['shape'] = (c.shape[0], c.shape[1] + 1)

            new_op = op.copy().reset_key()
            new_op._loc = op.loc - left_bound

            if isinstance(value, (SERIES_TYPE, TENSOR_TYPE)):
                inputs = [c, value.chunks[c.index[0]]]
            else:
                inputs = [c]
            chunks.append(new_op.new_chunk(inputs, **params))

        new_op = op.copy().reset_key()
        return new_op.new_tileables(
            [inp], chunks=chunks, nsplits=(inp.nsplits[0], tuple(new_split)), **out.params)

    @classmethod
    def execute(cls, ctx, op: 'DataFrameInsert'):
        input_ = ctx[op.inputs[0].key]
        value = op.value
        if isinstance(value, (SERIES_CHUNK_TYPE, TENSOR_CHUNK_TYPE)):
            value = ctx[value.key]
        ctx[op.outputs[0].key] = copied = input_.copy()
        copied.insert(loc=op.loc, column=op.column, allow_duplicates=op.allow_duplicates, value=value)


def df_insert(df, loc, column, value, allow_duplicates=False):
    """
    Insert column into DataFrame at specified location.

    Raises a ValueError if `column` is already contained in the DataFrame,
    unless `allow_duplicates` is set to True.

    Parameters
    ----------
    loc : int
        Insertion index. Must verify 0 <= loc <= len(columns).
    column : str, number, or hashable object
        Label of the inserted column.
    value : int, Series, or array-like
    allow_duplicates : bool, optional
    """
    if isinstance(value, TENSOR_TYPE) and value.ndim > 1:
        raise ValueError(f'Wrong number of items passed {value.ndim}, placement implies 1')

    op = DataFrameInsert(loc=loc, column=column, value=value, allow_duplicates=allow_duplicates)
    out_df = op(df)
    df.data = out_df.data
