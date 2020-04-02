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

from ...serialize import AnyField, BoolField
from ... import opcodes as OperandDef
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import build_empty_df, parse_index


class DataFrameSetIndex(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_SET_INDEX

    _keys = AnyField('keys')
    _drop = BoolField('drop')
    _append = BoolField('append')
    _verify_integrity = BoolField('verify_integrity')

    def __init__(self, keys=None, drop=True, append=False, verify_integrity=False,
                 object_type=None, **kw):
        super().__init__(_keys=keys, _drop=drop, _append=append,
                         _verify_integrity=verify_integrity, _object_type=object_type, **kw)

    @property
    def keys(self):
        return self._keys

    @property
    def drop(self):
        return self._drop

    @property
    def append(self):
        return self._append

    @property
    def verify_integrity(self):
        return self._verify_integrity

    def __call__(self, df):
        new_df = build_empty_df(df.dtypes).set_index(keys=self.keys, drop=self.drop, append=self.append,
                                                     verify_integrity=self.verify_integrity)
        return self.new_dataframe([df], shape=(df.shape[0], new_df.shape[1]), dtypes=new_df.dtypes,
                                  index_value=parse_index(new_df.index),
                                  columns_value=parse_index(new_df.columns, store_data=True))

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        if not isinstance(op.keys, str):
            raise NotImplementedError('DataFrame.set_index only support label')
        if op.verify_integrity:
            raise NotImplementedError('DataFrame.set_index not support verify_integrity yet')

        out_chunks = []

        try:
            column_index = in_df.columns_value.to_pandas().get_loc(op.keys)
        except KeyError:
            raise NotImplementedError('The new index label must be a column of the original dataframe')

        chunk_index = np.searchsorted(np.cumsum(in_df.nsplits[1]), column_index + 1)

        for row_idx in range(in_df.chunk_shape[0]):
            index_chunk = in_df.cix[row_idx, chunk_index]
            for col_idx in range(in_df.chunk_shape[1]):
                input_chunk = in_df.cix[row_idx, col_idx]
                if op.drop and input_chunk.key == index_chunk.key:
                    new_shape = (input_chunk.shape[0], input_chunk.shape[1] - 1)
                    columns = parse_index(input_chunk.columns_value.to_pandas().drop(op.keys), store_data=True)
                else:
                    new_shape = input_chunk.shape
                    columns = input_chunk.columns_value
                out_op = op.copy().reset_key()
                out_chunk = out_op.new_chunk([index_chunk, input_chunk],
                                             shape=new_shape, dtypes=out_df.dtypes, index=input_chunk.index,
                                             index_value=parse_index(pd.Int64Index([])),
                                             columns_value=columns)
                out_chunks.append(out_chunk)

        new_op = op.copy()
        columns_nsplits = list(in_df.nsplits[1])
        if op.drop:
            columns_nsplits = tuple(split - 1 if i == 0 else split for i, split in enumerate(columns_nsplits))
        nsplits = (in_df.nsplits[0], columns_nsplits)
        return new_op.new_dataframes(op.inputs, out_df.shape, dtypes=out_df.dtypes,
                                     index_value=out_df.index_value,
                                     columns_value=out_df.columns_value,
                                     chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        index_chunk, input_chunk = op.inputs
        # Optimization: we don't need to get value of the column that is set as new index.
        if input_chunk.key == index_chunk.key:
            new_index = op.keys
        else:
            new_index = ctx[index_chunk.key][op.keys]
        ctx[chunk.key] = ctx[input_chunk.key].set_index(new_index, drop=op.drop, append=op.append,
                                                        verify_integrity=op.verify_integrity)


def set_index(df, keys, drop=True, append=False, verify_integrity=False, **kw):
    op = DataFrameSetIndex(keys=keys, drop=drop, append=append,
                           verify_integrity=verify_integrity, object_type=ObjectType.dataframe, **kw)
    return op(df)
