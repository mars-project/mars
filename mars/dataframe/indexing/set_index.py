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
from ...core import OutputType
from ...serialization.serializables import AnyField, BoolField
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import build_empty_df, parse_index


class DataFrameSetIndex(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_SET_INDEX

    _keys = AnyField('keys')
    _drop = BoolField('drop')
    _append = BoolField('append')
    _verify_integrity = BoolField('verify_integrity')

    def __init__(self, keys=None, drop=True, append=False, verify_integrity=False,
                 output_types=None, **kw):
        super().__init__(_keys=keys, _drop=drop, _append=append,
                         _verify_integrity=verify_integrity, _output_types=output_types, **kw)

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
    def _tile_column_axis_n_chunk(cls, op, in_df, out_df, out_chunks):
        if not isinstance(op.keys, str):  # pragma: no cover
            raise NotImplementedError('DataFrame.set_index only support label')
        if op.verify_integrity:  # pragma: no cover
            raise NotImplementedError('DataFrame.set_index not support verify_integrity yet')

        try:
            column_index = in_df.columns_value.to_pandas().get_loc(op.keys)
        except KeyError:  # pragma: no cover
            raise NotImplementedError('The new index label must be a column of the original dataframe')

        chunk_index = np.searchsorted(np.cumsum(in_df.nsplits[1]), column_index + 1)

        for row_idx in range(in_df.chunk_shape[0]):
            index_chunk = in_df.cix[row_idx, chunk_index]
            for col_idx in range(in_df.chunk_shape[1]):
                input_chunk = in_df.cix[row_idx, col_idx]
                if op.drop and input_chunk.key == index_chunk.key:
                    new_shape = (input_chunk.shape[0], input_chunk.shape[1] - 1)
                    selected = input_chunk.columns_value.to_pandas().drop(op.keys)
                    columns = parse_index(selected, store_data=True)
                    dtypes = input_chunk.dtypes.loc[selected]
                else:
                    new_shape = input_chunk.shape
                    columns = input_chunk.columns_value
                    dtypes = input_chunk.dtypes
                out_op = op.copy().reset_key()
                out_chunk = out_op.new_chunk([index_chunk, input_chunk],
                                             shape=new_shape, dtypes=dtypes, index=input_chunk.index,
                                             index_value=parse_index(pd.Int64Index([])),
                                             columns_value=columns)
                out_chunks.append(out_chunk)

    @classmethod
    def _tile_column_axis_1_chunk(cls, op, in_df, out_df, out_chunks):
        out_pd_index = out_df.index_value.to_pandas()
        for c in in_df.chunks:
            chunk_op = op.copy().reset_key()
            chunk_shape = (c.shape[0], out_df.shape[1])
            index_value = parse_index(out_pd_index, c)
            out_chunk = chunk_op.new_chunk([c], shape=chunk_shape,
                                           dtypes=out_df.dtypes, index=c.index,
                                           index_value=index_value,
                                           columns_value=out_df.columns_value)
            out_chunks.append(out_chunk)

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        out_chunks = []
        if in_df.chunk_shape[1] > 1:
            cls._tile_column_axis_n_chunk(op, in_df, out_df, out_chunks)
        else:
            cls._tile_column_axis_1_chunk(op, in_df, out_df, out_chunks)

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

        if len(op.inputs) == 2:
            # axis 1 has more than 1 chunk
            index_chunk, input_chunk = op.inputs
            # Optimization: we don't need to get value of the column
            # that is set as new index.
            if input_chunk.key == index_chunk.key:
                new_index = op.keys
            else:
                new_index = ctx[index_chunk.key][op.keys]
            ctx[chunk.key] = ctx[input_chunk.key].set_index(
                new_index, drop=op.drop, append=op.append,
                verify_integrity=op.verify_integrity)
        else:
            # axis 1 has 1 chunk
            inp = ctx[op.inputs[0].key]
            ctx[chunk.key] = inp.set_index(
                op.keys, drop=op.drop, append=op.append,
                verify_integrity=op.verify_integrity)


def set_index(df, keys, drop=True, append=False, inplace=False, verify_integrity=False):
    op = DataFrameSetIndex(keys=keys, drop=drop, append=append,
                           verify_integrity=verify_integrity, output_types=[OutputType.dataframe])
    result = op(df)
    if not inplace:
        return result
    else:
        df.data = result.data
