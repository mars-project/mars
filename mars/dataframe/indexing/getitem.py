# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import pandas as pd

from ... import opcodes as OperandDef
from ...serialize import AnyField
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import parse_index
from .utils import calc_columns_index


class DataFrameIndex(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.INDEX

    _indexes = AnyField('indexes')

    def __init__(self, indexes=None, object_type=ObjectType.series, **kw):
        super(DataFrameIndex, self).__init__(_indexes=indexes, _object_type=object_type, **kw)

    @property
    def indexes(self):
        return self._indexes

    def __call__(self, df):
        # if indexes is a tuple or list, return a DataFrame, else return a Series
        if isinstance(self._indexes, (list, tuple)):
            dtypes = df.dtypes[self._indexes]
            columns = parse_index(pd.Index(self._indexes), store_data=True)
            return self.new_dataframe([df], shape=(df.shape[0], len(self._indexes)), dtypes=dtypes,
                                      index_value=df.index_value, columns_value=columns)
        else:
            dtype = df.dtypes[self._indexes]
            return self.new_series([df], shape=(df.shape[0],), dtype=dtype, index_value=df.index_value,
                                   name=self._indexes)

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]
        indexes = op.indexes
        if not isinstance(indexes, (tuple, list)):
            column_index = calc_columns_index(indexes, in_df)
            out_chunks = []
            dtype = in_df.dtypes[indexes]
            for i in range(in_df.chunk_shape[0]):
                c = in_df.cix[(i, column_index)]
                op = DataFrameIndex(indexes=indexes)
                out_chunks.append(op.new_chunk([c], shape=(c.shape[0],), index=(i,), dtype=dtype,
                                               index_value=c.index_value, name=indexes))
            new_op = op.copy()
            return new_op.new_seriess(op.inputs, shape=out_df.shape, dtype=out_df.dtype,
                                      index_value=out_df.index_value, name=out_df.name,
                                      nsplits=(in_df.nsplits[0],), chunks=out_chunks)
        else:
            selected_index = []
            for ind in indexes:
                column_index = calc_columns_index(ind, in_df)
                selected_index.append(column_index)

            # combine columns in one chunk and keep the columns order at the same time.
            column_splits = []
            start = 0
            for i in range(0, len(selected_index)):
                if i == len(selected_index) - 1:
                    column_splits.append((indexes[slice(start, i + 1)], selected_index[start]))
                    continue
                if selected_index[i] != selected_index[i + 1]:
                    column_splits.append((indexes[slice(start, i + 1)], selected_index[start]))
                    start = i + 1

            out_chunks = [[] for _ in range(in_df.chunk_shape[0])]
            column_nsplits = []
            for i, (columns, column_idx) in enumerate(column_splits):
                dtypes = in_df.dtypes[columns]
                column_nsplits.append(len(columns))
                for j in range(in_df.chunk_shape[0]):
                    c = in_df.cix[(j, column_idx)]
                    index_op = DataFrameIndex(indexes=columns, object_type=ObjectType.dataframe)
                    out_chunk = index_op.new_chunk([c], shape=(c.shape[0], len(columns)), index=(j, i),
                                                   dtypes=dtypes, index_value=c.index_value,
                                                   columns_value=parse_index(pd.Index(columns),
                                                                             store_data=True))
                    out_chunks[j].append(out_chunk)
            out_chunks = [item for l in out_chunks for item in l]
            new_op = op.copy()
            nsplits = (in_df.nsplits[0], tuple(column_nsplits))
            return new_op.new_dataframes(op.inputs, shape=out_df.shape, dtypes=out_df.dtypes,
                                         index_value=out_df.index_value,
                                         columns_value=out_df.columns,
                                         chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        df = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = df[op.indexes]


def _getitem(df, item):
    columns = df.columns.to_pandas()
    if isinstance(item, list):
        for col_name in item:
            if col_name not in columns:
                raise KeyError('%s not in columns' % col_name)
        op = DataFrameIndex(indexes=item, object_type=ObjectType.dataframe)
    else:
        if item not in columns:
            raise KeyError('%s not in columns' % item)
        op = DataFrameIndex(indexes=item)
    return op(df)
