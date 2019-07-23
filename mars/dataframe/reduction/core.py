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

from ...config import options
from ...compat import reduce
from ..utils import parse_index
from ..operands import DataFrameOperandMixin, ObjectType
from ..merge import DataFrameConcat


class SeriesReductionMixin(DataFrameOperandMixin):
    @classmethod
    def tile(cls, op):
        df = op.outputs[0]
        chunks = op.inputs[0].chunks
        combine_size = options.tensor.combine_size

        if len(chunks) == 1:
            chk = chunks[0]
            out_index = cls._get_reduced_index(chk.index_value.to_pandas(), op.axis, op.level)
            new_chunk_op = op.copy().reset_key()
            chunk = new_chunk_op.new_chunk(chunks, shape=out_index.shape, index=chk.index,
                                           index_value=parse_index(out_index), name=df.name)
            new_op = op.copy()
            nsplits = tuple((s,) for s in chunk.shape)
            return new_op.new_dataframes(op.inputs, df.shape, nsplits=nsplits, chunks=[chunk],
                                         index_value=df.index_value)

        while len(chunks) > 1:
            new_chunks = []
            for i in range(0, len(chunks), combine_size):
                chks = chunks[i: i + combine_size]
                if len(chks) == 1:
                    chk = chks[0]
                else:
                    concat_op = DataFrameConcat(dtype=chks[0].dtype, object_type=ObjectType.series)
                    length = sum([c.shape[0] for c in chks])
                    index = reduce(pd.Index.append, [c.index_value.to_pandas() for c in chks])
                    chk = concat_op.new_chunk(chks, shape=(length,), index=(i,),
                                              index_value=parse_index(index), name=df.name)
                new_index = cls._get_reduced_index(chk.index_value.to_pandas(), op.axis, op.level)
                new_op = op.copy().reset_key()
                new_chunks.append(new_op.new_chunk([chk], shape=new_index.shape, index=(i,),
                                                   index_value=parse_index(new_index), name=df.name))
            chunks = new_chunks

        new_op = op.copy()
        nsplits = tuple((s,) for s in chunks[0].shape)
        return new_op.new_dataframes(op.inputs, df.shape,
                                     nsplits=tuple(tuple(ns) for ns in nsplits),
                                     chunks=chunks, dtype=df.dtype, index_value=df.index_value)

    @classmethod
    def execute(cls, ctx, op):
        kwargs = dict(axis=op.axis, level=op.level, skipna=op.skipna)
        in_df = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = getattr(in_df, getattr(cls, '_func_name'))(**kwargs)

    @classmethod
    def _get_reduced_index(cls, index, axis, level):
        func_name = getattr(cls, '_func_name')
        empty_df = pd.DataFrame(index=index)
        return getattr(empty_df, func_name)(axis=axis, level=level).index

    def __call__(self, series):
        axis = getattr(self, 'axis', None)
        level = getattr(self, 'level', None)

        out_index = self._get_reduced_index(series.index_value.to_pandas(), axis, level)
        return self.new_series([series], shape=(), dtype=series.dtype,
                               index_value=parse_index(out_index), name=series.name)
