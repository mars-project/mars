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

from ... import opcodes as OperandDef
from ...serialize import KeyField
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..core import DATAFRAME_TYPE, SERIES_TYPE


class DataFrameToGPU(DataFrameOperand, DataFrameOperandMixin):
    _op_code_ = OperandDef.TO_GPU

    _input = KeyField('input')

    def __init__(self, dtypes=None, gpu=None, sparse=None, object_type=None, **kw):
        super(DataFrameToGPU, self).__init__(_dtypes=dtypes, _gpu=gpu, _sparse=sparse,
                                             _object_type=object_type, **kw)
        if not self._gpu:
            self._gpu = True

    @property
    def input(self):
        return self._input

    def _set_inputs(self, inputs):
        super(DataFrameToGPU, self)._set_inputs(inputs)
        self._input = inputs[0]

    def __call__(self, obj):
        if isinstance(obj, DATAFRAME_TYPE):
            self._object_type = ObjectType.dataframe
            return self.new_dataframe([obj], shape=obj.shape, dtypes=obj.dtypes,
                                      index_value=obj.index_value,
                                      columns_value=obj.columns)
        else:
            assert isinstance(obj, SERIES_TYPE)
            self._object_type = ObjectType.series
            return self.new_series([obj], shape=obj.shape, dtype=obj.dtype,
                                   index_value=obj.index_value, name=obj.name)

    @classmethod
    def tile(cls, op):
        out_chunks = []
        for c in op.input.chunks:
            chunk_op = op.copy().reset_key()
            out_chunk = chunk_op.new_chunk([c], shape=c.shape, index=c.index,
                                           index_value=c.index_value,
                                           columns_value=c.columns,
                                           dtypes=c.dtypes)
            out_chunks.append(out_chunk)

        new_op = op.copy().reset_key()
        out = op.outputs[0]
        return new_op.new_dataframes(op.inputs, shape=out.shape, dtypes=out.dtypes,
                                     index_value=out.index_value, columns_value=out.columns,
                                     chunks=out_chunks, nsplits=out.nsplits)

    @classmethod
    def execute(cls, ctx, op):
        import cudf

        if op.object_type == ObjectType.dataframe:
            ctx[op.outputs[0].key] = cudf.DataFrame.from_pandas(ctx[op.inputs[0].key])
        else:
            ctx[op.outputs[0].key] = cudf.Series.from_pandas(ctx[op.inputs[0].key])


def to_gpu(df_or_series):
    if df_or_series.op.gpu:
        return df_or_series

    op = DataFrameToGPU(dtypes=df_or_series.dtypes)
    return op(df_or_series)
