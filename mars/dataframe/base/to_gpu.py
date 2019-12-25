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

from ... import opcodes as OperandDef
from ..operands import ObjectType
from .core import DataFrameDeviceConversionBase


class DataFrameToGPU(DataFrameDeviceConversionBase):
    _op_type_ = OperandDef.TO_GPU

    def __init__(self, dtypes=None, gpu=None, sparse=None, object_type=None, **kw):
        super().__init__(_dtypes=dtypes, _gpu=gpu, _sparse=sparse, _object_type=object_type, **kw)
        if not self._gpu:
            self._gpu = True

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

    op = DataFrameToGPU()
    return op(df_or_series)
