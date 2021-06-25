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

from ... import opcodes as OperandDef
from ..array_utils import move_to_device
from ..datasource import tensor as astensor
from .core import TensorDeviceConversionBase


class TensorToGPU(TensorDeviceConversionBase):
    _op_type_ = OperandDef.TO_GPU

    def __init__(self, dtype=None, gpu=None, sparse=None, **kw):
        super().__init__(_dtype=dtype, _gpu=gpu, _sparse=sparse, **kw)
        if not self.gpu:
            self.gpu = True

    @classmethod
    def execute(cls, ctx, op):
        device = op.device or 0
        ctx[op.outputs[0].key] = move_to_device(ctx[op.input.key], device)


def to_gpu(x):
    x = astensor(x)

    if x.op.gpu:
        return x

    op = TensorToGPU(dtype=x.dtype)
    return op(x)
