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

from ... import opcodes as OperandDef
from ...serialize import AnyField
from .core import TensorNoInput
from ..array_utils import create_array


class Scalar(TensorNoInput):
    """
    Operand represents scalar type.
    """

    _op_type_ = OperandDef.SCALAR

    _data = AnyField('data')

    def __init__(self, data=None, dtype=None, gpu=False, **kw):
        super().__init__(_data=data, _dtype=dtype, _gpu=gpu, **kw)

    @classmethod
    def tile(cls, op):
        chunk_op = op.copy().reset_key()
        chunk = chunk_op.new_chunk(None, shape=(), index=())
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, op.outputs[0].shape,
                                  chunks=[chunk], nsplits=())

    @property
    def data(self):
        return self._data

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        if chunk.ndim != 0:
            raise ValueError('Missing op for chunk')
        ctx[chunk.key] = create_array(op)('asarray', op.data)


def scalar(data, dtype=None, gpu=False):
    try:
        arr = np.array(data, dtype=dtype)
        op = Scalar(arr, dtype=arr.dtype, gpu=gpu)
        shape = ()
        return op(shape)
    except ValueError:
        raise TypeError(f'Expect scalar, got: {data}')
