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

from ...serialization.serializables import KeyField
from ..operands import TensorOperand, TensorOperandMixin


class TensorDeviceConversionBase(TensorOperand, TensorOperandMixin):
    _input = KeyField('input')

    @property
    def input(self):
        return self._input

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = inputs[0]

    def __call__(self, tensor):
        return self.new_tensor([tensor], shape=tensor.shape, dtype=tensor.dtype,
                               order=tensor.order)

    @classmethod
    def tile(cls, op):
        out_chunks = []
        for c in op.input.chunks:
            chunk_op = op.copy().reset_key()
            out_chunk = chunk_op.new_chunk([c], **c.params)
            out_chunks.append(out_chunk)

        new_op = op.copy().reset_key()
        out = op.outputs[0]
        return new_op.new_tensors(op.inputs, nsplits=op.input.nsplits,
                                  chunks=out_chunks, **out.params)
