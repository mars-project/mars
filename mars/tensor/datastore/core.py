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

from ..operands import TensorHasInput, TensorOperandMixin


class TensorDataStore(TensorHasInput, TensorOperandMixin):
    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = inputs[0]

    def __call__(self, a, order=None):
        shape = (0,) * a.ndim
        order = a.order if order is None else order
        return self.new_tensor([a], shape, order=order)

    @classmethod
    def _get_out_chunk(cls, op, in_chunk):
        chunk_op = op.copy().reset_key()
        out_chunk_shape = (0,) * in_chunk.ndim
        return chunk_op.new_chunk([in_chunk], out_chunk_shape,
                                  index=in_chunk.index, order=op.outputs[0].order)

    @classmethod
    def _process_out_chunks(cls, op, out_chunks):
        return out_chunks

    @classmethod
    def tile(cls, op):
        in_tensor = op.input

        out_chunks = []
        for chunk in in_tensor.chunks:
            out_chunk = cls._get_out_chunk(op, chunk)
            out_chunks.append(out_chunk)
        out_chunks = cls._process_out_chunks(op, out_chunks)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, op.outputs[0].shape,
                                  chunks=out_chunks,
                                  nsplits=((0,) for _ in range(in_tensor.ndim)))
