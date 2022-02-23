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

from ... import opcodes as OperandDef
from ...serialization.serializables import AnyField
from ...utils import has_unknown_shape
from ..core import Tensor
from ..datasource import tensor as astensor
from ..operands import TensorOperand, TensorOperandMixin
from ..utils import calc_sliced_size
from .core import chunk_size_type, get_nsplits, gen_rechunk_infos


class TensorRechunk(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.RECHUNK

    chunk_size = AnyField("chunk_size")

    def __call__(self, tensor: Tensor):
        return self.new_tensor([tensor], tensor.shape, order=tensor.order)

    @classmethod
    def tile(cls, op: "TensorRechunk"):
        from ..indexing.slice import TensorSlice
        from ..merge.concatenate import TensorConcatenate

        if has_unknown_shape(*op.inputs):
            yield

        out = op.outputs[0]
        tensor = astensor(op.inputs[0])
        chunk_size = get_nsplits(tensor, op.chunk_size, tensor.dtype.itemsize)
        if chunk_size == tensor.nsplits:
            return [tensor]

        rechunk_infos = gen_rechunk_infos(tensor, chunk_size)
        out_chunks = []
        for rechunk_info in rechunk_infos:
            chunk_index = rechunk_info.out_index
            shape = rechunk_info.shape
            inp_chunks = rechunk_info.input_chunks
            inp_chunk_slices = rechunk_info.input_slices
            inp_slice_chunks = []
            for inp_chunk, inp_chunk_slice in zip(inp_chunks, inp_chunk_slices):
                if all(slc == slice(None) for slc in inp_chunk_slice):
                    inp_slice_chunks.append(inp_chunk)
                else:
                    slc_chunk = TensorSlice(slices=list(inp_chunk_slice)).new_chunk(
                        [inp_chunk],
                        dtype=inp_chunk.dtype,
                        shape=tuple(
                            calc_sliced_size(s, slc)
                            for s, slc in zip(inp_chunk.shape, inp_chunk_slice)
                        ),
                        index=inp_chunk.index,
                    )
                    inp_slice_chunks.append(slc_chunk)

            if len(inp_slice_chunks) > 1 or inp_slice_chunks[0].index != chunk_index:
                chunk_op = TensorConcatenate()
                out_chunk = chunk_op.new_chunk(
                    inp_slice_chunks,
                    shape=shape,
                    index=chunk_index,
                    dtype=out.dtype,
                    order=out.order,
                )
                out_chunks.append(out_chunk)
            else:
                out_chunks.append(inp_slice_chunks[0])

        new_op = op.copy()
        params = out.params
        params["nsplits"] = chunk_size
        params["chunks"] = out_chunks
        tensor = new_op.new_tileable(op.inputs, kws=[params])

        if op.reassign_worker:
            for c in tensor.chunks:
                c.op.reassign_worker = True

        return [tensor]


def rechunk(
    tensor: Tensor, chunk_size: chunk_size_type, reassign_worker=False
) -> Tensor:
    if not any(np.isnan(s) for s in tensor.shape) and not tensor.is_coarse():
        if not has_unknown_shape(tensor):
            # do client check only when tensor has no unknown shape,
            # otherwise, recalculate chunk_size in `tile`
            chunk_size = get_nsplits(tensor, chunk_size, tensor.dtype.itemsize)
            if chunk_size == tensor.nsplits:
                return tensor

    op = TensorRechunk(
        chunk_size=chunk_size,
        reassign_worker=reassign_worker,
        dtype=tensor.dtype,
        sparse=tensor.issparse(),
    )
    return op(tensor)
