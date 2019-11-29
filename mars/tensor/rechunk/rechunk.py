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

import itertools

from ...compat import izip, lzip
from ... import opcodes as OperandDef
from ...serialize import KeyField, AnyField, Int32Field, Int64Field
from ..utils import calc_sliced_size
from ..operands import TensorHasInput, TensorOperandMixin
from .core import plan_rechunks, get_nsplits, compute_rechunk_slices


class TensorRechunk(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.RECHUNK

    _input = KeyField('input')
    _chunk_size = AnyField('chunk_size')
    _threshold = Int32Field('threshold')
    _chunk_size_limit = Int64Field('chunk_size_limit')

    def __init__(self, chunk_size=None, threshold=None, chunk_size_limit=None, dtype=None, sparse=False, **kw):
        super(TensorRechunk, self).__init__(_chunk_size=chunk_size, _threshold=threshold,
                                            _chunk_size_limit=chunk_size_limit, _dtype=dtype, _sparse=sparse, **kw)

    @property
    def chunk_size(self):
        return self._chunk_size

    @property
    def threshold(self):
        return self._threshold

    @property
    def chunk_size_limit(self):
        return self._chunk_size_limit

    def _set_inputs(self, inputs):
        super(TensorRechunk, self)._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, tensor):
        return self.new_tensor([tensor], tensor.shape, order=tensor.order)

    @classmethod
    def tile(cls, op):
        new_chunk_size = op.chunk_size
        steps = plan_rechunks(op.inputs[0], new_chunk_size, op.inputs[0].dtype.itemsize,
                              threshold=op.threshold,
                              chunk_size_limit=op.chunk_size_limit)
        tensor = op.outputs[0]
        for c in steps:
            tensor = compute_rechunk(tensor.inputs[0], c)

        return [tensor]


def rechunk(tensor, chunk_size, threshold=None, chunk_size_limit=None):
    chunk_size = get_nsplits(tensor, chunk_size, tensor.dtype.itemsize)
    if chunk_size == tensor.nsplits:
        return tensor

    op = TensorRechunk(chunk_size, threshold, chunk_size_limit,
                       dtype=tensor.dtype, sparse=tensor.issparse())
    return op(tensor)


def compute_rechunk(tensor, chunk_size):
    from ..indexing.slice import TensorSlice
    from ..merge.concatenate import TensorConcatenate

    result_slices = compute_rechunk_slices(tensor, chunk_size)
    result_chunks = []
    idxes = itertools.product(*[range(len(c)) for c in chunk_size])
    chunk_slices = itertools.product(*result_slices)
    chunk_shapes = itertools.product(*chunk_size)
    for idx, chunk_slice, chunk_shape in izip(idxes, chunk_slices, chunk_shapes):
        to_merge = []
        merge_idxes = itertools.product(*[range(len(i)) for i in chunk_slice])
        for merge_idx, index_slices in izip(merge_idxes, itertools.product(*chunk_slice)):
            chunk_index, chunk_slice = lzip(*index_slices)
            old_chunk = tensor.cix[chunk_index]
            merge_chunk_shape = tuple(calc_sliced_size(s, chunk_slice[0]) for s in old_chunk.shape)
            merge_chunk_op = TensorSlice(chunk_slice, dtype=old_chunk.dtype, sparse=old_chunk.op.sparse)
            merge_chunk = merge_chunk_op.new_chunk([old_chunk], shape=merge_chunk_shape,
                                                   index=merge_idx, order=tensor.order)
            to_merge.append(merge_chunk)
        if len(to_merge) == 1:
            chunk_op = to_merge[0].op.copy()
            out_chunk = chunk_op.new_chunk(to_merge[0].op.inputs, shape=chunk_shape,
                                           index=idx, order=tensor.order)
            result_chunks.append(out_chunk)
        else:
            chunk_op = TensorConcatenate(dtype=to_merge[0].dtype, sparse=to_merge[0].op.sparse)
            out_chunk = chunk_op.new_chunk(to_merge, shape=chunk_shape,
                                           index=idx, order=tensor.order)
            result_chunks.append(out_chunk)

    op = TensorRechunk(chunk_size, sparse=tensor.issparse())
    return op.new_tensor([tensor], tensor.shape, dtype=tensor.dtype, order=tensor.order,
                         nsplits=chunk_size, chunks=result_chunks)
