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

import numpy as np

from ... import opcodes as OperandDef
from ...config import options
from ...serialize import ValueType, KeyField, Int64Field, Int32Field, \
    BoolField, StringField, ListField
from ...utils import ceildiv
from ..operands import TensorOperand, TensorOperandMixin
from ..datasource import tensor as astensor
from ..array_utils import as_same_device, device
from ..utils import validate_axis


class TensorTopk(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.TOPK

    _input = KeyField('input')
    _k = Int64Field('k')
    _axis = Int32Field('axis')
    _largest = BoolField('largest')
    _sorted = BoolField('sorted')
    _parallel_kind = StringField('parallel_kind')
    _psrs_kinds = ListField('psrs_kinds', ValueType.string)

    def __init__(self, k=None, axis=None, largest=None, sorted=None,
                 parallel_kind=None, psrs_kinds=None, dtype=None,
                 gpu=None, **kw):
        super().__init__(_k=k, _axis=axis, _largest=largest, _sorted=sorted,
                         _parallel_kind=parallel_kind, _psrs_kinds=psrs_kinds,
                         _dtype=dtype, _gpu=gpu, **kw)

    @property
    def input(self):
        return self._input

    @property
    def k(self):
        return self._k

    @property
    def axis(self):
        return self._axis

    @property
    def largest(self):
        return self._largest

    @property
    def sorted(self):
        return self._sorted

    @property
    def parallel_kind(self):
        return self._parallel_kind

    @property
    def psrs_kinds(self):
        return self._psrs_kinds

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, a):
        shape = list(a.shape)
        shape[self._axis] = min(a.shape[self._axis], self._k)
        return self.new_tensor([a], shape=tuple(shape), order=a.order)

    @classmethod
    def _tile_one_chunk(cls, op):
        out = op.outputs[0]
        chunk_op = op.copy().reset_key()
        chunk = chunk_op.new_chunk([op.input.chunks[0]], shape=out.shape,
                                   order=out.order)
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=out.shape, order=out.order,
                                  chunks=[chunk], nsplits=tuple((s,) for s in out.shape))

    @classmethod
    def _tile_via_psrs(cls, op):
        pass

    @classmethod
    def _gen_topk_chunk(cls, input_chunk, op, is_terminate_node, chunk_index=None):
        chunk_op = op.copy().reset_key()
        if not is_terminate_node:
            # no need to sort if not the terminated node
            chunk_op._sorted = False
        shape = list(input_chunk.shape)
        shape[op.axis] = min(op.k, input_chunk.shape[op.axis])
        return chunk_op.new_chunk([input_chunk], shape=tuple(shape),
                                  order=input_chunk.order,
                                  index=chunk_index)

    @classmethod
    def _merge_chunks(cls, input_chunks, axis):
        from ..merge import TensorConcatenate

        if len(input_chunks) == 1:
            return input_chunks[0]

        shape = list(input_chunks[0].shape)
        shape[axis] = sum(c.shape[axis] for c in input_chunks)

        merge_op = TensorConcatenate(axis=axis, dtype=input_chunks[0].dtype)
        return merge_op.new_chunk(input_chunks, shape=tuple(shape),
                                  order=input_chunks[0].order)

    @classmethod
    def _tile_via_tree(cls, op):
        a = op.input
        axis = op.axis
        out = op.outputs[0]
        combine_size = options.combine_size

        out_chunks = []
        for other_idx in itertools.product(
                *(range(s) for i, s in enumerate(a.chunk_shape) if i != axis)):
            merge_chunks = []
            for j in range(a.chunk_shape[axis]):
                idx = list(other_idx)
                idx.insert(axis, j)
                input_chunk = a.cix[tuple(idx)]
                merge_chunks.append(cls._gen_topk_chunk(input_chunk, op, False))
            while len(merge_chunks) > combine_size:
                new_size = ceildiv(len(merge_chunks), combine_size)
                new_merge_chunks = []
                for i in range(new_size):
                    to_merge_chunks = merge_chunks[i * combine_size: (i + 1) * combine_size]
                    merge_chunk = cls._merge_chunks(to_merge_chunks, axis)
                    topk_chunk = cls._gen_topk_chunk(merge_chunk, op, False)
                    new_merge_chunks.append(topk_chunk)
                merge_chunks = new_merge_chunks

            merge_chunk = cls._merge_chunks(merge_chunks, axis)
            chunk_index = list(other_idx)
            chunk_index.insert(axis, 0)
            out_chunks.append(cls._gen_topk_chunk(merge_chunk, op, True,
                                                  chunk_index=tuple(chunk_index)))

        new_op = op.copy()
        nsplits = list(a.nsplits)
        nsplits[axis] = (op.k,)
        return new_op.new_tensors(op.inputs, shape=out.shape, order=out.order,
                                  chunks=out_chunks, nsplits=tuple(nsplits))

    @classmethod
    def tile(cls, op):
        a = op.input
        combine_size = options.combine_size
        k = op.k
        axis = op.axis

        if len(a.chunks) == 1:
            return cls._tile_one_chunk(op)

        parallel_kind = op.parallel_kind.lower()

        if parallel_kind == 'auto':
            nsplit = a.nsplits[axis]
            max_chunk_size = max(nsplit)
            if np.isnan(max_chunk_size):
                # has unknown chunk shape and k > 100 just choose 'psrs'
                parallel_kind = 'psrs' if k > 100 else 'tree'
            else:
                if combine_size * k <= max_chunk_size:
                    # each chunk will have k elements on specified axis,
                    # if combined chunk which generated in the tree reduction
                    # is less than max chunk size, parallel kind `tree` will be adopted
                    parallel_kind = 'tree'
                else:
                    parallel_kind = 'psrs'

        if parallel_kind == 'tree':
            op._parallel_kind = 'tree'
            return cls._tile_via_tree(op)
        else:
            assert parallel_kind == 'psrs'
            op._parallel_kind = 'psrs'
            return cls._tile_via_psrs(op)

    @classmethod
    def execute(cls, ctx, op):
        (a,), device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)
        out = op.outputs[0]
        k = op.k
        axis = op.axis
        to_sort = op.sorted
        largest = op.largest

        with device(device_id):
            size = a.shape[axis]
            base_slc = (slice(None),) * axis

            if k >= size:
                if to_sort:
                    a = xp.sort(a, axis=axis)
                if largest:
                    a = a[base_slc + (slice(None, None, -1),)]
                ctx[out.key] = a
                return

            if largest:
                length = size - k
                a = xp.partition(a, length, axis=axis)[base_slc + (slice(-k, None),)]
                if to_sort:
                    # sort then reverse
                    a = xp.sort(a, axis=axis)[base_slc + (slice(None, None, -1),)]
            else:
                a = xp.partition(a, k, axis=axis)[base_slc + (slice(k),)]
                if to_sort:
                    a = xp.sort(a, axis=axis)

            ctx[out.key] = a


def topk(a, k, axis=-1, largest=True, sorted=True, parallel_kind='auto', psrs_kinds=None):
    a = astensor(a)
    if axis is None:
        a = a.flatten()
        axis = 0
    else:
        axis = validate_axis(a.ndim, axis)
    if parallel_kind.lower() not in {'auto', 'tree', 'psrs'}:
        raise ValueError('`parallel_kind` could only be `auto`, `tree`, or `psrs`')

    op = TensorTopk(k=k, axis=axis, largest=largest, sorted=sorted,
                    parallel_kind=parallel_kind, psrs_kinds=psrs_kinds,
                    dtype=a.dtype)
    return op(a)
