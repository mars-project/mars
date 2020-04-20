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
from ...core import ExecutableTuple
from ...serialize import ValueType, KeyField, Int64Field, Int32Field, \
    BoolField, StringField, ListField
from ...operands import OperandStage
from ...utils import ceildiv, flatten, recursive_tile
from ..operands import TensorOperand, TensorOperandMixin, TensorOrder
from ..datasource import tensor as astensor
from ..array_utils import as_same_device, device
from ..utils import validate_axis, validate_order
from .sort import _validate_sort_psrs_kinds


class TensorTopk(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.TOPK

    _input = KeyField('input')
    _k = Int64Field('k')
    _axis = Int32Field('axis')
    _largest = BoolField('largest')
    _sorted = BoolField('sorted')
    _order = ListField('order', ValueType.string)
    _parallel_kind = StringField('parallel_kind')
    _psrs_kinds = ListField('psrs_kinds', ValueType.string)
    _return_value = BoolField('return_value')
    _return_indices = BoolField('return_indices')
    _axis_offset = Int64Field('axis_offset')

    def __init__(self, k=None, axis=None, largest=None, sorted=None, order=None,
                 parallel_kind=None, psrs_kinds=None, return_value=None, return_indices=None,
                 axis_offset=None, stage=None, dtype=None, gpu=None, **kw):
        super().__init__(_k=k, _axis=axis, _largest=largest, _sorted=sorted,
                         _parallel_kind=parallel_kind, _psrs_kinds=psrs_kinds,
                         _return_value=return_value, _return_indices=return_indices,
                         _order=order, _axis_offset=axis_offset, _stage=stage,
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
    def order(self):
        return self._order

    @property
    def parallel_kind(self):
        return self._parallel_kind

    @property
    def psrs_kinds(self):
        return self._psrs_kinds

    @property
    def return_value(self):
        return self._return_value

    @property
    def return_indices(self):
        return self._return_indices

    @property
    def axis_offset(self):
        return self._axis_offset

    @property
    def output_limit(self):
        if self._stage != OperandStage.agg:
            return 1
        else:
            return int(bool(self._return_value)) + int(bool(self._return_indices))

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, a):
        shape = list(a.shape)
        shape[self._axis] = min(a.shape[self._axis], self._k)
        kws = []
        if self._return_value:
            kws.append({
                'shape': tuple(shape),
                'order': a.order,
                'dtype': a.dtype,
                'type': 'topk'
            })
        if self._return_indices:
            kws.append({
                'shape': tuple(shape),
                'order': TensorOrder.C_ORDER,
                'dtype': np.dtype(np.int64),
                'type': 'argtopk'
            })
        ret = self.new_tensors([a], kws=kws)
        if len(kws) == 1:
            return ret[0]
        return ExecutableTuple(ret)

    @classmethod
    def _tile_one_chunk(cls, op):
        return_value, return_indices = op.return_value, op.return_indices
        out = op.outputs[0]
        chunk_op = op.copy().reset_key()
        kws = []
        if return_value:
            kws.append({
                'shape': out.shape,
                'order': out.order,
                'index': (0,) * out.ndim,
                'dtype': out.dtype,
                'type': 'topk'
            })
        if return_indices:
            kws.append({
                'shape': out.shape,
                'order': TensorOrder.C_ORDER,
                'index': (0,) * out.ndim,
                'dtype': np.dtype(np.int64),
                'type': 'argtopk'
            })
        chunks = chunk_op.new_chunks([op.input.chunks[0]], kws=kws)
        kws = [out.params for out in op.outputs]
        nsplits = tuple((s,) for s in out.shape)
        if return_value:
            kws[0]['nsplits'] = nsplits
            kws[0]['chunks'] = [chunks[0]]
        if return_indices:
            kws[-1]['nsplits'] = nsplits
            kws[-1]['chunks'] = [chunks[1]]
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, kws=kws)

    @classmethod
    def _tile_via_psrs(cls, op):
        from .sort import TensorSort

        return_value = op.return_value
        return_indices = op.return_indices

        # just sort, force need_align=True
        psrs_kinds = op.psrs_kinds or ['quicksort', 'mergesort', 'mergesort']
        sort_op = TensorSort(axis=op.axis, order=op.order,
                             psrs_kinds=psrs_kinds, need_align=True,
                             return_value=return_value, return_indices=return_indices)
        ret = sort_op(op.input)

        if not isinstance(ret, tuple):
            ret = (ret,)

        base_slcs = (slice(None),) * op.axis
        if op.largest:
            ret = [r[base_slcs + (slice(-1, -op.k - 1, -1),)] for r in ret]
        else:
            ret = [r[base_slcs + (slice(op.k),)] for r in ret]

        ret = [recursive_tile(r) for r in ret]
        new_op = op.copy()
        kws = [o.params for o in op.outputs]
        if return_value:
            kws[0]['nsplits'] = ret[0].nsplits
            kws[0]['chunks'] = ret[0].chunks
        if return_indices:
            kws[-1]['nsplits'] = ret[-1].nsplits
            kws[-1]['chunks'] = ret[-1].chunks
        return new_op.new_tensors(op.inputs, kws=kws)

    @classmethod
    def _gen_topk_chunk(cls, input_chunk, op, is_terminate_node,
                        axis_offset=None, chunk_index=None):
        chunk_op = op.copy().reset_key()
        if axis_offset is not None:
            chunk_op._axis_offset = axis_offset
        if not is_terminate_node:
            # no need to sort if not the terminated node
            chunk_op._sorted = False
        shape = list(input_chunk.shape)
        shape[op.axis] = min(op.k, input_chunk.shape[op.axis])
        if not is_terminate_node:
            # whenever return_indices, value is required
            chunk_op._return_value = True
            if axis_offset is not None:
                chunk_op._stage = OperandStage.map
            else:
                chunk_op._stage = OperandStage.combine
            return chunk_op.new_chunk([input_chunk], shape=tuple(shape),
                                      order=input_chunk.order,
                                      index=chunk_index)
        else:
            chunk_op._stage = OperandStage.agg
            kws = []
            if op.return_value:
                kws.append({
                    'shape': tuple(shape),
                    'order': input_chunk.order,
                    'dtype': input_chunk.dtype,
                    'index': chunk_index,
                    'type': 'topk'
                })
            if op.return_indices:
                kws.append({
                    'shape': tuple(shape),
                    'order': TensorOrder.C_ORDER,
                    'dtype': np.dtype(np.int64),
                    'index': chunk_index,
                    'type': 'argtopk'
                })
            return chunk_op.new_chunks([input_chunk], kws=kws)

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
        return_value, return_indices = op.return_value, op.return_indices
        combine_size = options.combine_size
        axis_offsets = [0] + np.cumsum(a.nsplits[axis]).tolist()[:-1]

        out_chunks, indices_chunks = [], []
        for other_idx in itertools.product(
                *(range(s) for i, s in enumerate(a.chunk_shape) if i != axis)):
            merge_chunks = []
            for j in range(a.chunk_shape[axis]):
                idx = list(other_idx)
                idx.insert(axis, j)
                input_chunk = a.cix[tuple(idx)]
                merge_chunks.append(cls._gen_topk_chunk(input_chunk, op, False,
                                                        axis_offset=axis_offsets[j]))
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
            chunks = \
                cls._gen_topk_chunk(merge_chunk, op, True, chunk_index=tuple(chunk_index))
            if return_value:
                out_chunks.append(chunks[0])
            if return_indices:
                indices_chunks.append(chunks[-1])

        new_op = op.copy()
        nsplits = list(a.nsplits)
        nsplits[axis] = (min(a.shape[axis], op.k),)
        kws = [out.params for out in op.outputs]
        if return_value:
            kws[0]['nsplits'] = nsplits
            kws[0]['chunks'] = out_chunks
        if return_indices:
            kws[-1]['nsplits'] = nsplits
            kws[-1]['chunks'] = indices_chunks
        return new_op.new_tensors(op.inputs, kws=kws)

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
        inputs, device_id, xp = as_same_device(
            flatten([ctx[inp.key] for inp in op.inputs]),
            device=op.device, ret_extra=True)
        if len(inputs) == 2:
            a, indices = inputs
        else:
            a, indices = inputs[0], None

        k = op.k
        axis = op.axis
        to_sort = op.sorted
        largest = op.largest
        return_value = op.return_value
        return_indices = op.return_indices
        axis_offset = op.axis_offset

        with device(device_id):
            av, ap = _topk_helper(xp, a, k, axis=axis, largest=largest, sorted=to_sort,
                                  order=op.order, indices=indices, axis_offset=axis_offset,
                                  return_value=return_value, return_indices=return_indices)
            if op.stage != OperandStage.agg:
                out = [av]
                if op.return_indices:
                    out.append(ap)
                ctx[op.outputs[0].key] = tuple(out)
            else:
                if op.return_value:
                    ctx[op.outputs[0].key] = av
                if op.return_indices:
                    ctx[op.outputs[-1].key] = ap


def _gen_indices(shape, axis, xp):
    ap = xp.swapaxes(xp.empty(shape, dtype=np.int64), axis, -1)
    ap[...] = xp.arange(shape[axis]).reshape((1,) * (ap.ndim - 1) + (-1,))
    return xp.swapaxes(ap, -1, axis)


def _topk_helper(xp, a, k, axis=-1, largest=True, sorted=True, order=None,
                 indices=None, axis_offset=None, return_value=True, return_indices=False):
    size = a.shape[axis]
    base_slc = (slice(None),) * axis
    kw = {}
    if order is not None:
        kw['order'] = order

    ap = None
    if return_indices:
        # do partition
        if largest:
            if k < size:
                length = size - k
                ap = xp.argpartition(a, length, axis=axis, **kw)[
                    base_slc + (slice(-k, None),)]
                av = xp.take_along_axis(a, ap, axis)
                if indices is not None:
                    ap = xp.take_along_axis(indices, ap, axis)
            else:
                av = a
                if indices is not None:
                    ap = indices
                else:
                    ap = _gen_indices(a.shape, axis, xp)
            if sorted:
                # sort then reverse
                ags = xp.argsort(av, axis=axis, **kw)[
                    base_slc + (slice(None, None, -1),)]
                ap = xp.take_along_axis(ap, ags, axis)
                av = xp.take_along_axis(av, ags, axis)
        else:
            if k < size:
                ap = xp.argpartition(a, k, axis=axis, **kw)[
                    base_slc + (slice(k),)]
                av = xp.take_along_axis(a, ap, axis)
                if indices is not None:
                    ap = xp.take_along_axis(indices, ap, axis)
            else:
                av = a
                if indices is not None:
                    ap = indices
                else:
                    ap = _gen_indices(a.shape, axis, xp)
            if sorted:
                ags = xp.argsort(av, axis=axis, **kw)
                ap = xp.take_along_axis(ap, ags, axis)
                av = xp.take_along_axis(av, ags, axis)
        if axis_offset:
            ap = ap + axis_offset
    else:
        assert return_value
        if largest:
            if k < size:
                length = size - k
                av = xp.partition(a, length, axis=axis, **kw)[
                    base_slc + (slice(-k, None),)]
            else:
                av = a
            if sorted:
                # sort then reverse
                av = xp.sort(av, axis=axis, **kw)[
                    base_slc + (slice(None, None, -1),)]
        else:
            if k < size:
                av = xp.partition(a, k, axis=axis, **kw)[
                    base_slc + (slice(k),)]
            else:
                av = a
            if sorted:
                av = xp.sort(av, axis=axis, **kw)

    return av, ap


def _validate_topk_arguments(a, k, axis, largest, sorted, order,
                             parallel_kind, psrs_kinds):
    a = astensor(a)
    if axis is None:
        a = a.flatten()
        axis = 0
    else:
        axis = validate_axis(a.ndim, axis)
    # if a is structure type and order is not None
    order = validate_order(a.dtype, order)
    if parallel_kind.lower() not in {'auto', 'tree', 'psrs'}:
        raise ValueError('`parallel_kind` could only be `auto`, `tree`, or `psrs`')
    # if psrs is chosen, sort will be used,
    # psrs_kinds will be passed into it, so use the validation logic in sort
    psrs_kinds = _validate_sort_psrs_kinds(psrs_kinds)
    return a, k, axis, largest, sorted, order, parallel_kind, psrs_kinds


def topk(a, k, axis=-1, largest=True, sorted=True, order=None, parallel_kind='auto',
         psrs_kinds=None, return_index=False):
    a, k, axis, largest, sorted, order, parallel_kind, psrs_kinds = _validate_topk_arguments(
        a, k, axis, largest, sorted, order, parallel_kind, psrs_kinds)
    op = TensorTopk(k=k, axis=axis, largest=largest, sorted=sorted, order=order,
                    parallel_kind=parallel_kind, psrs_kinds=psrs_kinds,
                    dtype=a.dtype, return_value=True, return_indices=return_index,
                    stage=OperandStage.agg)
    return op(a)
