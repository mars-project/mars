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
from ...serialize import ValueType, KeyField, Int32Field, \
    StringField, ListField, BoolField, AnyField
from ...utils import check_chunks_unknown_shape, flatten, stack_back
from ...tiles import TilesError
from ...core import Base, Entity, ExecutableTuple
from ..operands import TensorOperand, TensorShuffleProxy
from ..core import TENSOR_TYPE, CHUNK_TYPE, TensorOrder
from ..array_utils import as_same_device, device
from ..datasource import tensor as astensor
from ..utils import validate_axis, validate_order
from .psrs import TensorPSRSOperandMixin


class ParallelPartitionMixin(TensorPSRSOperandMixin):
    @classmethod
    def calc_paritions_info(cls, op, kth, size, sort_info_chunks):
        # stage5, collect sort infos and calculate partition info for each partitions
        if isinstance(kth, TENSOR_TYPE):
            kth = kth.chunks[0]
            is_kth_input = True
        else:
            is_kth_input = False
        calc_op = CalcPartitionsInfo(kth=kth, size=size,
                                     dtype=np.dtype(np.int32), gpu=op.gpu)
        kws = []
        for i, sort_info_chunk in enumerate(sort_info_chunks):
            kws.append({
                'shape': sort_info_chunk.shape + (len(kth),),
                'order': sort_info_chunk.order,
                'index': sort_info_chunk.index,
                'pos': i
            })
        inputs = list(sort_info_chunks)
        if is_kth_input:
            inputs.insert(0, kth)
        return calc_op.new_chunks(inputs, kws=kws, output_limit=len(kws))

    @classmethod
    def partition_on_merged(cls, op, need_align, partition_merged_chunks,
                            partition_indices_chunks, partition_info_chunks):
        # Stage 6: partition on each partitions
        return_value, return_indices = op.return_value, op.return_indices
        partitioned_chunks, partitioned_indices_chunks = [], []
        for i, partition_merged_chunk, partition_info_chunk in \
                zip(itertools.count(), partition_merged_chunks, partition_info_chunks):
            partition_op = PartitionMerged(
                return_value=return_value, return_indices=return_indices,
                order=op.order, kind=op.kind, need_align=need_align,
                dtype=partition_merged_chunk.dtype, gpu=op.gpu)
            chunk_inputs = []
            kws = []
            if return_value:
                chunk_inputs.append(partition_merged_chunk)
                kws.append({
                    'shape': partition_merged_chunk.shape,
                    'order': partition_merged_chunk.order,
                    'index': partition_merged_chunk.index,
                    'dtype': partition_merged_chunk.dtype,
                    'type': 'partitioned'
                })
            if return_indices:
                if not return_value:
                    # value is required even it's not returned
                    chunk_inputs.append(partition_merged_chunk)
                chunk_inputs.append(partition_indices_chunks[i])
                kws.append({
                    'shape': partition_merged_chunk.shape,
                    'order': TensorOrder.C_ORDER,
                    'index': partition_merged_chunk.index,
                    'dtype': np.dtype(np.int64),
                    'type': 'argpartition'
                })
            chunk_inputs.append(partition_info_chunk)
            partition_chunks = partition_op.new_chunks(chunk_inputs, kws=kws)
            if return_value:
                partitioned_chunks.append(partition_chunks[0])
            if return_indices:
                partitioned_indices_chunks.append(partition_chunks[-1])

        return partitioned_chunks, partitioned_indices_chunks


class TensorPartition(TensorOperand, ParallelPartitionMixin):
    _op_type_ = OperandDef.PARTITION

    _input = KeyField('input')
    _kth = AnyField('kth')
    _axis = Int32Field('axis')
    _kind = StringField('kind')
    _order = ListField('order', ValueType.string)
    _need_align = BoolField('need_align')
    _return_value = BoolField('return_value')
    _return_indices = BoolField('return_indices')

    def __init__(self, kth=None, axis=None, kind=None, order=None,
                 need_align=None, return_value=None, return_indices=None,
                 dtype=None, gpu=None, **kw):
        super().__init__(_kth=kth, _axis=axis, _kind=kind, _order=order,
                         _need_align=need_align, _return_value=return_value,
                         _return_indices=return_indices, _dtype=dtype, _gpu=gpu, **kw)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if len(self._inputs) > 1:
            self._kth = self._inputs[1]

    @property
    def psrs_kinds(self):
        # to keep compatibility with PSRS
        # remember when merging data in PSRSShuffle(reduce),
        # we don't need sort, thus set psrs_kinds[2] to None
        return ['quicksort', 'mergesort', None]

    @property
    def need_align(self):
        return self._need_align

    @property
    def input(self):
        return self._input

    @property
    def kth(self):
        return self._kth

    @property
    def axis(self):
        return self._axis

    @property
    def kind(self):
        return self._kind

    @property
    def order(self):
        return self._order

    @property
    def return_value(self):
        return self._return_value

    @property
    def return_indices(self):
        return self._return_indices

    @property
    def output_limit(self):
        return int(bool(self._return_value)) + int(bool(self._return_indices))

    def __call__(self, a, kth):
        inputs = [a]
        if isinstance(kth, TENSOR_TYPE):
            inputs.append(kth)
        kws = []
        if self._return_value:
            kws.append({
                'shape': a.shape,
                'order': a.order,
                'type': 'sorted',
                'dtype': a.dtype,
            })
        if self._return_indices:
            kws.append({
                'shape': a.shape,
                'order': TensorOrder.C_ORDER,
                'type': 'argsort',
                'dtype': np.dtype(np.int64)
            })
        ret = self.new_tensors(inputs, kws=kws)
        if len(kws) == 1:
            return ret[0]
        return ExecutableTuple(ret)

    @classmethod
    def _tile_psrs(cls, op, kth):
        """
        Approach here would be almost like PSRSSorter, but there are definitely some differences
        Main processes are listed below:
        Stage 1, local sort and regular samples collected
        State 2, gather and merge samples, choose and broadcast p-1 pivots
        Stage 3, Local data is partitioned
        Stage 4: all *ith* classes are gathered and merged, sizes should be calculated as well
        Stage 5: collect sizes from partitions, calculate how to partition given kth
        Stage 6: partition on each partitions
        Stage 7: align if axis is given, and more than 1 dimension
        """
        out_tensor = op.outputs[0]
        return_value, return_indices = op.return_value, op.return_indices
        # preprocess, to make sure chunk shape on axis are approximately same
        in_tensor, axis_chunk_shape, out_idxes, need_align = cls.preprocess(op)
        axis_offsets = [0] + np.cumsum(in_tensor.nsplits[op.axis]).tolist()[:-1]

        out_chunks, out_indices_chunks = [], []
        for out_idx in out_idxes:
            # stage 1: local sort and regular samples collected
            sorted_chunks, indices_chunks, sampled_chunks = \
                cls.local_sort_and_regular_sample(op, in_tensor, axis_chunk_shape,
                                                  axis_offsets, out_idx)

            # stage 2: gather and merge samples, choose and broadcast p-1 pivots
            concat_pivot_chunk = cls.concat_and_pivot(
                op, axis_chunk_shape, out_idx, sorted_chunks, sampled_chunks)

            # stage 3: Local data is partitioned
            partition_chunks = cls.partition_local_data(
                op, axis_chunk_shape, sorted_chunks, indices_chunks, concat_pivot_chunk)

            proxy_chunk = TensorShuffleProxy(dtype=partition_chunks[0].dtype).new_chunk(
                partition_chunks, shape=())

            # stage 4: all *ith* classes are gathered and merged,
            # note that we don't need sort here, op.psrs_kinds[2] is None
            # force need_align=True to get sort info
            partition_merged_chunks, partition_indices_chunks, sort_info_chunks = \
                cls.partition_merge_data(op, True, True, partition_chunks, proxy_chunk)

            # stage5, collect sort infos and calculate partition info for each partitions
            partition_info_chunks = cls.calc_paritions_info(
                op, kth, in_tensor.shape[op.axis], sort_info_chunks)

            # Stage 6: partition on each partitions
            partitioned_chunks, partitioned_indices_chunks = cls.partition_on_merged(
                op, need_align, partition_merged_chunks, partition_indices_chunks, partition_info_chunks)

            if not need_align:
                if return_value:
                    out_chunks.extend(partitioned_chunks)
                if return_indices:
                    out_indices_chunks.extend(partitioned_indices_chunks)
            else:
                align_reduce_chunks, align_reduce_indices_chunks = cls.align_partitions_data(
                    op, out_idx, in_tensor, partitioned_chunks,
                    partitioned_indices_chunks, sort_info_chunks)
                if return_value:
                    out_chunks.extend(align_reduce_chunks)
                if return_indices:
                    out_indices_chunks.extend(align_reduce_indices_chunks)

        new_op = op.copy()
        nsplits = list(in_tensor.nsplits)
        if not need_align:
            nsplits[op.axis] = (np.nan,) * axis_chunk_shape
        kws = []
        if return_value:
            kws.append({
                'shape': out_tensor.shape,
                'order': out_tensor.order,
                'chunks': out_chunks,
                'nsplits': tuple(nsplits),
                'dtype': out_tensor.dtype,
                'type': 'partitioned'
            })
        if return_indices:
            kws.append({
                'shape': out_tensor.shape,
                'order': TensorOrder.C_ORDER,
                'chunks': out_indices_chunks,
                'nsplits': tuple(nsplits),
                'dtype': np.dtype(np.int64),
                'type': 'argpartition'
            })
        return new_op.new_tensors(op.inputs, kws=kws)

    @classmethod
    def tile(cls, op):
        in_tensor = op.input
        if np.isnan(in_tensor.shape[op.axis]):
            raise TilesError('Tensor has unknown shape on axis {}'.format(op.axis))

        kth = op.kth
        if isinstance(op.kth, TENSOR_TYPE):
            # if `kth` is a tensor, make sure no unknown shape
            check_chunks_unknown_shape([kth], TilesError)
            kth = kth.rechunk(kth.shape)._inplace_tile()

        return_value, return_indices = op.return_value, op.return_indices
        if in_tensor.chunk_shape[op.axis] == 1:
            out_chunks, out_indices_chunks = [], []
            for chunk in in_tensor.chunks:
                chunk_op = op.copy().reset_key()
                kws = []
                if return_value:
                    kws.append({
                        'shape': chunk.shape,
                        'index': chunk.index,
                        'order': chunk.order,
                        'dtype': chunk.dtype,
                        'type': 'partitioned'
                    })
                if return_indices:
                    kws.append({
                        'shape': chunk.shape,
                        'index': chunk.index,
                        'order': TensorOrder.C_ORDER,
                        'dtype': np.dtype(np.int64),
                        'type': 'argpartition'
                    })
                chunk_inputs = [chunk]
                if isinstance(kth, TENSOR_TYPE):
                    chunk_inputs.append(kth.chunks[0])
                chunks = chunk_op.new_chunks(chunk_inputs, kws=kws)
                if return_value:
                    out_chunks.append(chunks[0])
                if return_indices:
                    out_indices_chunks.append(chunks[-1])

            new_op = op.copy()
            kws = [out.params for out in op.outputs]
            if return_value:
                kws[0]['nsplits'] = in_tensor.nsplits
                kws[0]['chunks'] = out_chunks
            if return_indices:
                kws[-1]['nsplits'] = in_tensor.nsplits
                kws[-1]['chunks'] = out_indices_chunks
            return new_op.new_tensors([in_tensor], kws=kws)
        else:
            return cls._tile_psrs(op, kth)

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)
        a = inputs[0]
        if len(inputs) == 2:
            kth = inputs[1]
        else:
            kth = op.kth
        return_value, return_indices = op.return_value, op.return_indices

        with device(device_id):
            kw = {}
            if op.kind is not None:
                kw['kind'] = op.kind
            if op.order is not None:
                kw['order'] = op.order

            if return_indices:
                if not return_value:
                    ctx[op.outputs[0].key] = xp.argpartition(a, kth, axis=op.axis, **kw)
                else:
                    argparts = ctx[op.outputs[1].key] = xp.argpartition(a, kth, axis=op.axis, **kw)
                    ctx[op.outputs[0].key] = xp.take_along_axis(a, argparts, op.axis)
            else:
                ctx[op.outputs[0].key] = xp.partition(a, kth, axis=op.axis, **kw)


class CalcPartitionsInfo(TensorOperand, TensorPSRSOperandMixin):
    _op_type_ = OperandDef.CALC_PARTITIONS_INFO

    _kth = AnyField('kth')
    _size = Int32Field('size')

    def __init__(self, kth=None, size=None, dtype=None, gpu=None, **kw):
        super().__init__(_kth=kth, _size=size, _dtype=dtype, _gpu=gpu, **kw)

    @property
    def kth(self):
        return self._kth

    @property
    def size(self):
        return self._size

    @property
    def output_limit(self):
        return np.inf

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if isinstance(self._kth, (Base, Entity)):
            self._kth = self._inputs[0]

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            if isinstance(op.kth, CHUNK_TYPE):
                kth = inputs[0]
                sort_infos = inputs[1:]
                # make kth all positive
                kth = _validate_kth_value(kth, op.size)
            else:
                kth = op.kth
                sort_infos = inputs

            sort_info_shape = sort_infos[0].shape
            # create arrays filled with -1, -1 means do nothing about partition
            partition_infos = [xp.full(sort_info_shape + (len(kth),), -1) for _ in sort_infos]
            concat_sort_info = xp.stack([sort_info.ravel() for sort_info in sort_infos])
            cumsum_sort_info = xp.cumsum(concat_sort_info, axis=0)

            for j in range(cumsum_sort_info.shape[1]):
                idx = xp.unravel_index(j, sort_infos[0].shape)
                sizes = cumsum_sort_info[:, j]
                to_partition_chunk_idxes = xp.searchsorted(sizes, kth, side='right')
                for i, to_partition_chunk_idx in enumerate(to_partition_chunk_idxes):
                    partition_idx = tuple(idx) + (i,)
                    k = kth[i]
                    # if to partition on chunk 0, just set to kth
                    # else kth - {size of previous chunks}
                    chunk_k = k if to_partition_chunk_idx == 0 else \
                        k - sizes[to_partition_chunk_idx - 1]
                    partition_infos[to_partition_chunk_idx][partition_idx] = chunk_k

            for out, partition_info in zip(op.outputs, partition_infos):
                ctx[out.key] = partition_info


class PartitionMerged(TensorOperand, TensorPSRSOperandMixin):
    _op_type_ = OperandDef.PARTITION_MERGED

    _return_value = BoolField('return_value')
    _return_indices = BoolField('return_indices')
    _order = ListField('order', ValueType.string)
    _kind = StringField('kind')
    _need_align = BoolField('need_align')

    def __init__(self, return_value=None, return_indices=None, order=None, kind=None,
                 need_align=None, dtype=None, gpu=None, **kw):
        super().__init__(_return_value=return_value, _return_indices=return_indices,
                         _order=order, _kind=kind, _need_align=need_align,
                         _dtype=dtype, _gpu=gpu, **kw)

    @property
    def return_value(self):
        return self._return_value

    @property
    def return_indices(self):
        return self._return_indices

    @property
    def order(self):
        return self._order

    @property
    def kind(self):
        return self._kind

    @property
    def need_align(self):
        return self._need_align

    @property
    def output_limit(self):
        return int(bool(self._return_value)) + int(bool(self._return_indices))

    @classmethod
    def execute(cls, ctx, op):
        return_value, return_indices = op.return_value, op.return_indices

        raw_inputs = [ctx[inp.key] for inp in op.inputs]
        flatten_inputs = flatten(raw_inputs)
        inputs, device_id, xp = as_same_device(flatten_inputs, device=op.device, ret_extra=True)
        inputs = stack_back(inputs, raw_inputs)
        partition_info = inputs[-1]
        merged_data, merged_indices = None, None
        if return_value:
            merged_data = inputs[0]
        if return_indices:
            # if return indices, value should be returned
            assert len(inputs) == 3
            if not return_value:
                merged_data = inputs[0]
            merged_indices = inputs[1]

        outs, out_indices = [], []
        with device(device_id):
            kw = {}
            if op.kind is not None:
                kw['kind'] = op.kind
            if op.order is not None:
                kw['order'] = op.order

            ravel_partition_info = partition_info.reshape(-1, partition_info.shape[-1])
            for i, merged_vec, kth in zip(itertools.count(), merged_data, ravel_partition_info):
                kth = kth[kth > -1]
                if kth.size == 0:
                    if return_value:
                        outs.append(merged_vec)
                    if return_indices:
                        out_indices.append(merged_indices[i])
                else:
                    if return_indices:
                        argparts = xp.argpartition(merged_vec, kth, **kw)
                        if return_value:
                            outs.append(xp.take(merged_vec, argparts))
                        out_indices.append(xp.take(merged_indices[i], argparts))
                    else:
                        outs.append(xp.partition(merged_vec, kth, **kw))

        if not op.need_align:
            assert len(outs or out_indices) == 1
            i = 0
            if return_value:
                ctx[op.outputs[0].key] = outs[0]
                i += 1
            if return_indices:
                ctx[op.outputs[i].key] = out_indices[0]
        else:
            i = 0
            if return_value:
                ctx[op.outputs[0].key] = tuple(outs)
                i += 1
            if return_indices:
                ctx[op.outputs[i].key] = tuple(out_indices)


def _check_kth_dtype(dtype):
    if not np.issubdtype(dtype, np.integer):
        raise TypeError('Partition index must be integer')


def _validate_kth_value(kth, size):
    kth = np.where(kth < 0, kth + size, kth)
    if np.any((kth < 0) | (kth >= size)):
        invalid_kth = next(k for k in kth if k < 0 or k >= size)
        raise ValueError('kth(={}) out of bounds ({})'.format(
            invalid_kth, size))
    return kth


def _validate_partition_arguments(a, kth, axis, kind, order, kw):
    a = astensor(a)
    if axis is None:
        a = a.flatten()
        axis = 0
    else:
        axis = validate_axis(a.ndim, axis)
    if isinstance(kth, (Base, Entity)):
        kth = astensor(kth)
        _check_kth_dtype(kth.dtype)
    else:
        kth = np.atleast_1d(kth)
        kth = _validate_kth_value(kth, a.shape[axis])
    if kth.ndim > 1:
        raise ValueError('object too deep for desired array')
    if kind != 'introselect':
        raise ValueError('{} is an unrecognized kind of select'.format(kind))
    # if a is structure type and order is not None
    order = validate_order(a.dtype, order)
    need_align = kw.pop('need_align', None)
    if len(kw) > 0:
        raise TypeError('partition() got an unexpected keyword '
                        'argument \'{}\''.format(next(iter(kw))))

    return a, kth, axis, kind, order, need_align


def partition(a, kth, axis=-1, kind='introselect', order=None, **kw):
    r"""
    Return a partitioned copy of a tensor.

    Creates a copy of the tensor with its elements rearranged in such a
    way that the value of the element in k-th position is in the
    position it would be in a sorted tensor. All elements smaller than
    the k-th element are moved before this element and all equal or
    greater are moved behind it. The ordering of the elements in the two
    partitions is undefined.

    Parameters
    ----------
    a : array_like
        Tensor to be sorted.
    kth : int or sequence of ints
        Element index to partition by. The k-th value of the element
        will be in its final sorted position and all smaller elements
        will be moved before it and all equal or greater elements behind
        it. The order of all elements in the partitions is undefined. If
        provided with a sequence of k-th it will partition all elements
        indexed by k-th  of them into their sorted position at once.
    axis : int or None, optional
        Axis along which to sort. If None, the tensor is flattened before
        sorting. The default is -1, which sorts along the last axis.
    kind : {'introselect'}, optional
        Selection algorithm. Default is 'introselect'.
    order : str or list of str, optional
        When `a` is a tensor with fields defined, this argument
        specifies which fields to compare first, second, etc.  A single
        field can be specified as a string.  Not all fields need be
        specified, but unspecified fields will still be used, in the
        order in which they come up in the dtype, to break ties.

    Returns
    -------
    partitioned_tensor : Tensor
        Tensor of the same type and shape as `a`.

    See Also
    --------
    Tensor.partition : Method to sort a tensor in-place.
    argpartition : Indirect partition.
    sort : Full sorting

    Notes
    -----
    The various selection algorithms are characterized by their average
    speed, worst case performance, work space size, and whether they are
    stable. A stable sort keeps items with the same key in the same
    relative order. The available algorithms have the following
    properties:

    ================= ======= ============= ============ =======
       kind            speed   worst case    work space  stable
    ================= ======= ============= ============ =======
    'introselect'        1        O(n)           0         no
    ================= ======= ============= ============ =======

    All the partition algorithms make temporary copies of the data when
    partitioning along any but the last axis.  Consequently,
    partitioning along the last axis is faster and uses less space than
    partitioning along any other axis.

    The sort order for complex numbers is lexicographic. If both the
    real and imaginary parts are non-nan then the order is determined by
    the real parts except when they are equal, in which case the order
    is determined by the imaginary parts.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> a = mt.array([3, 4, 2, 1])
    >>> mt.partition(a, 3).execute()
    array([2, 1, 3, 4])

    >>> mt.partition(a, (1, 3)).execute()
    array([1, 2, 3, 4])
    """
    return_indices = kw.pop('return_index', False)
    a, kth, axis, kind, order, need_align = \
        _validate_partition_arguments(a, kth, axis, kind, order, kw)
    op = TensorPartition(kth=kth, axis=axis, kind=kind, order=order,
                         need_align=need_align, return_value=True,
                         return_indices=return_indices,
                         dtype=a.dtype, gpu=a.op.gpu)
    return op(a, kth)
