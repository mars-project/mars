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

import itertools
from functools import partial

import numpy as np

from ... import opcodes as OperandDef
from ...core import recursive_tile
from ...core.operand import OperandStage
from ...serialization.serializables import FieldTypes, Int32Field, \
    ListField, StringField, BoolField, AnyField
from ...utils import flatten, stack_back
from ..core import TensorOrder
from ..operands import TensorOperand, TensorMapReduceOperand, \
    TensorShuffleProxy, TensorOperandMixin
from ..array_utils import as_same_device, device, cp


class PSRSOperandMixin:
    @classmethod
    def preprocess(cls, op, in_data=None):
        if in_data is None:
            in_data = op.inputs[0]
        axis_shape = in_data.shape[op.axis]
        axis_chunk_shape = in_data.chunk_shape[op.axis]

        # rechunk to ensure all chunks on axis have rough same size
        has_unknown_shape = False
        for ns in in_data.nsplits:
            if any(np.isnan(s) for s in ns):
                has_unknown_shape = True
                break

        if not has_unknown_shape:
            axis_chunk_shape = min(axis_chunk_shape, int(np.sqrt(axis_shape)))
            if np.isnan(axis_shape) or any(np.isnan(s) for s in in_data.nsplits[op.axis]):
                yield
            chunk_size = int(axis_shape / axis_chunk_shape)
            chunk_sizes = [chunk_size for _ in range(int(axis_shape // chunk_size))]
            if axis_shape % chunk_size > 0:
                chunk_sizes[-1] += axis_shape % chunk_size
            in_data = yield from recursive_tile(
                in_data.rechunk({op.axis: tuple(chunk_sizes)}))
            axis_chunk_shape = in_data.chunk_shape[op.axis]

        left_chunk_shape = in_data.chunk_shape[:op.axis] + in_data.chunk_shape[op.axis + 1:]
        if len(left_chunk_shape) > 0:
            out_idxes = itertools.product(*(range(s) for s in left_chunk_shape))
        else:
            out_idxes = [()]
        # if the size except axis has more than 1, the sorted values on each one may be different
        # another shuffle would be required to make sure each axis except to sort
        # has elements with identical size
        extra_shape = [s for i, s in enumerate(in_data.shape) if i != op.axis]
        if getattr(op, 'need_align', None) is None:
            need_align = bool(np.prod(extra_shape, dtype=int) != 1)
        else:
            need_align = op.need_align

        return in_data, axis_chunk_shape, out_idxes, need_align

    @classmethod
    def local_sort_and_regular_sample(cls, op, in_data, axis_chunk_shape, axis_offsets, out_idx):
        raise NotImplementedError

    @classmethod
    def concat_and_pivot(cls, op, axis_chunk_shape, out_idx, sorted_chunks, sampled_chunks):
        raise NotImplementedError

    @classmethod
    def partition_local_data(cls, op, axis_chunk_shape, sorted_chunks,
                             indices_chunks, concat_pivot_chunk):
        raise NotImplementedError

    @classmethod
    def partition_merge_data(cls, op, need_align, return_value, partition_chunks, proxy_chunk):
        raise NotImplementedError

    @classmethod
    def align_partitions_data(cls, op, out_idx, in_data,
                              partition_sort_chunks, partition_indices_chunks, sort_info_chunks):
        raise NotImplementedError


class TensorPSRSOperandMixin(TensorOperandMixin, PSRSOperandMixin):
    @classmethod
    def local_sort_and_regular_sample(cls, op, in_data, axis_chunk_shape, axis_offsets, out_idx):
        # stage 1: local sort and regular samples collected
        sorted_chunks, indices_chunks, sampled_chunks = [], [], []
        sampled_dtype = np.dtype([(o, in_data.dtype[o]) for o in op.order]) \
            if op.order is not None else in_data.dtype
        for i in range(axis_chunk_shape):
            idx = list(out_idx)
            idx.insert(op.axis, i)
            in_chunk = in_data.cix[tuple(idx)]
            kind = None if op.psrs_kinds is None else op.psrs_kinds[0]
            chunk_op = PSRSSortRegularSample(axis=op.axis, order=op.order, kind=kind,
                                             return_indices=op.return_indices,
                                             n_partition=axis_chunk_shape,
                                             axis_offset=axis_offsets[i],
                                             gpu=op.gpu)
            kws = []
            sort_shape = in_chunk.shape
            kws.append({'shape': sort_shape,
                        'order': in_chunk.order,
                        'dtype': in_chunk.dtype,
                        'index': in_chunk.index,
                        'type': 'sorted'})
            if op.return_indices:
                kws.append({'shape': sort_shape,
                            'order': TensorOrder.C_ORDER,
                            'dtype': np.dtype(np.int64),
                            'index': in_chunk.index,
                            'type': 'argsort'})
            sampled_shape = (axis_chunk_shape,)
            kws.append({'shape': sampled_shape,
                        'order': in_chunk.order,
                        'dtype': sampled_dtype,
                        'index': (i,),
                        'type': 'regular_sampled'})
            chunks = chunk_op.new_chunks([in_chunk], kws=kws, output_limit=len(kws))
            if len(chunks) == 2:
                sort_chunk, sampled_chunk = chunks
                sorted_chunks.append(sort_chunk)
                sampled_chunks.append(sampled_chunk)
            else:
                sort_chunk, indices_chunk, sampled_chunk = chunks
                sorted_chunks.append(sort_chunk)
                indices_chunks.append(indices_chunk)
                sampled_chunks.append(sampled_chunk)

        return sorted_chunks, indices_chunks, sampled_chunks

    @classmethod
    def concat_and_pivot(cls, op, axis_chunk_shape, out_idx, sorted_chunks, sampled_chunks):
        # stage 2: gather and merge samples, choose and broadcast p-1 pivots
        concat_pivot_op = PSRSConcatPivot(axis=op.axis,
                                          order=op.order,
                                          kind=None if op.psrs_kinds is None else op.psrs_kinds[1],
                                          dtype=sampled_chunks[0].dtype,
                                          gpu=op.gpu)
        concat_pivot_shape = \
            sorted_chunks[0].shape[:op.axis] + (axis_chunk_shape - 1,) + \
            sorted_chunks[0].shape[op.axis + 1:]
        concat_pivot_index = out_idx[:op.axis] + (0,) + out_idx[op.axis:]
        concat_pivot_chunk = concat_pivot_op.new_chunk(sampled_chunks,
                                                       shape=concat_pivot_shape,
                                                       index=concat_pivot_index)
        return concat_pivot_chunk

    @classmethod
    def partition_local_data(cls, op, axis_chunk_shape, sorted_chunks,
                             indices_chunks, concat_pivot_chunk):
        # stage 3: Local data is partitioned
        return_value = op.return_value
        return_indices = op.return_indices
        if return_indices:
            # if return indices and psrs_kind[2] is not None
            # value has to be output
            map_return_value = True
        else:
            map_return_value = return_value
        partition_chunks = []
        length = len(sorted_chunks or indices_chunks)
        for i in range(length):
            chunk_inputs = []
            if sorted_chunks:
                chunk_inputs.append(sorted_chunks[i])
            if indices_chunks:
                chunk_inputs.append(indices_chunks[i])
            chunk_inputs.append(concat_pivot_chunk)
            partition_shuffle_map = PSRSShuffle(return_value=map_return_value,
                                                return_indices=return_indices,
                                                stage=OperandStage.map, axis=op.axis,
                                                n_partition=axis_chunk_shape,
                                                input_sorted=op.psrs_kinds[0] is not None,
                                                order=op.order, dtype=chunk_inputs[0].dtype,
                                                gpu=chunk_inputs[0].op.gpu)
            partition_chunk = partition_shuffle_map.new_chunk(chunk_inputs,
                                                              shape=chunk_inputs[0].shape,
                                                              index=chunk_inputs[0].index,
                                                              order=chunk_inputs[0].order)
            partition_chunks.append(partition_chunk)
        return partition_chunks

    @classmethod
    def partition_merge_data(cls, op, need_align, return_value, partition_chunks, proxy_chunk):
        # stage 4: all *ith* classes are gathered and merged
        return_value = return_value if return_value is not None else op.return_value
        return_indices = op.return_indices
        partition_sort_chunks, partition_indices_chunks, sort_info_chunks = [], [], []
        for i, partition_chunk in enumerate(partition_chunks):
            kind = None if op.psrs_kinds is None else op.psrs_kinds[2]
            partition_shuffle_reduce = PSRSShuffle(return_value=return_value,
                                                   return_indices=return_indices,
                                                   stage=OperandStage.reduce,
                                                   axis=op.axis, order=op.order,
                                                   kind=kind,
                                                   reducer_index=(i,),
                                                   dtype=partition_chunk.dtype,
                                                   gpu=partition_chunk.op.gpu,
                                                   need_align=need_align)
            kws = []
            chunk_shape = list(partition_chunk.shape)
            chunk_shape[op.axis] = np.nan
            if return_value:
                kws.append({
                    'shape': tuple(chunk_shape),
                    'order': partition_chunk.order,
                    'index': partition_chunk.index,
                    'dtype': partition_chunk.dtype,
                    'type': 'sorted',
                })
            if return_indices:
                kws.append({
                    'shape': tuple(chunk_shape),
                    'order': TensorOrder.C_ORDER,
                    'index': partition_chunk.index,
                    'dtype': np.dtype(np.int64),
                    'type': 'argsort'
                })
            if need_align:
                s = list(chunk_shape)
                s.pop(op.axis)
                kws.append({
                    'shape': tuple(s),
                    'order': TensorOrder.C_ORDER,
                    'index': partition_chunk.index,
                    'dtype': np.dtype(np.int32),
                    'type': 'sort_info',
                })
            cs = partition_shuffle_reduce.new_chunks([proxy_chunk], kws=kws)
            i = 0
            if return_value:
                partition_sort_chunks.append(cs[0])
                i += 1
            if return_indices:
                partition_indices_chunks.append(cs[i])
            if need_align:
                sort_info_chunks.append(cs[-1])

        return partition_sort_chunks, partition_indices_chunks, sort_info_chunks

    @classmethod
    def align_partitions_data(cls, op, out_idx, in_tensor,
                              partition_sort_chunks, partition_indices_chunks, sort_info_chunks):
        return_value, return_indices = op.return_value, op.return_indices
        align_map_chunks = []
        length = len(partition_sort_chunks or partition_indices_chunks)
        for i in range(length):
            chunk_inputs = []
            if return_value:
                chunk_inputs.append(partition_sort_chunks[i])
            if return_indices:
                chunk_inputs.append(partition_indices_chunks[i])
            chunk_inputs.extend(sort_info_chunks)
            align_map_op = PSRSAlign(return_value=return_value, return_indices=return_indices,
                                     stage=OperandStage.map, axis=op.axis,
                                     output_sizes=list(in_tensor.nsplits[op.axis]),
                                     dtype=chunk_inputs[0].dtype,
                                     gpu=chunk_inputs[0].op.gpu)
            align_map_chunk = align_map_op.new_chunk(chunk_inputs,
                                                     shape=chunk_inputs[0].shape,
                                                     index=chunk_inputs[0].index,
                                                     order=TensorOrder.C_ORDER)
            align_map_chunks.append(align_map_chunk)
        proxy_chunk = TensorShuffleProxy(dtype=align_map_chunks[0].dtype).new_chunk(
            align_map_chunks, shape=())
        align_reduce_value_chunks, align_reduce_indices_chunks = [], []
        for i, align_map_chunk in enumerate(align_map_chunks):
            align_reduce_op = PSRSAlign(return_value=return_value, return_indices=return_indices,
                                        stage=OperandStage.reduce, axis=op.axis,
                                        reducer_index=(i,), dtype=align_map_chunk.dtype,
                                        gpu=align_map_chunk.op.gpu)
            idx = list(out_idx)
            idx.insert(op.axis, i)
            in_chunk = in_tensor.cix[tuple(idx)]
            kws = []
            if return_value:
                kws.append({
                    'shape': in_chunk.shape,
                    'index': in_chunk.index,
                    'order': in_chunk.order,
                    'dtype': in_chunk.dtype,
                    'type': 'sorted'
                })
            if return_indices:
                kws.append({
                    'shape': in_chunk.shape,
                    'index': in_chunk.index,
                    'order': TensorOrder.C_ORDER,
                    'dtype': np.dtype(np.int64),
                    'type': 'argsort'
                })
            align_reduce_chunks = align_reduce_op.new_chunks([proxy_chunk], kws=kws)
            if return_value:
                align_reduce_value_chunks.append(align_reduce_chunks[0])
            if return_indices:
                align_reduce_indices_chunks.append(align_reduce_chunks[-1])

        return align_reduce_value_chunks, align_reduce_indices_chunks


def _sort(a, op, xp, axis=None, kind=None, order=None, inplace=False):
    axis = axis if axis is not None else op.axis
    kind = kind if kind is not None else op.kind
    order = order if order is not None else op.order
    if xp is np:
        method = a.sort if inplace else partial(np.sort, a)
        return method(axis=axis, kind=kind, order=order)
    else:  # pragma: no cover
        # cupy does not support structure type
        assert xp is cp
        assert order is not None
        method = a.sort if inplace else partial(cp.sort, a)
        # cupy does not support kind, thus just ignore it
        return method(axis=axis)


def _argsort(a, op, xp, axis=None, kind=None, order=None):
    axis = axis if axis is not None else op.axis
    kind = kind if kind is not None else op.kind
    order = order if order is not None else op.order
    if xp is np:
        return np.argsort(a, axis=axis, kind=kind, order=order)
    else:  # pragma: no cover
        # cupy does not support structure type
        assert xp is cp
        assert order is not None
        return cp.argsort(a, axis=axis)


class PSRSSortRegularSample(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.PSRS_SORT_REGULAR_SMAPLE

    _axis = Int32Field('axis')
    _order = ListField('order', FieldTypes.string)
    _kind = StringField('kind')
    _return_indices = BoolField('return_indices')
    _n_partition = Int32Field('n_partition')
    _axis_offset = AnyField('axis_offset')

    def __init__(self, axis=None, order=None, kind=None, return_indices=None,
                 n_partition=None, axis_offset=None, dtype=None, gpu=None, **kw):
        super().__init__(_axis=axis, _order=order, _kind=kind, _return_indices=return_indices,
                         _n_partition=n_partition, _axis_offset=axis_offset,
                         _dtype=dtype, _gpu=gpu, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def order(self):
        return self._order

    @property
    def kind(self):
        return self._kind

    @property
    def return_indices(self):
        return self._return_indices

    @property
    def n_partition(self):
        return self._n_partition

    @property
    def axis_offset(self):
        return self._axis_offset

    @property
    def output_limit(self):
        # return sorted tensor, indices(optional) and regular sampled tensor
        return 2 if not self._return_indices else 3

    @classmethod
    def execute(cls, ctx, op):
        (a,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        if len(a) == 0:
            # when chunk is empty, return the empty chunk itself
            ctx[op.outputs[0].key] = ctx[op.outputs[-1].key] = a
            return

        with device(device_id):
            n = op.n_partition
            w = a.shape[op.axis] * 1.0 / (n + 1)
            if not op.return_indices:
                if op.kind is not None:
                    # sort
                    res = ctx[op.outputs[0].key] = _sort(a, op, xp)
                else:
                    # do not sort, prepare for sample by `xp.partition`
                    kth = xp.linspace(max(w - 1, 0), a.shape[op.axis] - 1,
                                      num=n, endpoint=False).astype(int)
                    ctx[op.outputs[0].key] = res = xp.partition(
                        a, kth, axis=op.axis, order=op.order)
            else:
                if op.kind is not None:
                    # argsort
                    indices = _argsort(a, op, xp)
                else:
                    # do not sort, use `xp.argpartition`
                    kth = xp.linspace(max(w - 1, 0), a.shape[op.axis] - 1,
                                      num=n, endpoint=False).astype(int)
                    indices = xp.argpartition(
                        a, kth, axis=op.axis, order=op.order)
                ctx[op.outputs[0].key] = res = xp.take_along_axis(a, indices, op.axis)
                ctx[op.outputs[1].key] = op.axis_offset + indices

            # do regular sample
            if op.order is not None:
                res = res[op.order]
            slc = xp.linspace(max(w - 1, 0), a.shape[op.axis] - 1,
                              num=n, endpoint=False).astype(int)
            slc = (slice(None),) * op.axis + (slc,)
            ctx[op.outputs[-1].key] = res[slc]


class PSRSConcatPivot(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.PSRS_CONCAT_PIVOT

    _axis = Int32Field('axis')
    _order = ListField('order', FieldTypes.string)
    _kind = StringField('kind')

    def __init__(self, axis=None, order=None, kind=None, dtype=None, gpu=None, **kw):
        super().__init__(_axis=axis, _order=order, _kind=kind, _dtype=dtype, _gpu=gpu, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def order(self):
        return self._order

    @property
    def kind(self):
        return self._kind

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs if len(ctx[c.key]) > 0], device=op.device, ret_extra=True)

        with device(device_id):
            a = xp.concatenate(inputs, axis=op.axis)
            p = len(inputs)
            assert a.shape[op.axis] == p * len(op.inputs)

            if op.kind is not None:
                # sort
                _sort(a, op, xp, inplace=True)
            else:
                # prepare for sampling via `partition`
                kth = xp.linspace(p - 1, a.shape[op.axis] - 1,
                                  num=p - 1, endpoint=False).astype(int)
                a.partition(kth, axis=op.axis)

            select = xp.linspace(p - 1, a.shape[op.axis] - 1,
                                 num=len(op.inputs) - 1, endpoint=False).astype(int)
            slc = (slice(None),) * op.axis + (select,)
            ctx[op.outputs[0].key] = a[slc]


class PSRSShuffle(TensorMapReduceOperand, TensorOperandMixin):
    _op_type_ = OperandDef.PSRS_SHUFFLE

    # public
    _return_value = BoolField('return_value')
    _return_indices = BoolField('return_indices')

    # for shuffle map
    _axis = Int32Field('axis')
    _order = ListField('order', FieldTypes.string)
    _n_partition = Int32Field('n_partition')
    _input_sorted = BoolField('input_sorted')

    # for shuffle reduce
    _kind = StringField('kind')
    _need_align = BoolField('need_align')

    def __init__(self, return_value=None, return_indices=None,
                 axis=None, order=None, n_partition=None, input_sorted=None,
                 kind=None, need_align=None, **kw):
        super().__init__(_return_value=return_value, _return_indices=return_indices,
                         _axis=axis, _order=order, _n_partition=n_partition,
                         _input_sorted=input_sorted, _kind=kind,
                         _need_align=need_align, **kw)

    @property
    def return_value(self):
        return self._return_value

    @property
    def return_indices(self):
        return self._return_indices

    @property
    def axis(self):
        return self._axis

    @property
    def order(self):
        return self._order

    @property
    def n_partition(self):
        return self._n_partition

    @property
    def input_sorted(self):
        return self._input_sorted

    @property
    def kind(self):
        return self._kind

    @property
    def need_align(self):
        return self._need_align

    @property
    def output_limit(self):
        if self.stage == OperandStage.map:
            return 1
        else:
            limit = int(bool(self._return_value)) + \
                    int(bool(self._return_indices))
            if self._need_align:
                limit += 1
            return limit

    @classmethod
    def _execute_map(cls, ctx, op):
        return_value = op.return_value
        return_indices = op.return_indices

        inputs, device_id, xp = as_same_device([ctx[c.key] for c in op.inputs],
                                               device=op.device, ret_extra=True)
        out = op.outputs[0]
        a = inputs[0]
        pivots = inputs[-1]
        a_indices = None
        if return_indices:
            a_indices = inputs[-2]

        with device(device_id):
            shape = tuple(s for i, s in enumerate(a.shape) if i != op.axis)
            reduce_outputs = [np.empty(shape, dtype=object) for _ in range(op.n_partition)]
            for idx in itertools.product(*(range(s) for s in shape)):
                slc = list(idx)
                slc.insert(op.axis, slice(None))
                slc = tuple(slc)
                a_1d, pivots_1d = a[slc], pivots[slc]
                a_indices_1d = a_indices[slc] if a_indices is not None else None
                raw_a_1d = a_1d
                if op.order is not None:
                    a_1d = a_1d[op.order]
                if op.input_sorted:
                    # a is sorted already
                    poses = xp.searchsorted(a_1d, pivots_1d, side='right')
                    poses = (None,) + tuple(poses) + (None,)
                    for i in range(op.n_partition):
                        reduce_out = []
                        if return_value:
                            values = raw_a_1d[poses[i]: poses[i + 1]]
                            reduce_out.append(values)
                        if return_indices:
                            indices = a_indices_1d[poses[i]: poses[i + 1]]
                            reduce_out.append(indices)
                        reduce_outputs[i][idx] = tuple(reduce_out)
                else:
                    # a is not sorted, search every element in pivots
                    out_idxes = xp.searchsorted(pivots_1d, a_1d, side='right')
                    for i in range(op.n_partition):
                        cond = out_idxes == i
                        reduce_out = []
                        if return_value:
                            values = raw_a_1d[cond]
                            reduce_out.append(values)
                        if return_indices:
                            indices = a_indices_1d[cond]
                            reduce_out.append(indices)
                        reduce_outputs[i][idx] = tuple(reduce_out)
            for i in range(op.n_partition):
                ctx[out.key, (i,)] = tuple(reduce_outputs[i].ravel())

    @classmethod
    def _execute_reduce(cls, ctx, op: "PSRSShuffle"):
        raw_inputs = list(op.iter_mapper_data(ctx))
        # flatten inputs
        flatten_inputs = flatten(raw_inputs)
        inputs, device_id, xp = as_same_device(flatten_inputs, device=op.device, ret_extra=True)
        # organize back inputs
        inputs = stack_back(inputs, raw_inputs)

        out = op.outputs[0]
        extra_shape = list(out.shape)
        extra_shape.pop(op.axis)

        return_value = op.return_value
        return_indices = op.return_indices

        with device(device_id):
            sort_res = np.empty(len(inputs[0]), dtype=object)
            if extra_shape:
                sort_res = sort_res.reshape(*extra_shape)
            sort_info = np.empty(sort_res.shape, dtype=np.int32)
            it = itertools.count(0)
            for inps in zip(*inputs):
                cur = itertools.count()
                values, indices = None, None
                ret = []
                if return_value or len(inps[0]) == 2:
                    i = next(cur)
                    values = xp.concatenate([inp[i] for inp in inps])
                    if return_value:
                        ret.append(values)
                if return_indices:
                    i = next(cur)
                    indices = xp.concatenate([inp[i] for inp in inps])
                    ret.append(indices)

                if op.kind is not None:
                    # sort only if kind specified
                    if return_indices:
                        # if kind specified and return_indices
                        # values cannot be None
                        assert values is not None
                        values_indices = _argsort(values, op, xp, axis=0)
                        if return_value:
                            xp.take(values, values_indices, out=values)
                        xp.take(indices, values_indices, out=indices)
                    else:
                        _sort(values, op, xp, axis=0, inplace=True)

                j = next(it)
                sort_res.ravel()[j] = ret
                sort_info.ravel()[j] = len(ret[0])

            if not op.need_align:
                assert len(sort_res) == 1
                shape = list(extra_shape)
                shape.insert(op.axis, len(sort_res[0]))
                i = 0
                if return_value:
                    ctx[op.outputs[0].key] = sort_res[0][i]
                    i += 1
                if return_indices:
                    ctx[op.outputs[i].key] = sort_res[0][i]
            else:
                i = 0
                if return_value:
                    ctx[op.outputs[0].key] = tuple(r[0] for r in sort_res.ravel())
                    i += 1
                if return_indices:
                    ctx[op.outputs[i].key] = tuple(r[i] for r in sort_res.ravel())
                ctx[op.outputs[-1].key] = sort_info

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        else:
            cls._execute_reduce(ctx, op)


class PSRSAlign(TensorMapReduceOperand, TensorOperandMixin):
    _op_type_ = OperandDef.PSRS_ALIGN

    _return_value = BoolField('return_value')
    _return_indices = BoolField('return_indices')
    _axis = Int32Field('axis')
    _output_sizes = ListField('output_sizes', FieldTypes.int32)

    def __init__(self, return_value=None, return_indices=None, axis=None,
                 output_sizes=None, **kw):
        super().__init__(_return_value=return_value, _return_indices=return_indices,
                         _axis=axis, _output_sizes=output_sizes, **kw)

    @property
    def return_value(self):
        return self._return_value

    @property
    def return_indices(self):
        return self._return_indices

    @property
    def axis(self):
        return self._axis

    @property
    def output_sizes(self):
        return self._output_sizes

    @property
    def output_limit(self):
        if self.stage == OperandStage.map:
            return 1
        else:
            return int(bool(self._return_value)) + int(bool(self._return_indices))

    @classmethod
    def _execute_map(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)
        sort_res, sort_indices = None, None
        i = 0
        if op.return_value:
            sort_res = inputs[0]
            i += 1
        if op.return_indices:
            sort_indices = inputs[i]
            i += 1
        sort_infos = inputs[i:]
        out = op.outputs[0]

        with device(device_id):
            length = len(sort_res or sort_indices)
            outs = np.empty((len(op.output_sizes), length), dtype=object)
            out_sizes = op.output_sizes
            cum_out_sizes = (0,) + tuple(np.cumsum(out_sizes))
            for i in range(length):
                sort_1d = sort_res[i] if sort_res is not None else None
                indices_1d = sort_indices[i] if sort_indices is not None else None
                sort_lengths = [sort_info.flat[i] for sort_info in sort_infos]
                cum_sort_lengths = (0,) + tuple(np.cumsum(sort_lengths))
                j = out.index[op.axis]
                start_pos = cum_sort_lengths[j]
                end_pos = cum_sort_lengths[j + 1]
                out_idx_start, out_idx_end = np.searchsorted(cum_out_sizes, [start_pos, end_pos])
                out_idx_start = max(out_idx_start - 1, 0)
                for out_idx in range(out_idx_start, out_idx_end):
                    out_start_pos = cum_out_sizes[out_idx]
                    out_end_pos = cum_out_sizes[out_idx + 1]
                    s = max(start_pos, out_start_pos)
                    size = max(min(end_pos, out_end_pos) - s, 0)
                    s = max(0, s - start_pos)
                    ret = []
                    if sort_1d is not None:
                        ret.append(sort_1d[s: s + size])
                    if indices_1d is not None:
                        ret.append(indices_1d[s: s + size])
                    outs[out_idx, i] = tuple(ret)

            for idx in range(len(op.output_sizes)):
                ret = []
                for ar in outs[idx]:
                    if ar is None:
                        item = []
                        if sort_res is not None:
                            item.append(xp.empty((0,), dtype=out.dtype))
                        if sort_indices is not None:
                            item.append(xp.empty((0,), dtype=np.dtype(np.int64)))
                        ret.append(tuple(item))
                    else:
                        ret.append(ar)
                ctx[op.outputs[0].key, (idx,)] = tuple(ret)

    @classmethod
    def _execute_reduce(cls, ctx, op: "PSRSAlign"):
        axis = op.axis
        raw_inputs = list(op.iter_mapper_data(ctx))
        flatten_inputs = flatten(raw_inputs)
        inputs, device_id, xp = as_same_device(flatten_inputs, device=op.device, ret_extra=True)
        inputs = stack_back(flatten_inputs, raw_inputs)

        out = op.outputs[0]
        extra_shape = list(out.shape)
        extra_shape.pop(axis)

        return_value = op.return_value
        return_indices = op.return_indices

        with device(device_id):
            if return_value:
                values_res = xp.empty(out.shape, dtype=out.dtype)
            else:
                values_res = None
            if return_indices:
                indices_res = xp.empty(out.shape, dtype=np.dtype(np.int64))
            else:
                indices_res = None
            it = itertools.product(*(range(s) for i, s in enumerate(out.shape) if i != axis))
            for inps in zip(*inputs):
                slc = list(next(it))
                slc.insert(op.axis, slice(None))
                i = 0
                if return_value:
                    value_concat_1d = xp.concatenate([inp[0] for inp in inps])
                    values_res[tuple(slc)] = value_concat_1d
                    i += 1
                if return_indices:
                    ind_concat_id = xp.concatenate([inp[i] for inp in inps])
                    indices_res[tuple(slc)] = ind_concat_id

            i = 0
            if return_value:
                ctx[op.outputs[0].key] = values_res.astype(values_res.dtype,
                                                           order=op.outputs[0].order.value)
                i += 1
            if return_indices:
                ctx[op.outputs[i].key] = indices_res.astype(indices_res.dtype,
                                                            order=op.outputs[i].order.value)

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        else:
            cls._execute_reduce(ctx, op)
