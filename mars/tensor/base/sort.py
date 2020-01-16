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
from functools import partial

import numpy as np

from ... import opcodes as OperandDef
from ...operands import OperandStage
from ...serialize import ValueType, Int32Field, StringField, ListField, BoolField
from ...tiles import NotSupportTile, TilesError
from ...utils import get_shuffle_input_keys_idxes
from ..operands import TensorOperand, TensorOperandMixin, TensorMapReduceOperand, \
    TensorShuffleProxy, TensorOrder
from ..array_utils import as_same_device, device, cp
from ..datasource import tensor as astensor
from ..utils import validate_axis, validate_order


class TensorSort(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.SORT

    _axis = Int32Field('axis')
    _kind = StringField('kind')
    _parallel_kind = StringField('parallel_kind')
    _order = ListField('order', ValueType.string)
    _psrs_kinds = ListField('psrs_kinds', ValueType.string)
    _need_align = BoolField('need_align')

    def __init__(self, axis=None, kind=None, parallel_kind=None, order=None,
                 psrs_kinds=None, need_align=None, dtype=None, gpu=None, **kw):
        super().__init__(_axis=axis, _kind=kind, _parallel_kind=parallel_kind,
                         _order=order, _psrs_kinds=psrs_kinds,
                         _need_align=need_align, _dtype=dtype, _gpu=gpu, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def kind(self):
        return self._kind

    @property
    def parallel_kind(self):
        return self._parallel_kind

    @property
    def order(self):
        return self._order

    @property
    def psrs_kinds(self):
        return self._psrs_kinds

    @property
    def need_align(self):
        return self._need_align

    def __call__(self, a):
        return self.new_tensor([a], shape=a.shape, order=a.order)

    @classmethod
    def tile(cls, op):
        in_tensor = op.inputs[0]

        if in_tensor.chunk_shape[op.axis] == 1:
            out_chunks = []
            for chunk in in_tensor.chunks:
                chunk_op = op.copy().reset_key()
                out_chunk = chunk_op.new_chunk([chunk], shape=chunk.shape,
                                               index=chunk.index, order=chunk.order)
                out_chunks.append(out_chunk)

            new_op = op.copy()
            return new_op.new_tensors([in_tensor], shape=in_tensor.shape, order=in_tensor.order,
                                      chunks=out_chunks, nsplits=in_tensor.nsplits)
        else:
            # use parallel sorting by regular sampling
            return PSRSSorter.tile(op)

    @classmethod
    def execute(cls, ctx, op):
        (a,), device_id, xp = as_same_device([ctx[inp.key] for inp in op.inputs],
                                             device=op.device, ret_extra=True)

        with device(device_id):
            kw = {}
            if op.kind is not None:
                kw['kind'] = op.kind
            if op.order is not None:
                kw['order'] = op.order
            ctx[op.outputs[0].key] = xp.sort(a, axis=op.axis, **kw)


class PSRSSorter(object):
    @classmethod
    def preprocess(cls, op):
        in_tensor = op.inputs[0]
        axis_shape = in_tensor.shape[op.axis]
        axis_chunk_shape = in_tensor.chunk_shape[op.axis]

        # rechunk to ensure all chunks on axis have rough same size
        axis_chunk_shape = min(axis_chunk_shape, int(np.sqrt(axis_shape)))
        if np.isnan(axis_shape) or any(np.isnan(s) for s in in_tensor.nsplits[op.axis]):
            raise TilesError('fail to tile because either the shape of '
                             'input tensor on axis {} has unknown shape or chunk shape'.format(op.axis))
        chunk_size = int(axis_shape / axis_chunk_shape)
        chunk_sizes = [chunk_size for _ in range(int(axis_shape // chunk_size))]
        if axis_shape % chunk_size > 0:
            chunk_sizes[-1] += axis_shape % chunk_size
        in_tensor = in_tensor.rechunk(
            {op.axis: tuple(chunk_sizes)})._inplace_tile()
        axis_chunk_shape = in_tensor.chunk_shape[op.axis]

        left_chunk_shape = in_tensor.chunk_shape[:op.axis] + in_tensor.chunk_shape[op.axis + 1:]
        if len(left_chunk_shape) > 0:
            out_idxes = itertools.product(*(range(s) for s in left_chunk_shape))
        else:
            out_idxes = [()]
        # if the size except axis has more than 1, the sorted values on each one may be different
        # another shuffle would be required to make sure each axis except to sort
        # has elements with identical size
        extra_shape = [s for i, s in enumerate(in_tensor.shape) if i != op.axis]
        if op.need_align is None:
            need_align = bool(np.prod(extra_shape, dtype=int) != 1)
        else:
            need_align = op.need_align

        return in_tensor, axis_chunk_shape, out_idxes, need_align

    @classmethod
    def local_sort_and_regular_sample(cls, op, in_tensor, axis_chunk_shape, out_idx):
        # stage 1: local sort and regular samples collected
        sorted_chunks, sampled_chunks = [], []
        sampled_dtype = np.dtype([(o, in_tensor.dtype[o]) for o in op.order]) \
            if op.order is not None else in_tensor.dtype
        for i in range(axis_chunk_shape):
            idx = list(out_idx)
            idx.insert(op.axis, i)
            in_chunk = in_tensor.cix[tuple(idx)]
            kind = None if op.psrs_kinds is None else op.psrs_kinds[0]
            chunk_op = PSRSSortRegularSample(axis=op.axis, order=op.order, kind=kind,
                                             n_partition=axis_chunk_shape, gpu=op.gpu)
            kws = []
            sort_shape = in_chunk.shape
            kws.append({'shape': sort_shape,
                        'order': in_chunk.order,
                        'dtype': in_chunk.dtype,
                        'index': in_chunk.index,
                        'type': 'sorted'})
            sampled_shape = (axis_chunk_shape,)
            kws.append({'shape': sampled_shape,
                        'order': in_chunk.order,
                        'dtype': sampled_dtype,
                        'index': (i,),
                        'type': 'regular_sampled'})
            sort_chunk, sampled_chunk = chunk_op.new_chunks([in_chunk], kws=kws)
            sorted_chunks.append(sort_chunk)
            sampled_chunks.append(sampled_chunk)

        return sorted_chunks, sampled_chunks

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
    def partition_local_data(cls, op, axis_chunk_shape, sorted_chunks, concat_pivot_chunk):
        # stage 3: Local data is partitioned
        partition_chunks = []
        for sorted_chunk in sorted_chunks:
            partition_shuffle_map = PSRSShuffle(stage=OperandStage.map, axis=op.axis,
                                                n_partition=axis_chunk_shape,
                                                input_sorted=op.psrs_kinds[0] is not None,
                                                order=op.order, dtype=sorted_chunk.dtype,
                                                gpu=sorted_chunk.op.gpu)
            partition_chunk = partition_shuffle_map.new_chunk([sorted_chunk, concat_pivot_chunk],
                                                              shape=sorted_chunk.shape,
                                                              index=sorted_chunk.index,
                                                              order=sorted_chunk.order)
            partition_chunks.append(partition_chunk)
        return partition_chunks

    @classmethod
    def partition_merge_data(cls, op, need_align, partition_chunks, proxy_chunk):
        # stage 4: all *ith* classes are gathered and merged
        partition_sort_chunks, sort_info_chunks = [], []
        for i, partition_chunk in enumerate(partition_chunks):
            kind = None if op.psrs_kinds is None else op.psrs_kinds[2]
            partition_shuffle_reduce = PSRSShuffle(stage=OperandStage.reduce,
                                                   axis=op.axis, order=op.order,
                                                   kind=kind,
                                                   shuffle_key=str(i),
                                                   dtype=partition_chunk.dtype,
                                                   gpu=partition_chunk.op.gpu,
                                                   need_align=need_align)
            kws = []
            chunk_shape = list(partition_chunk.shape)
            chunk_shape[op.axis] = np.nan
            kws.append({
                'shape': tuple(chunk_shape),
                'order': partition_chunk.order,
                'index': partition_chunk.index,
                'dtype': partition_chunk.dtype,
                'type': 'sorted',
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
            partition_sort_chunks.append(cs[0])
            if need_align:
                sort_info_chunks.append(cs[1])

        return partition_sort_chunks, sort_info_chunks

    @classmethod
    def align_partitions_data(cls, op, out_idx, in_tensor, partition_sort_chunks, sort_info_chunks):
        align_map_chunks = []
        for partition_sort_chunk in partition_sort_chunks:
            align_map_op = PSRSAlign(stage=OperandStage.map, axis=op.axis,
                                     output_sizes=list(in_tensor.nsplits[op.axis]),
                                     dtype=partition_sort_chunk.dtype,
                                     gpu=partition_sort_chunk.op.gpu)
            align_map_chunk = align_map_op.new_chunk([partition_sort_chunk] + sort_info_chunks,
                                                     shape=partition_sort_chunk.shape,
                                                     index=partition_sort_chunk.index,
                                                     order=TensorOrder.C_ORDER)
            align_map_chunks.append(align_map_chunk)
        proxy_chunk = TensorShuffleProxy(dtype=align_map_chunks[0].dtype).new_chunk(
            align_map_chunks, shape=())
        align_reduce_chunks = []
        for i, align_map_chunk in enumerate(align_map_chunks):
            align_reduce_op = PSRSAlign(stage=OperandStage.reduce, axis=op.axis,
                                        shuffle_key=str(i), dtype=align_map_chunk.dtype,
                                        gpu=align_map_chunk.op.gpu)
            idx = list(out_idx)
            idx.insert(op.axis, i)
            in_chunk = in_tensor.cix[tuple(idx)]
            align_reduce_chunk = align_reduce_op.new_chunk([proxy_chunk],
                                                           shape=in_chunk.shape,
                                                           index=in_chunk.index,
                                                           order=in_chunk.order)
            align_reduce_chunks.append(align_reduce_chunk)

        return align_reduce_chunks

    @classmethod
    def tile(cls, op):
        """
        Refer to http://csweb.cs.wfu.edu/bigiron/LittleFE-PSRS/build/html/PSRSalgorithm.html
        to see explanation of parallel sorting by regular sampling
        """
        out_tensor = op.outputs[0]
        in_tensor, axis_chunk_shape, out_idxes, need_align = cls.preprocess(op)

        out_chunks = []
        for out_idx in out_idxes:
            # stage 1: local sort and regular samples collected
            sorted_chunks, sampled_chunks = cls.local_sort_and_regular_sample(
                op, in_tensor, axis_chunk_shape, out_idx)

            # stage 2: gather and merge samples, choose and broadcast p-1 pivots
            concat_pivot_chunk = cls.concat_and_pivot(
                op, axis_chunk_shape, out_idx, sorted_chunks, sampled_chunks)

            # stage 3: Local data is partitioned
            partition_chunks = cls.partition_local_data(
                op, axis_chunk_shape, sorted_chunks, concat_pivot_chunk)

            proxy_chunk = TensorShuffleProxy(dtype=partition_chunks[0].dtype).new_chunk(
                partition_chunks, shape=())

            # stage 4: all *ith* classes are gathered and merged
            partition_sort_chunks, sort_info_chunks = cls.partition_merge_data(
                op, need_align, partition_chunks, proxy_chunk)

            if not need_align:
                out_chunks.extend(partition_sort_chunks)
            else:
                align_reduce_chunks = cls.align_partitions_data(
                    op, out_idx, in_tensor, partition_sort_chunks, sort_info_chunks)
                out_chunks.extend(align_reduce_chunks)

        new_op = op.copy()
        nsplits = list(in_tensor.nsplits)
        if not need_align:
            nsplits[op.axis] = (np.nan,) * axis_chunk_shape
        return new_op.new_tensors(op.inputs, shape=out_tensor.shape, order=out_tensor.order,
                                  chunks=out_chunks, nsplits=tuple(nsplits))


class PSRSOperandMixin(TensorOperandMixin):
    @classmethod
    def tile(cls, op):
        raise NotSupportTile('{} is a chunk op'.format(cls))


def _sort(a, op, xp, axis=None, kind=None, order=None, inplace=False):
    axis = axis if axis is not None else op.axis
    kind = kind if kind is not None else op.kind
    order = order if order is not None else op.order
    if xp is np:
        method = a.sort if inplace else partial(np.sort, a)
        return method(axis=axis, kind=kind, order=order)
    else:
        # cupy does not support structure type
        assert xp is cp
        assert order is not None
        method = a.sort if inplace else partial(cp.sort, a)
        # cupy does not support kind, thus just ignore it
        return method(axis=axis)


class PSRSSortRegularSample(TensorOperand, PSRSOperandMixin):
    _op_type_ = OperandDef.PSRS_SORT_REGULAR_SMAPLE

    _axis = Int32Field('axis')
    _order = ListField('order', ValueType.string)
    _kind = StringField('kind')
    _n_partition = Int32Field('n_partition')

    def __init__(self, axis=None, order=None, kind=None, n_partition=None,
                 dtype=None, gpu=None, **kw):
        super().__init__(_axis=axis, _order=order, _kind=kind, _n_partition=n_partition,
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
    def n_partition(self):
        return self._n_partition

    @property
    def output_limit(self):
        # return sorted tensor and regular sampled tensor
        return 2

    @classmethod
    def execute(cls, ctx, op):
        (a,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            n = op.n_partition
            w = int(a.shape[op.axis] // n)
            if op.kind is not None:
                # sort
                res = ctx[op.outputs[0].key] = _sort(a, op, xp)
            else:
                # do not sort, prepare for sample by `xp.partition`
                kth = np.arange(0, n * w, w)
                ctx[op.outputs[0].key] = res = xp.partition(
                    a, kth, axis=op.axis, order=op.order)
            # do regular sample
            if op.order is not None:
                res = res[op.order]
            slc = (slice(None),) * op.axis + (slice(0, n * w, w),)
            ctx[op.outputs[1].key] = res[slc]


class PSRSConcatPivot(TensorOperand, PSRSOperandMixin):
    _op_type_ = OperandDef.PSRS_CONCAT_PIVOT

    _axis = Int32Field('axis')
    _order = ListField('order', ValueType.string)
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
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            a = xp.concatenate(inputs, axis=op.axis)
            p = len(inputs)
            assert a.shape[op.axis] == p ** 2

            if op.kind is not None:
                # sort
                _sort(a, op, xp, inplace=True)
            else:
                # prepare for sampling via `partition`
                kth = xp.arange(p - 1, (p - 1) ** 2 + 1, p - 1)
                a.partition(kth, axis=op.axis)

            select = slice(p - 1, (p - 1) ** 2 + 1, p - 1)
            slc = (slice(None),) * op.axis + (select,)
            ctx[op.outputs[0].key] = a[slc]


class PSRSShuffle(TensorMapReduceOperand, PSRSOperandMixin):
    _op_type_ = OperandDef.PSRS_SHUFFLE

    _axis = Int32Field('axis')
    _order = ListField('order', ValueType.string)
    _n_partition = Int32Field('n_partition')
    _input_sorted = BoolField('input_sorted')

    _kind = StringField('kind')
    _need_align = BoolField('need_align')

    def __init__(self, axis=None, order=None, n_partition=None, input_sorted=None,
                 kind=None, need_align=None, stage=None, shuffle_key=None,
                 dtype=None, gpu=None, **kw):
        super().__init__(_axis=axis, _order=order, _n_partition=n_partition,
                         _input_sorted=input_sorted, _kind=kind, _need_align=need_align,
                         _stage=stage, _shuffle_key=shuffle_key, _dtype=dtype, _gpu=gpu, **kw)

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
        return 1 if not self._need_align else 2

    @classmethod
    def _execute_map(cls, ctx, op):
        (a, pivots), device_id, xp = as_same_device([ctx[c.key] for c in op.inputs],
                                                    device=op.device, ret_extra=True)
        out = op.outputs[0]

        with device(device_id):
            shape = tuple(s for i, s in enumerate(a.shape) if i != op.axis)
            reduce_outputs = [np.empty(shape, dtype=object) for _ in range(op.n_partition)]
            for idx in itertools.product(*(range(s) for s in shape)):
                slc = list(idx)
                slc.insert(op.axis, slice(None))
                slc = tuple(slc)
                a_1d, pivots_1d = a[slc], pivots[slc]
                raw_a_1d = a_1d
                if op.order is not None:
                    a_1d = a_1d[op.order]
                if op.input_sorted:
                    # a is sorted already
                    poses = xp.searchsorted(a_1d, pivots_1d, side='right')
                    poses = (None,) + tuple(poses) + (None,)
                    for i in range(op.n_partition):
                        reduce_outputs[i][idx] = raw_a_1d[poses[i]: poses[i + 1]]
                else:
                    # a is not sorted, search every element in pivots
                    out_idxes = xp.searchsorted(pivots_1d, a_1d, side='right')
                    for i in range(op.n_partition):
                        reduce_outputs[i][idx] = raw_a_1d[out_idxes == i]
            for i in range(op.n_partition):
                ctx[(out.key, str(i))] = tuple(reduce_outputs[i].ravel())

    @classmethod
    def _execute_reduce(cls, ctx, op):
        input_keys, _ = get_shuffle_input_keys_idxes(op.inputs[0])
        raw_inputs = [ctx[(input_key, op.shuffle_key)] for input_key in input_keys]
        flatten_inputs = []
        for raw_input in raw_inputs:
            flatten_inputs.extend(raw_input)
        inputs, device_id, xp = as_same_device(flatten_inputs, device=op.device, ret_extra=True)
        input_len = len(raw_inputs[0])
        inputs = [inputs[i * input_len: (i + 1) * input_len]
                  for i in range(len(raw_inputs))]
        out = op.outputs[0]
        extra_shape = list(out.shape)
        extra_shape.pop(op.axis)

        with device(device_id):
            sort_res = np.empty(len(inputs[0]), dtype=object)
            if extra_shape:
                sort_res = sort_res.reshape(*extra_shape)
            sort_info = np.empty(sort_res.shape, dtype=np.int32)
            it = itertools.count(0)
            for inps in zip(*inputs):
                out = xp.concatenate(inps)
                if op.kind is not None:
                    # skip sort
                    _sort(out, op, xp, axis=0, inplace=True)
                j = next(it)
                sort_res.ravel()[j] = out
                sort_info.ravel()[j] = len(out)

            if not op.need_align:
                assert len(sort_res) == 1
                shape = list(extra_shape)
                shape.insert(op.axis, len(sort_res[0]))
                ctx[op.outputs[0].key] = sort_res[0]
            else:
                ctx[op.outputs[0].key] = tuple(sort_res.ravel())
                ctx[op.outputs[1].key] = sort_info

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        else:
            cls._execute_reduce(ctx, op)


class PSRSAlign(TensorMapReduceOperand, PSRSOperandMixin):
    _op_type_ = OperandDef.PSRS_ALIGN

    _axis = Int32Field('axis')
    _output_sizes = ListField('output_sizes', ValueType.int32)

    def __init__(self, axis=None, output_sizes=None, stage=None, shuffle_key=None,
                 dtype=None, gpu=None, **kw):
        super().__init__(_axis=axis, _output_sizes=output_sizes, _stage=stage,
                         _shuffle_key=shuffle_key, _dtype=dtype, _gpu=gpu, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def output_sizes(self):
        return self._output_sizes

    @classmethod
    def _execute_map(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)
        sort_res, sort_infos = inputs[0], inputs[1:]
        out = op.outputs[0]

        with device(device_id):
            length = len(sort_res)
            outs = np.empty((len(op.output_sizes), length), dtype=object)
            out_sizes = op.output_sizes
            cum_out_sizes = (0,) + tuple(np.cumsum(out_sizes))
            for i, sort_1d in enumerate(sort_res):
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
                    outs[out_idx, i] = sort_1d[s: s + size]

            for idx in range(len(op.output_sizes)):
                ctx[(op.outputs[0].key, str(idx))] = \
                    tuple(ar if ar is not None else xp.empty((0,), dtype=out.dtype)
                          for ar in outs[idx])

    @classmethod
    def _execute_reduce(cls, ctx, op):
        in_chunk = op.inputs[0]
        axis = op.axis
        input_keys, _ = get_shuffle_input_keys_idxes(in_chunk)
        raw_inputs = [ctx[(input_key, op.shuffle_key)] for input_key in input_keys]
        flatten_inputs = []
        for raw_input in raw_inputs:
            flatten_inputs.extend(raw_input)
        inputs, device_id, xp = as_same_device(flatten_inputs, device=op.device, ret_extra=True)
        input_len = len(raw_inputs[0])
        inputs = [inputs[i * input_len: (i + 1) * input_len]
                  for i in range(len(raw_inputs))]
        out = op.outputs[0]
        extra_shape = list(out.shape)
        extra_shape.pop(axis)

        with device(device_id):
            res = xp.empty(out.shape, dtype=out.dtype)
            it = itertools.product(*(range(s) for i, s in enumerate(out.shape) if i != axis))
            for inps in zip(*inputs):
                slc = list(next(it))
                slc.insert(op.axis, slice(None))
                concat_1d = xp.concatenate(inps)
                res[tuple(slc)] = concat_1d
            ctx[out.key] = res.astype(res.dtype, order=out.order.value)

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        else:
            cls._execute_reduce(ctx, op)


_AVAILABLE_KINDS = {'QUICKSORT', 'MERGESORT', 'HEAPSORT', 'STABLE'}


def sort(a, axis=-1, kind=None, parallel_kind=None, psrs_kinds=None, order=None):
    r"""
    Return a sorted copy of a tensor.

    Parameters
    ----------
    a : array_like
        Tensor to be sorted.
    axis : int or None, optional
        Axis along which to sort. If None, the tensor is flattened before
        sorting. The default is -1, which sorts along the last axis.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
        Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
        and 'mergesort' use timsort or radix sort under the covers and, in general,
        the actual implementation will vary with data type. The 'mergesort' option
        is retained for backwards compatibility.
        Note that this argument would not take effect if `a` has more than
        1 chunk on the sorting axis.
    parallel_kind: {'PSRS'}, optional
        Parallel sorting algorithm, for the details, refer to:
        http://csweb.cs.wfu.edu/bigiron/LittleFE-PSRS/build/html/PSRSalgorithm.html
    psrs_kinds: list with 3 elements, optional
        Sorting algorithms during PSRS algorithm.
    order : str or list of str, optional
        When `a` is a tensor with fields defined, this argument specifies
        which fields to compare first, second, etc.  A single field can
        be specified as a string, and not all fields need be specified,
        but unspecified fields will still be used, in the order in which
        they come up in the dtype, to break ties.

    Returns
    -------
    sorted_tensor : Tensor
        Tensor of the same type and shape as `a`.

    See Also
    --------
    Tensor.sort : Method to sort a tensor in-place.
    argsort : Indirect sort.
    lexsort : Indirect stable sort on multiple keys.
    searchsorted : Find elements in a sorted tensor.
    partition : Partial sort.

    Notes
    -----
    The various sorting algorithms are characterized by their average speed,
    worst case performance, work space size, and whether they are stable. A
    stable sort keeps items with the same key in the same relative
    order. The four algorithms implemented in NumPy have the following
    properties:

    =========== ======= ============= ============ ========
       kind      speed   worst case    work space   stable
    =========== ======= ============= ============ ========
    'quicksort'    1     O(n^2)            0          no
    'heapsort'     3     O(n*log(n))       0          no
    'mergesort'    2     O(n*log(n))      ~n/2        yes
    'timsort'      2     O(n*log(n))      ~n/2        yes
    =========== ======= ============= ============ ========

    .. note:: The datatype determines which of 'mergesort' or 'timsort'
       is actually used, even if 'mergesort' is specified. User selection
       at a finer scale is not currently available.

    All the sort algorithms make temporary copies of the data when
    sorting along any but the last axis.  Consequently, sorting along
    the last axis is faster and uses less space than sorting along
    any other axis.

    The sort order for complex numbers is lexicographic. If both the real
    and imaginary parts are non-nan then the order is determined by the
    real parts except when they are equal, in which case the order is
    determined by the imaginary parts.

    quicksort has been changed to an introsort which will switch
    heapsort when it does not make enough progress. This makes its
    worst case O(n*log(n)).

    'stable' automatically choses the best stable sorting algorithm
    for the data type being sorted. It, along with 'mergesort' is
    currently mapped to timsort or radix sort depending on the
    data type. API forward compatibility currently limits the
    ability to select the implementation and it is hardwired for the different
    data types.

    Timsort is added for better performance on already or nearly
    sorted data. On random data timsort is almost identical to
    mergesort. It is now used for stable sort while quicksort is still the
    default sort if none is chosen. For details of timsort, refer to
    `CPython listsort.txt <https://github.com/python/cpython/blob/3.7/Objects/listsort.txt>`_.
    'mergesort' and 'stable' are mapped to radix sort for integer data types. Radix sort is an
    O(n) sort instead of O(n log n).

    Examples
    --------
    >>> import mars.tensor as mt
    >>> a = mt.array([[1,4],[3,1]])
    >>> mt.sort(a).execute()                # sort along the last axis
    array([[1, 4],
           [1, 3]])
    >>> mt.sort(a, axis=None).execute()     # sort the flattened tensor
    array([1, 1, 3, 4])
    >>> mt.sort(a, axis=0).execute()        # sort along the first axis
    array([[1, 1],
           [3, 4]])

    Use the `order` keyword to specify a field to use when sorting a
    structured array:

    >>> dtype = [('name', 'S10'), ('height', float), ('age', int)]
    >>> values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38),
    ...           ('Galahad', 1.7, 38)]
    >>> a = mt.array(values, dtype=dtype)       # create a structured tensor
    >>> mt.sort(a, order='height').execute()                # doctest: +SKIP
    array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41),
           ('Lancelot', 1.8999999999999999, 38)],
          dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])

    Sort by age, then height if ages are equal:

    >>> mt.sort(a, order=['age', 'height']).execute()       # doctest: +SKIP
    array([('Galahad', 1.7, 38), ('Lancelot', 1.8999999999999999, 38),
           ('Arthur', 1.8, 41)],
          dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])
    """
    a = astensor(a)
    if axis is None:
        a = a.flatten()
        axis = 0
    else:
        axis = validate_axis(a.ndim, axis)
    if kind is not None:
        raw_kind = kind
        kind = kind.upper()
        if kind not in _AVAILABLE_KINDS:
            # check kind
            raise ValueError('{} is an unrecognized kind of sort'.format(raw_kind))
    if parallel_kind is not None:
        raw_parallel_kind = parallel_kind
        parallel_kind = parallel_kind.upper()
        if parallel_kind not in {'PSRS'}:
            raise ValueError('{} is an unrecognized kind of '
                             'parallel sort'.format(raw_parallel_kind))
    if psrs_kinds is not None:
        if isinstance(psrs_kinds, (list, tuple)):
            psrs_kinds = list(psrs_kinds)
            if len(psrs_kinds) != 3:
                raise ValueError('psrs_kinds should have 3 elements')
            for i, psrs_kind in enumerate(psrs_kinds):
                if psrs_kind is None:
                    if i < 2:
                        continue
                    else:
                        raise ValueError('3rd element of psrs_kinds '
                                         'should be specified')
                upper_psrs_kind = psrs_kind.upper()
                if upper_psrs_kind not in _AVAILABLE_KINDS:
                    raise ValueError('{} is an unrecognized kind '
                                     'in psrs_kinds'.format(psrs_kind))
        else:
            raise TypeError('psrs_kinds should be list or tuple')
    else:
        psrs_kinds = ['quicksort', 'mergesort', 'mergesort']
    order = validate_order(a.dtype, order)

    op = TensorSort(axis=axis, kind=kind, parallel_kind=parallel_kind, order=order,
                    psrs_kinds=psrs_kinds, dtype=a.dtype, gpu=a.op.gpu)
    return op(a)
