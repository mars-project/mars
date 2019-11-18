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
from ...serialize import ValueType, Int32Field, StringField, ListField, BoolField
from ...tiles import NotSupportTile
from ...utils import get_shuffle_input_keys_idxes
from ..operands import TensorOperand, TensorOperandMixin, \
    TensorShuffleMap, TensorShuffleReduce, TensorShuffleProxy, TensorOrder
from ..array_utils import as_same_device, device, cp
from ..datasource import tensor as astensor
from ..utils import validate_axis


class TensorSort(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.SORT

    _axis = Int32Field('axis')
    _kind = StringField('kind')
    _parallel_kind = StringField('parallel_kind')
    _order = ListField('order', ValueType.string)
    _psrs_kinds = ListField('psrs_kinds', ValueType.string)

    def __init__(self, axis=None, kind=None, parallel_kind=None, order=None,
                 psrs_kinds=None, dtype=None, gpu=None, **kw):
        super(TensorSort, self).__init__(_axis=axis, _kind=kind,
                                         _parallel_kind=parallel_kind,
                                         _order=order, _psrs_kinds=psrs_kinds,
                                         _dtype=dtype, _gpu=gpu, **kw)

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
    def tile(cls, op):
        """
        Refer to http://csweb.cs.wfu.edu/bigiron/LittleFE-PSRS/build/html/PSRSalgorithm.html
        to see explanation of parallel sorting by regular sampling
        """
        in_tensor = op.inputs[0]
        out_tensor = op.outputs[0]
        axis_shape = in_tensor.shape[op.axis]
        axis_chunk_shape = in_tensor.chunk_shape[op.axis]
        while axis_chunk_shape ** 2 > axis_shape:
            axis_chunk_shape -= 1
        in_tensor = in_tensor.rechunk(
            {op.axis: int(axis_shape / axis_chunk_shape)}).single_tiles()
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
        need_align = np.prod(extra_shape, dtype=int) != 1

        out_chunks = []
        for out_idx in out_idxes:
            sorted_chunks, sampled_chunks = [], []
            # stage 1: local sort and regular samples collected
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
                sampled_shape = (axis_chunk_shape - 1,)
                if op.order is not None:
                    sampled_dtype = in_chunk.dtype[op.order[0]]
                else:
                    sampled_dtype = in_chunk.dtype
                kws.append({'shape': sampled_shape,
                            'order': in_chunk.order,
                            'dtype': sampled_dtype,
                            'index': (i,),
                            'type': 'regular_sampled'})
                sort_chunk, sampled_chunk = chunk_op.new_chunks([in_chunk], kws=kws)
                sorted_chunks.append(sort_chunk)
                sampled_chunks.append(sampled_chunk)

            # stage 2: gather and merge samples, choose and broadcast p-1 pivots
            concat_pivot_op = PSRSConcatPivot(axis=op.axis,
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

            # stage 3: Local data is partitioned
            partition_chunks = []
            for sorted_chunk in sorted_chunks:
                partition_shuffle_map = PSRSShuffleMap(axis=op.axis, n_partition=axis_chunk_shape,
                                                       order=op.order, dtype=sorted_chunk.dtype,
                                                       gpu=sorted_chunk.op.gpu)
                partition_chunk = partition_shuffle_map.new_chunk([sorted_chunk, concat_pivot_chunk],
                                                                  shape=sorted_chunk.shape,
                                                                  index=sorted_chunk.index,
                                                                  order=sorted_chunk.order)
                partition_chunks.append(partition_chunk)
            proxy_chunk = TensorShuffleProxy(dtype=partition_chunks[0].dtype).new_chunk(
                partition_chunks, shape=())

            # stage 4: all *ith* classes are gathered and merged
            partition_sort_chunks, sort_info_chunks = [], []
            for i, partition_chunk in enumerate(partition_chunks):
                kind = None if op.psrs_kinds is None else op.psrs_kinds[2]
                partition_shuffle_reduce = PSRSShuffleReduce(axis=op.axis, order=op.order,
                                                             kind=kind,
                                                             shuffle_key=str(i),
                                                             gpu=partition_chunk.dtype,
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

            if not need_align:
                out_chunks.extend(partition_sort_chunks)
            else:
                align_map_chunks = []
                for partition_sort_chunk in partition_sort_chunks:
                    align_map_op = PSRSAlignMap(axis=op.axis,
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
                for i, align_map_chunk in enumerate(align_map_chunks):
                    align_reduce_op = PSRSAlignReduce(axis=op.axis, shuffle_key=str(i),
                                                      dtype=align_map_chunk.dtype,
                                                      gpu=align_map_chunk.op.gpu)
                    idx = list(out_idx)
                    idx.insert(op.axis, i)
                    in_chunk = in_tensor.cix[tuple(idx)]
                    align_reduce_chunk = align_reduce_op.new_chunk([proxy_chunk],
                                                                   shape=in_chunk.shape,
                                                                   index=in_chunk.index,
                                                                   order=in_chunk.order)
                    out_chunks.append(align_reduce_chunk)

        new_op = op.copy()
        nsplits = list(in_tensor.nsplits)
        nsplits[op.axis] = (np.nan,) * axis_chunk_shape
        return new_op.new_tensors(op.inputs, shape=out_tensor.shape, order=out_tensor.order,
                                  chunks=out_chunks, nsplits=tuple(nsplits))


class PSRSOperandMixin(TensorOperandMixin):
    @classmethod
    def tile(cls, op):
        raise NotSupportTile('{} is a chunk op'.format(cls))


class PSRSSortRegularSample(TensorOperand, PSRSOperandMixin):
    _op_type_ = OperandDef.PSRS_SORT_REGULAR_SMAPLE

    _axis = Int32Field('axis')
    _order = ListField('order', ValueType.string)
    _kind = StringField('kind')
    _n_partition = Int32Field('n_partition')

    def __init__(self, axis=None, order=None, kind=None, n_partition=None,
                 dtype=None, gpu=None, **kw):
        super(PSRSSortRegularSample, self).__init__(_axis=axis, _order=order,
                                                    _kind=kind, _n_partition=n_partition,
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
    def _sort_cpu(cls, op, a, xp):
        return xp.sort(a, kind='quicksort', order=op.order)

    @classmethod
    def _sort_gpu(cls, op, a, xp):
        return xp.sort(a, axis=op.axis)

    @classmethod
    def execute(cls, ctx, op):
        (a,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            if xp is np:
                sort_res = ctx[op.outputs[0].key] = cls._sort_cpu(op, a, xp)
            else:
                # sparse is not supported for now
                assert xp is cp
                # cupy does not support structured type
                assert op.order is None
                sort_res = ctx[op.outputs[0].key] = cls._sort_gpu(op, a, xp)

            # do regular sample
            n = op.n_partition
            w = int(a.shape[op.axis] // n)
            if op.order is not None:
                # for structure type, only sort the first field
                sort_res = sort_res[op.order[0]]
            slc = (slice(None),) * op.axis + (slice(w, (n - 1) * w + 1, w),)
            ctx[op.outputs[1].key] = sort_res[slc]


def _merge_sort(a, axis, xp, order=None):
    if xp is np:
        a.sort(axis=axis, kind='mergesort', order=order)
    else:
        assert xp is cp
        a.sort(axis=axis)


class PSRSConcatPivot(TensorOperand, PSRSOperandMixin):
    _op_type_ = OperandDef.PSRS_CONCAT_PIVOT

    _axis = Int32Field('axis')
    _kind = StringField('kind')

    def __init__(self, axis=None, kind=None, dtype=None, gpu=None, **kw):
        super(PSRSConcatPivot, self).__init__(_axis=axis, _kind=kind,
                                              _dtype=dtype, _gpu=gpu, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def kind(self):
        return self._kind

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            a = xp.concatenate(inputs, axis=op.axis)
            _merge_sort(a, op.axis, xp)

            p = len(inputs)
            assert a.shape[op.axis] == (p - 1) * p
            select = slice(p - 1, (p - 1) ** 2 + 1, p - 1)
            slc = (slice(None),) * op.axis + (select,)
            ctx[op.outputs[0].key] = a[slc]


class PSRSShuffleMap(TensorShuffleMap, PSRSOperandMixin):
    _op_type_ = OperandDef.PSRS_SHUFFLE_MAP

    _axis = Int32Field('axis')
    _order = ListField('order', ValueType.string)
    _n_partition = Int32Field('n_partition')

    def __init__(self, axis=None, order=None, n_partition=None, dtype=None, gpu=None, **kw):
        super(PSRSShuffleMap, self).__init__(_axis=axis, _order=order,
                                             _n_partition=n_partition,
                                             _dtype=dtype, _gpu=gpu, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def order(self):
        return self._order

    @property
    def n_partition(self):
        return self._n_partition

    @classmethod
    def execute(cls, ctx, op):
        (a, pivots), device_id, xp = as_same_device([ctx[c.key] for c in op.inputs],
                                                    device=op.device, ret_extra=True)
        out = op.outputs[0]

        with device(device_id):
            shape = tuple(s for i, s in enumerate(a.shape) if i != op.axis)
            reduce_outputs = [np.empty(shape, dtype=object) for _ in range(op.n_partition)]
            for idx in itertools.product(*(range(s) for s in shape)):
                a_1d, pivots_1d = a[idx], pivots[idx]
                raw_a_1d = a_1d
                if op.order is not None:
                    a_1d = a_1d[op.order[0]]
                poses = xp.searchsorted(a_1d, pivots_1d, side='right')
                poses = (None,) + tuple(poses) + (None,)
                for i in range(op.n_partition):
                    reduce_outputs[i][idx] = raw_a_1d[poses[i]: poses[i + 1]]
            for i in range(op.n_partition):
                ctx[(out.key, str(i))] = tuple(reduce_outputs[i].ravel())


class PSRSShuffleReduce(TensorShuffleReduce, PSRSOperandMixin):
    _op_type_ = OperandDef.PSRS_SHUFFLE_REDUCE

    _axis = Int32Field('axis')
    _order = ListField('order', ValueType.string)
    _kind = StringField('kind')
    _need_align = BoolField('need_align')

    def __init__(self, axis=None, order=None, kind=None, need_align=None,
                 shuffle_key=None, dtype=None, gpu=None, **kw):
        super(PSRSShuffleReduce, self).__init__(_axis=axis, _order=order,
                                                _kind=kind, _need_align=need_align,
                                                _shuffle_key=shuffle_key,
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
    def need_align(self):
        return self._need_align

    @property
    def output_limit(self):
        return 1 if not self._need_align else 2

    @classmethod
    def execute(cls, ctx, op):
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
                _merge_sort(out, 0, xp, order=op.order)
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


class PSRSAlignMap(TensorShuffleMap, PSRSOperandMixin):
    _op_type_ = OperandDef.PSRS_ALIGN_MAP

    _axis = Int32Field('axis')
    _output_sizes = ListField('output_sizes', ValueType.int32)

    def __init__(self, axis=None, output_sizes=None, dtype=None, gpu=None, **kw):
        super(PSRSAlignMap, self).__init__(_axis=axis, _output_sizes=output_sizes,
                                           _dtype=dtype, _gpu=gpu, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def output_sizes(self):
        return self._output_sizes

    @classmethod
    def execute(cls, ctx, op):
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
                ctx[(op.outputs[0].key, str(idx))] = tuple(ar if ar is not None else xp.empty((0,))
                                                           for ar in outs[idx])


class PSRSAlignReduce(TensorShuffleReduce, PSRSOperandMixin):
    _op_type_ = OperandDef.PSRS_ALIGN_REDUCE

    _axis = Int32Field('axis')

    def __init__(self, axis=None, shuffle_key=None, dtype=None, gpu=None, **kw):
        super(PSRSAlignReduce, self).__init__(_axis=axis, _shuffle_key=shuffle_key,
                                              _dtype=dtype, _gpu=gpu, **kw)

    @property
    def axis(self):
        return self._axis

    @classmethod
    def execute(cls, ctx, op):
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
                res[slc] = concat_1d
            ctx[out.key] = res.astype(None, order=out.order.value)


_AVAILABLE_KINDS = {'QUICKSORT', 'MERGESORT', 'HEAPSORT', 'STABLE'}


def sort(a, axis=-1, kind=None, parallel_kind=None, psrs_kinds=None, order=None):
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
            for psrs_kind in psrs_kinds:
                upper_psrs_kind = psrs_kind.upper()
                if upper_psrs_kind not in _AVAILABLE_KINDS:
                    raise ValueError('{} is an unrecognized kind '
                                     'in psrs_kinds'.format(psrs_kind))
        else:
            raise TypeError('psrs_kinds should be list or tuple')
    # if a is structure type and order is None
    if getattr(a.dtype, 'fields', None) is not None and order is None:
        order = a.dtype.names

    op = TensorSort(axis=axis, kind=kind, parallel_kind=parallel_kind, order=order,
                    psrs_kinds=psrs_kinds, dtype=a.dtype, gpu=a.op.gpu)
    return op(a)
