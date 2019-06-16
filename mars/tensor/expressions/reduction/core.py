#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import copy
import itertools
from collections import Iterable
from math import ceil, log

from ....compat import lrange, izip, irange, builtins, six
from ....config import options
from ...core import Tensor
from ..utils import check_out_param, validate_axis
from ..core import TensorOperandMixin
from ..datasource import tensor as astensor


class TensorReduction(TensorOperandMixin):
    __slots__ = ()

    @classmethod
    def _is_cum(cls):
        return False

    def _call(self, a, out):
        a = astensor(a)
        if out is not None and not isinstance(out, Tensor):
            raise TypeError('out should be Tensor object, got {0} instead'.format(type(out)))

        axis = getattr(self, 'axis', None)
        keepdims = getattr(self, 'keepdims', None)

        if self._is_cum():
            if axis is None:
                a, axis = a.ravel(), 0
                setattr(self, 'axis', axis)
            shape = a.shape
        else:
            axis = lrange(len(a.shape)) if axis is None else axis
            if not isinstance(axis, Iterable):
                axis = (validate_axis(a.ndim, axis),)
            axis = set(axis)

            shape = tuple(s if i not in axis else 1 for i, s in enumerate(a.shape)
                          if keepdims or i not in axis)

        t = self.new_tensor([a], shape)

        if out is None:
            return t

        check_out_param(out, t, 'same_kind')
        out_shape, out_dtype = out.shape, out.dtype
        # if `out` is specified, use out's dtype and shape
        if out_shape != t.shape:
            if out.ndim > t.ndim:
                raise ValueError('output has too many dimensions')
            raise ValueError('output shape should be {0}, got {1}'.format(t.shape, out_shape))

        setattr(self, '_dtype', out_dtype)

        out.data = t.data
        return out

    def new_chunks(self, inputs, shape, **kw):
        chunks = super(TensorReduction, self).new_chunks(inputs, shape, **kw)
        setattr(self, '_input', getattr(self, '_inputs')[0])
        return chunks

    def new_tensors(self, inputs, shape, **kw):
        tensors = super(TensorReduction, self).new_tensors(inputs, shape, **kw)
        setattr(self, '_input', getattr(self, '_inputs')[0])
        return tensors

    def __call__(self, a, out=None):
        return self._call(a, out=out)

    @staticmethod
    def _reduced_shape(shape, axes):
        return tuple(1 if i in axes else s for i, s in enumerate(shape))

    @staticmethod
    def _reduced_nsplits(nsplits, axes):
        return tuple((1,) * len(c) if i in axes else c
                     for i, c in enumerate(nsplits))

    @staticmethod
    def _concatenate_shape(tensor, combine_block):
        return tuple(builtins.sum(nsplit[i] for i in cb)
                     for nsplit, cb in izip(tensor.nsplits, combine_block))

    @staticmethod
    def _combine_split(ax, combine_size, chunk_shape):
        if ax not in combine_size:
            return tuple((i,) for i in range(chunk_shape[ax]))
        else:
            size = combine_size[ax]
            shape = chunk_shape[ax]
            index = tuple(range(shape))
            return tuple(index[i:i + size] for i in irange(0, shape, size))

    def _get_op_kw(self):
        return None

    @staticmethod
    def _get_op_types():
        raise NotImplementedError

    @classmethod
    def _tree_reduction(cls, op, tensor, agg_op_type, axis, combine_op_type=None):
        kw = getattr(op, '_get_op_kw')() or {}
        keepdims = op.keepdims
        combine_size = op.combine_size or options.tensor.combine_size
        if isinstance(combine_size, dict):
            combine_size = dict((ax, combine_size.get(ax)) for ax in axis)
        else:
            assert isinstance(combine_size, six.integer_types)
            n = builtins.max(int(combine_size ** (1.0 / (len(axis) or 1))), 2)
            combine_size = dict((ax, n) for ax in axis)

        times = 1
        for i, n in enumerate(tensor.chunk_shape):
            if i in combine_size and combine_size[i] != 1:
                times = int(builtins.max(times, ceil(log(n, combine_size[i]))))

        for i in range(times - 1):
            combine_op = combine_op_type or agg_op_type
            [tensor] = cls._partial_reduction(combine_op, tensor, axis, op.dtype, True, combine_size)

        return cls._partial_reduction(agg_op_type, tensor, axis, op.dtype, keepdims, combine_size, kw)

    @classmethod
    def _partial_reduction(cls, agg_op_type, tensor, axis, dtype, keepdims, combine_size, kw=None):
        from ..merge.concatenate import TensorConcatenate
        kw = kw or {}
        axes = sorted(combine_size.keys())

        combine_blocks = [cls._combine_split(i, combine_size, tensor.chunk_shape)
                          for i in range(tensor.ndim)]
        combine_blocks_idxes = [range(len(blocks)) for blocks in combine_blocks]

        chunks = []
        for combine_block_idx, combine_block in izip(itertools.product(*combine_blocks_idxes),
                                                     itertools.product(*combine_blocks)):
            chks = [tensor.cix[idx] for idx in itertools.product(*combine_block)]
            if len(chks) > 1:
                op = TensorConcatenate(axis=axes, dtype=chks[0].dtype)
                chk = op.new_chunk(chks, shape=cls._concatenate_shape(tensor, combine_block))
            else:
                chk = chks[0]
            shape = tuple(s if i not in combine_size else 1
                          for i, s in enumerate(chk.shape) if keepdims or i not in combine_size)
            agg_op = agg_op_type(axis=axis, dtype=dtype, keepdims=keepdims, **kw)
            chunk = agg_op.new_chunk([chk], shape=shape,
                                     index=tuple(idx for i, idx in enumerate(combine_block_idx)
                                                 if keepdims or i not in combine_size))
            chunks.append(chunk)

        nsplits = [
            tuple(c.shape[i] for c in chunks if builtins.all(idx == 0 for j, idx in enumerate(c.index) if j != i))
            for i in range(len(chunks[0].shape))]
        shape = tuple(builtins.sum(nsplit) for nsplit in nsplits)
        agg_op = agg_op_type(axis=axis, dtype=dtype, keepdims=keepdims, combine_size=combine_size, **kw)
        return agg_op.new_tensors([tensor], shape, chunks=chunks, nsplits=nsplits)

    @classmethod
    def tile(cls, op):
        in_tensor = op.inputs[0]
        axis = tuple(range(in_tensor.ndim)) if op.axis is None else op.axis
        if isinstance(axis, six.integer_types):
            axis = (axis,)
        axis = tuple(validate_axis(in_tensor.ndim, ax) for ax in axis)

        if len(in_tensor.chunks) == 1:
            c = in_tensor.chunks[0]
            new_op = op.copy().reset_key()
            setattr(new_op, '_axis', axis)
            shape = list(cls._reduced_shape(c.shape, axis))
            nsplits = list(cls._reduced_nsplits(in_tensor.nsplits, axis))
            if not op.keepdims and axis:
                for ax in axis:
                    shape[ax] = None
                    nsplits[ax] = None
            shape = tuple(s for s in shape if s is not None)
            nsplits = tuple(ns for ns in nsplits if ns is not None)

            chunks = new_op.new_chunks([c], shape, index=c.index)
            return op.copy().new_tensors(op.inputs, op.outputs[0].shape, chunks=chunks, nsplits=nsplits)

        chunk_op_type, agg_op_type, combine_op_type = getattr(op, '_get_op_types')()

        chunks = []
        kw = getattr(op, '_get_op_kw')() or {}
        for c in in_tensor.chunks:
            chunk_op = chunk_op_type(axis=axis, dtype=op.dtype, keepdims=True,
                                     combine_size=op.combine_size, **kw)
            chunks.append(chunk_op.new_chunk([c], cls._reduced_shape(c.shape, axis), index=c.index))

        new_op = op.copy()
        tensor = new_op.new_tensor(op.inputs, cls._reduced_shape(in_tensor.shape, axis),
                                   nsplits=cls._reduced_nsplits(in_tensor.nsplits, axis), chunks=chunks)
        return cls._tree_reduction(new_op, tensor, agg_op_type, axis, combine_op_type=combine_op_type)


class TensorArgReduction(TensorReduction):
    __slots__ = ()

    @staticmethod
    def _get_arg_axis(axis, ndim):
        if axis is None:
            axis = tuple(range(ndim))
            ravel = True
        elif isinstance(axis, six.integer_types):
            axis = validate_axis(ndim, axis)
            axis = (axis,)
            ravel = ndim == 1
        else:
            raise TypeError("axis must be either `None` or int, "
                            "got '{0}'".format(axis))
        return axis, ravel

    @staticmethod
    def _get_offset(tensor, axis, chunk, ravel):
        nsplits = tensor.nsplits
        offset = tuple(builtins.sum(split[:idx]) for split, idx in zip(nsplits, chunk.index))
        if not ravel:
            offset = offset[axis[0]]
        return offset

    @classmethod
    def tile(cls, op):
        in_tensor = op.inputs[0]
        axis, ravel = cls._get_arg_axis(op.axis, in_tensor.ndim)
        chunk_op_type, agg_op_type, combine_op_type = getattr(op, '_get_op_types')()

        chunks = []
        for c in in_tensor.chunks:
            offset = cls._get_offset(in_tensor, axis, c, ravel)
            chunk_op = chunk_op_type(axis=axis, dtype=op.dtype, keepdims=True,
                                     offset=offset, total_shape=in_tensor.shape,
                                     combine_size=op.combine_size)
            chunk = chunk_op.new_chunk([c], cls._reduced_shape(c.shape, axis), index=c.index)
            chunks.append(chunk)
        new_op = op.copy()
        tensor = new_op.new_tensor(op.inputs, cls._reduced_shape(in_tensor.shape, axis),
                                   nsplits=cls._reduced_nsplits(in_tensor.nsplits, axis), chunks=chunks)
        return cls._tree_reduction(new_op, tensor, agg_op_type, axis, combine_op_type=combine_op_type)


class TensorCumReduction(TensorReduction):
    __slots__ = ()

    @classmethod
    def _is_cum(cls):
        return True

    @staticmethod
    def _get_op_types():
        raise NotImplementedError

    @classmethod
    def tile(cls, op):
        from ..indexing.slice import TensorSlice

        in_tensor = op.inputs[0]
        axis = op.axis
        if not isinstance(axis, six.integer_types):
            raise ValueError("axis must be a integer")
        axis = validate_axis(in_tensor.ndim, axis)
        if axis is None:
            raise NotImplementedError

        op_type, binop_type = getattr(op, '_get_op_types')()

        chunks = []
        for c in in_tensor.chunks:
            chunk_op = op_type(axis=op.axis, dtype=op.dtype)
            chunks.append(chunk_op.new_chunk([c], c.shape, index=c.index))
        inter_tensor = copy.copy(in_tensor)
        inter_tensor._chunks = chunks

        slc = tuple(slice(None) if i != axis else slice(-1, None) for i in range(in_tensor.ndim))

        output_chunks = []
        for chunk in chunks:
            if chunk.index[axis] == 0:
                output_chunks.append(chunk)
                continue

            to_cum_chunks = [chunk]
            for i in range(chunk.index[axis]):
                to_cum_index = chunk.index[:axis] + (i,) + chunk.index[axis + 1:]
                shape = chunk.shape[:axis] + (1,) + chunk.shape[axis + 1:]
                to_cum_chunk = inter_tensor.cix[to_cum_index]
                slice_op = TensorSlice(slices=slc, dtype=chunk.dtype)
                sliced_chunk = slice_op.new_chunk([to_cum_chunk], shape, index=to_cum_index)
                to_cum_chunks.append(sliced_chunk)

            binop = binop_type(dtype=chunk.dtype)
            output_chunk = binop.new_chunk(to_cum_chunks, chunk.shape, index=chunk.index)
            output_chunks.append(output_chunk)

        nsplits = tuple((builtins.sum(c),) if i == axis else c for i, c in enumerate(in_tensor.nsplits))
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, in_tensor.shape, chunks=output_chunks, nsplits=nsplits)
