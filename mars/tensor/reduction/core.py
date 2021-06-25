#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import builtins
import copy
import inspect
import itertools
import operator
from collections.abc import Iterable
from functools import reduce
from math import ceil, log

import numpy as np

from ...config import options
from ...core.operand import OperandStage
from ...serialization.serializables import KeyField, AnyField, BoolField, Int32Field
from ..core import Tensor, TensorOrder
from ..array_utils import get_array_module, as_same_device, device, cp
from ..utils import check_out_param, validate_axis
from ..operands import TensorHasInput, TensorOperandMixin
from ..datasource import tensor as astensor


def numel(x, **kwargs):
    xp = get_array_module(x)
    return xp.sum(xp.ones_like(x), **kwargs)


def nannumel(x, **kwargs):
    x_size = reduce(operator.mul, x.shape)
    xp = get_array_module(x)
    return x_size - xp.sum(xp.isnan(x), **kwargs)


class TensorReductionMixin(TensorOperandMixin):
    __slots__ = ()

    @classmethod
    def _is_cum(cls):
        return False

    @classmethod
    def _calc_order(cls, a, out):
        return out.order if out is not None else a.order

    @classmethod
    def _is_sparse(cls, input_sparse, shape):
        return False

    def _call(self, a, out):
        a = astensor(a)
        if out is not None and not isinstance(out, Tensor):
            raise TypeError(f'out should be Tensor object, got {type(out)} instead')

        axis = getattr(self, 'axis', None)
        keepdims = getattr(self, 'keepdims', None)
        order = self._calc_order(a, out)

        if self._is_cum():
            if axis is None:
                a, axis = a.ravel(), 0
                setattr(self, '_axis', axis)
            shape = a.shape
        else:
            axis = list(range(len(a.shape))) if axis is None else axis
            if not isinstance(axis, Iterable):
                axis = (validate_axis(a.ndim, axis),)
            axis = set(axis)

            shape = tuple(s if i not in axis else 1 for i, s in enumerate(a.shape)
                          if keepdims or i not in axis)

        self.sparse = self._is_sparse(a.issparse(), shape)
        t = self.new_tensor([a], shape, order=order)

        if out is None:
            return t

        check_out_param(out, t, 'same_kind')
        out_shape, out_dtype = out.shape, out.dtype
        # if `out` is specified, use out's dtype and shape
        if out_shape != t.shape:
            if out.ndim > t.ndim:
                raise ValueError('output has too many dimensions')
            raise ValueError(f'output shape should be {t.shape}, got {out_shape}')

        setattr(self, 'dtype', out_dtype)

        out.data = t.data
        return out

    def _new_chunks(self, inputs, kws=None, **kw):
        chunks = super()._new_chunks(inputs, kws=kws, **kw)
        setattr(self, '_input', getattr(self, '_inputs')[0])
        return chunks

    def _new_tileables(self, inputs, kws=None, **kw):
        tensors = super()._new_tileables(inputs, kws=kws, **kw)
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
                     for nsplit, cb in zip(tensor.nsplits, combine_block))

    @staticmethod
    def _combine_split(ax, combine_size, chunk_shape):
        if ax not in combine_size:
            return tuple((i,) for i in range(chunk_shape[ax]))
        else:
            size = combine_size[ax]
            shape = chunk_shape[ax]
            index = tuple(range(shape))
            return tuple(index[i:i + size] for i in range(0, shape, size))

    def _get_op_kw(self):
        return None

    @classmethod
    def get_axis(cls, axis):
        return tuple(axis) if axis is not None else axis

    @classmethod
    def get_arg_axis(cls, axis, ndim):
        return None if len(axis) == ndim or ndim == 1 else axis[0]

    @classmethod
    def _tree_reduction(cls, tensor, axis):
        op = tensor.op
        kw = getattr(op, '_get_op_kw')() or {}
        keepdims = op.keepdims
        combine_size = op.combine_size or options.combine_size
        if isinstance(combine_size, dict):
            combine_size = dict((ax, combine_size.get(ax)) for ax in axis)
        else:
            assert isinstance(combine_size, int)
            n = builtins.max(int(combine_size ** (1.0 / (len(axis) or 1))), 2)
            combine_size = dict((ax, n) for ax in axis)

        times = 1
        for i, n in enumerate(tensor.chunk_shape):
            if i in combine_size and combine_size[i] != 1:
                times = int(builtins.max(times, ceil(log(n, combine_size[i]))))

        for i in range(times - 1):
            [tensor] = cls._partial_reduction(tensor, axis, op.dtype, True, combine_size, OperandStage.combine)

        return cls._partial_reduction(tensor, axis, op.dtype, keepdims, combine_size, OperandStage.agg, kw)

    @classmethod
    def _partial_reduction(cls, tensor, axis, dtype, keepdims, combine_size, stage, kw=None):
        from ..merge.concatenate import TensorConcatenate
        kw = kw or {}
        axes = sorted(combine_size.keys())
        op_type = type(tensor.op)

        combine_blocks = [cls._combine_split(i, combine_size, tensor.chunk_shape)
                          for i in range(tensor.ndim)]
        combine_blocks_idxes = [range(len(blocks)) for blocks in combine_blocks]

        chunks = []
        for combine_block_idx, combine_block in zip(itertools.product(*combine_blocks_idxes),
                                                    itertools.product(*combine_blocks)):
            chks = [tensor.cix[idx] for idx in itertools.product(*combine_block)]
            if len(chks) > 1:
                op = TensorConcatenate(axis=axes, dtype=chks[0].dtype)
                chk = op.new_chunk(chks, shape=cls._concatenate_shape(tensor, combine_block),
                                   order=tensor.order)
            else:
                chk = chks[0]
            shape = tuple(s if i not in combine_size else 1
                          for i, s in enumerate(chk.shape) if keepdims or i not in combine_size)
            agg_op = op_type(stage=stage, axis=axis, dtype=dtype, keepdims=keepdims, **kw)
            chunk = agg_op.new_chunk([chk], shape=shape,
                                     index=tuple(idx for i, idx in enumerate(combine_block_idx)
                                                 if keepdims or i not in combine_size),
                                     order=tensor.order)
            chunks.append(chunk)

        nsplits = [
            tuple(c.shape[i] for c in chunks if builtins.all(idx == 0 for j, idx in enumerate(c.index) if j != i))
            for i in range(len(chunks[0].shape))]
        shape = tuple(builtins.sum(nsplit) for nsplit in nsplits)
        agg_op = op_type(stage=stage, axis=axis, dtype=dtype, keepdims=keepdims, combine_size=combine_size, **kw)
        return agg_op.new_tensors([tensor], shape, order=tensor.order,
                                  chunks=chunks, nsplits=nsplits)

    @classmethod
    def tile(cls, op):
        in_tensor = op.inputs[0]
        out_tensor = op.outputs[0]
        axis = tuple(range(in_tensor.ndim)) if op.axis is None else op.axis
        if isinstance(axis, int):
            axis = (axis,)
        axis = tuple(validate_axis(in_tensor.ndim, ax) for ax in axis)

        if len(in_tensor.chunks) == 1:
            c = in_tensor.chunks[0]
            new_op = op.copy().reset_key()
            setattr(new_op, '_axis', axis)
            shape = list(cls._reduced_shape(c.shape, axis))
            nsplits = list(cls._reduced_nsplits(in_tensor.nsplits, axis))
            chunk_index = list(c.index)
            if not op.keepdims and axis:
                for ax in axis:
                    shape[ax] = None
                    nsplits[ax] = None
                    chunk_index[ax] = None
            shape = tuple(s for s in shape if s is not None)
            nsplits = tuple(ns for ns in nsplits if ns is not None)
            chunk_index = tuple(i for i in chunk_index if i is not None)

            chunks = new_op.new_chunks([c], shape=shape, index=chunk_index, order=out_tensor.order)
            return op.copy().new_tensors(op.inputs, op.outputs[0].shape, order=out_tensor.order,
                                         chunks=chunks, nsplits=nsplits)

        chunks = []
        kw = getattr(op, '_get_op_kw')() or {}
        for c in in_tensor.chunks:
            chunk_op = type(op)(stage=OperandStage.map, axis=axis, dtype=op.dtype, keepdims=True,
                                combine_size=op.combine_size, **kw)
            chunks.append(chunk_op.new_chunk([c], shape=cls._reduced_shape(c.shape, axis),
                                             order=out_tensor.order, index=c.index))

        new_op = op.copy()
        tensor = new_op.new_tensor(op.inputs, cls._reduced_shape(in_tensor.shape, axis),
                                   order=out_tensor.order,
                                   nsplits=cls._reduced_nsplits(in_tensor.nsplits, axis), chunks=chunks)
        return cls._tree_reduction(tensor, axis)

    @classmethod
    def execute_agg(cls, ctx, op):
        (input_chunk,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)
        axis = cls.get_axis(op.axis)
        func_name = getattr(cls, '_func_name', None)
        reduce_func = getattr(xp, func_name)
        out = op.outputs[0]
        with device(device_id):
            if input_chunk.size == 0 and op.keepdims:
                # input chunk is empty, when keepdims is True, return itself
                ret = input_chunk
            elif "dtype" in inspect.getfullargspec(reduce_func).args:
                ret = reduce_func(input_chunk, axis=axis,
                                  dtype=op.dtype,
                                  keepdims=bool(op.keepdims))
            else:
                ret = reduce_func(input_chunk, axis=axis,
                                  keepdims=bool(op.keepdims))

            if hasattr(ret, 'astype'):
                # for non-object dtype
                ret = ret.astype(op.dtype, order=out.order.value, copy=False)
            ctx[out.key] = ret

    @classmethod
    def execute_one_chunk(cls, ctx, op):
        cls.execute_agg(ctx, op)

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            return cls.execute_map(ctx, op)
        elif op.stage == OperandStage.combine:
            return cls.execute_combine(ctx, op)
        elif op.stage == OperandStage.agg:
            return cls.execute_agg(ctx, op)
        else:
            return cls.execute_one_chunk(ctx, op)


class TensorArgReductionMixin(TensorReductionMixin):
    __slots__ = ()

    @staticmethod
    def _get_arg_axis(axis, ndim):
        if axis is None:
            axis = tuple(range(ndim))
            ravel = True
        elif isinstance(axis, int):
            axis = validate_axis(ndim, axis)
            axis = (axis,)
            ravel = ndim == 1
        else:
            raise TypeError("axis must be either `None` or int, "
                            f"got '{axis}'")
        return axis, ravel

    @staticmethod
    def _get_offset(tensor, axis, chunk, ravel):
        nsplits = tensor.nsplits
        offset = tuple(builtins.sum(split[:idx]) for split, idx in zip(nsplits, chunk.index))
        if not ravel:
            offset = offset[axis[0]]
        return offset

    @classmethod
    def _calc_order(cls, a, out):
        return out.order if out is not None else TensorOrder.C_ORDER

    @classmethod
    def tile(cls, op):
        in_tensor = op.inputs[0]
        out_tensor = op.outputs[0]
        axis, ravel = cls._get_arg_axis(op.axis, in_tensor.ndim)

        chunks = []
        for c in in_tensor.chunks:
            offset = cls._get_offset(in_tensor, axis, c, ravel)
            chunk_op = type(op)(stage=OperandStage.map, axis=axis, dtype=op.dtype,
                                offset=offset, total_shape=in_tensor.shape,
                                combine_size=op.combine_size)
            chunk = chunk_op.new_chunk([c], shape=cls._reduced_shape(c.shape, axis),
                                       index=c.index, order=out_tensor.order)
            chunks.append(chunk)
        new_op = op.copy()
        tensor = new_op.new_tensor(op.inputs, cls._reduced_shape(in_tensor.shape, axis),
                                   order=out_tensor.order,
                                   nsplits=cls._reduced_nsplits(in_tensor.nsplits, axis), chunks=chunks)
        return cls._tree_reduction(tensor, axis)

    @classmethod
    def execute_agg(cls, ctx, op):
        axis = cls.get_arg_axis(op.axis, op.inputs[0].ndim)
        (vals, arg), device_id, xp = as_same_device(
            ctx[op.inputs[0].key], device=op.device, ret_extra=True)

        func_name = getattr(cls, '_func_name')
        arg_func = getattr(xp, func_name)

        with device(device_id):
            if xp.any(xp.isnan(vals)) and 'nan' in func_name:
                raise ValueError("All NaN slice encountered")
            if axis is None:
                local_args = arg_func(vals, axis=axis)
                arg = arg.ravel()[local_args]
            else:
                local_args = arg_func(vals, axis=axis)
                inds = np.ogrid[tuple(map(slice, local_args.shape))]
                if xp != np:
                    inds = [xp.asarray(it) for it in inds]
                inds.insert(axis, local_args)
                arg = arg[tuple(inds)]
            ctx[op.outputs[0].key] = arg

    @classmethod
    def execute_map(cls, ctx, op):
        arg_axis = cls.get_arg_axis(op.axis, op.inputs[0].ndim)
        (in_chunk,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        func_name = getattr(cls, '_func_name')
        agg_func_name = getattr(cls, '_agg_func_name')
        arg_func = getattr(xp, func_name)
        agg_func_name = getattr(xp, agg_func_name)

        offset = op.offset
        chunk = op.outputs[0]
        with device(device_id):
            vals = agg_func_name(in_chunk, axis=arg_axis)
            if hasattr(vals, 'reshape'):
                vals = vals.reshape(chunk.shape)
            try:
                arg = arg_func(in_chunk, axis=arg_axis)
                if hasattr(arg, 'reshape'):
                    arg = arg.reshape(chunk.shape)
            except ValueError:
                # handle all NaN
                arg = arg_func(xp.where(xp.isnan(in_chunk), np.inf, in_chunk),
                               axis=arg_axis).reshape(chunk.shape)

            if arg_axis is None:
                if xp == cp:
                    # we need to copy to do cpu computation, then copy back to gpu
                    # cuz unravel_index and ravel_multi_index are not implemented in cupy
                    in_chunk = in_chunk.get()

                total_shape = op.total_shape
                ind = np.unravel_index(arg.ravel()[0], in_chunk.shape)
                total_ind = tuple(o + i for (o, i) in zip(offset, ind))
                res = np.ravel_multi_index(total_ind, total_shape)

                if xp == cp:
                    # copy back
                    with xp.cuda.Device(in_chunk.device.id):
                        arg[:] = xp.asarray(res)
                else:
                    arg[:] = res
            else:
                arg += offset
            ctx[op.outputs[0].key] = (vals, arg)

    @classmethod
    def execute_combine(cls, ctx, op):
        axis = cls.get_arg_axis(op.axis, op.inputs[0].ndim)
        (vals, arg), device_id, xp = as_same_device(
            ctx[op.inputs[0].key], device=op.device, ret_extra=True)

        func_name = getattr(cls, '_func_name')
        arg_func = getattr(xp, func_name)
        with device(device_id):
            if axis is None:
                local_args = arg_func(vals, axis=axis).reshape(op.outputs[0].shape)
                vals = vals.ravel()[local_args]
                arg = arg.ravel()[local_args]
            else:
                local_args = arg_func(vals, axis=axis)
                inds = np.ogrid[tuple(map(slice, local_args.shape))]
                if xp != np:
                    inds = [xp.asarray(it) for it in inds]
                inds.insert(axis, local_args)
                inds_tuple = tuple(inds)
                vals = vals[inds_tuple].reshape(op.outputs[0].shape)
                arg = arg[inds_tuple].reshape(op.outputs[0].shape)
            ctx[op.outputs[0].key] = (vals, arg)


class TensorCumReductionMixin(TensorReductionMixin):
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
        out_tensor = op.outputs[0]
        axis = op.axis
        if not isinstance(axis, int):
            raise ValueError("axis must be a integer")
        axis = validate_axis(in_tensor.ndim, axis)
        if axis is None:
            raise NotImplementedError

        op_type, bin_op_type = getattr(op, '_get_op_types')()

        chunks = []
        for c in in_tensor.chunks:
            chunk_op = op_type(axis=op.axis, dtype=op.dtype)
            chunks.append(chunk_op.new_chunk([c], shape=c.shape,
                                             index=c.index, order=out_tensor.order))
        inter_tensor = copy.copy(in_tensor)
        inter_tensor._chunks = chunks

        slc = [slice(None) if i != axis else slice(-1, None)
               for i in range(in_tensor.ndim)]

        output_chunks = []
        for chunk in chunks:
            if chunk.index[axis] == 0:
                output_chunks.append(chunk)
                continue

            to_cum_chunks = []
            for i in range(chunk.index[axis]):
                to_cum_index = chunk.index[:axis] + (i,) + chunk.index[axis + 1:]
                shape = chunk.shape[:axis] + (1,) + chunk.shape[axis + 1:]
                to_cum_chunk = inter_tensor.cix[to_cum_index]
                slice_op = TensorSlice(slices=slc, dtype=chunk.dtype)
                sliced_chunk = slice_op.new_chunk([to_cum_chunk], shape=shape,
                                                  index=to_cum_index, order=out_tensor.order)
                to_cum_chunks.append(sliced_chunk)
            to_cum_chunks.append(chunk)

            bin_op = bin_op_type(args=to_cum_chunks, dtype=chunk.dtype)
            output_chunk = bin_op.new_chunk(to_cum_chunks, shape=chunk.shape,
                                            index=chunk.index, order=out_tensor.order)
            output_chunks.append(output_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, in_tensor.shape, order=out_tensor.order,
                                  chunks=output_chunks, nsplits=in_tensor.nsplits)

    @classmethod
    def execute(cls, ctx, op):
        (x,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        func_name = getattr(cls, '_func_name')
        cum_func = getattr(xp, func_name)
        if xp != np:
            func = getattr(xp, cum_func.__name__)
        else:
            func = cum_func

        with device(device_id):
            ctx[op.outputs[0].key] = func(x, axis=op.axis, dtype=op.dtype)


class TensorReduction(TensorHasInput):
    _input = KeyField('input')
    _out = KeyField('out')
    _axis = AnyField('axis')  # can be None or int or tuple of ints, just infer the data
    _keepdims = BoolField('keepdims')
    _combine_size = AnyField('combine_size')

    @property
    def axis(self):
        return getattr(self, '_axis', None)

    @property
    def keepdims(self):
        return getattr(self, '_keepdims', None)

    @property
    def combine_size(self):
        return getattr(self, '_combine_size', None)

    def _rewrite_stage(self, stage):
        if stage == OperandStage.map and not hasattr(self, 'execute_map'):
            return OperandStage.agg
        elif stage == OperandStage.combine and not hasattr(self, 'execute_combine'):
            return OperandStage.agg
        return stage


class TensorCumReduction(TensorHasInput):
    _input = KeyField('input')
    _axis = Int32Field('axis')

    @property
    def axis(self):
        return getattr(self, '_axis', None)
