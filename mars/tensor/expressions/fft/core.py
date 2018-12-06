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

from collections import Iterable

from ..utils import validate_axis, decide_chunks, recursive_tile
from ..core import TensorOperandMixin


class TensorFFTMixin(TensorOperandMixin):
    __slots__ = ()

    @classmethod
    def _get_shape(cls, op, shape):
        raise NotImplementedError

    @classmethod
    def tile(cls, op):
        in_tensor = op.inputs[0]
        axis = op.axis

        if in_tensor.chunk_shape[axis] != 1:
            # fft requires only 1 chunk for the specified axis, so we do rechunk first
            chunks = {validate_axis(in_tensor.ndim, axis): in_tensor.shape[axis]}
            new_chunks = decide_chunks(in_tensor.shape, chunks, in_tensor.dtype.itemsize)
            in_tensor = in_tensor.rechunk(new_chunks).single_tiles()

        out_chunks = []
        for c in in_tensor.chunks:
            chunk_op = op.copy().reset_key()
            chunk_shape = cls._get_shape(op, c.shape)
            out_chunk = chunk_op.new_chunk([c], chunk_shape, index=c.index)
            out_chunks.append(out_chunk)

        nsplits = [tuple(c.shape[i] for c in out_chunks
                         if all(idx == 0 for j, idx in enumerate(c.index) if j != i))
                   for i in range(len(out_chunks[0].shape))]
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, op.outputs[0].shape,
                                  chunks=out_chunks, nsplits=nsplits)


def validate_fft(tensor, axis=-1, norm=None):
    validate_axis(tensor.ndim, axis)
    if norm is not None and norm not in ('ortho',):
        raise ValueError('Invalid norm value {0}, should be None or "ortho"'.format(norm))


class TensorFFTNMixin(TensorOperandMixin):
    __slots__ = ()

    @classmethod
    def _get_shape(cls, op, shape):
        raise NotImplementedError

    @classmethod
    def tile(cls, op):
        in_tensor = op.inputs[0]
        axes = op.axes

        if any(in_tensor.chunk_shape[axis] != 1 for axis in axes):
            new_chunks = decide_chunks(
                in_tensor.shape, {validate_axis(in_tensor.ndim, axis): in_tensor.shape[axis] for axis in axes},
                in_tensor.dtype.itemsize)
            in_tensor = in_tensor.rechunk(new_chunks).single_tiles()

        out_chunks = []
        for c in in_tensor.chunks:
            chunk_op = op.copy().reset_key()
            chunk_shape = cls._get_shape(op, c.shape)
            out_chunk = chunk_op.new_chunk([c], chunk_shape, index=c.index)
            out_chunks.append(out_chunk)

        nsplits = [tuple(c.shape[i] for c in out_chunks
                         if all(idx == 0 for j, idx in enumerate(c.index) if j != i))
                   for i in range(len(out_chunks[0].shape))]
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, op.outputs[0].shape,
                                  chunks=out_chunks, nsplits=nsplits)


def validate_fftn(tensor, s=None, axes=None, norm=None):
    if axes is None:
        if s is None:
            axes = tuple(range(tensor.ndim))
        else:
            axes = tuple(range(len(s)))
    else:
        [validate_axis(tensor.ndim, axis) for axis in axes]
        if len(set(axes)) < len(axes):
            raise ValueError('Duplicate axes not allowed')

    if norm is not None and norm not in ('ortho',):
        raise ValueError('Invalid norm value {0}, should be None or "ortho"'.format(norm))

    return axes


class TensorFFTShiftMixin(TensorOperandMixin):
    __slots__ = ()

    @classmethod
    def _is_inverse(cls):
        return False

    @classmethod
    def _process_axes(cls, x, axes):
        if axes is None:
            axes = tuple(range(x.ndim))
        elif isinstance(axes, Iterable):
            axes = tuple(axes)
        else:
            axes = (axes,)

        return axes

    @classmethod
    def tile(cls, op):
        from ..merge import concatenate

        axes = op.axes
        in_tensor = op.input
        is_inverse = cls._is_inverse()

        x = in_tensor
        for axis in axes:
            size = in_tensor.shape[axis]
            slice_on = (size + 1) // 2 if not is_inverse else size // 2
            slc1 = [slice(None)] * axis + [slice(slice_on)]
            slc2 = [slice(None)] * axis + [slice(slice_on, None)]
            x = concatenate([x[slc2], x[slc1]], axis=axis)

        recursive_tile(x)
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, op.outputs[0].shape,
                                  chunks=x.chunks, nsplits=x.nsplits)
