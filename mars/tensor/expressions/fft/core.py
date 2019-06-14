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

from ....compat import izip
from ....serialize import ValueType, KeyField, StringField, Int32Field, \
    Int64Field, ListField
from ..utils import validate_axis, decide_chunk_sizes, recursive_tile
from ..core import TensorHasInput, TensorOperandMixin


class TensorFFTBaseMixin(TensorOperandMixin):
    __slots__ = ()

    @classmethod
    def _get_shape(cls, op, shape):
        raise NotImplementedError

    @classmethod
    def _tile_fft(cls, op, axes):
        in_tensor = op.inputs[0]

        if any(in_tensor.chunk_shape[axis] != 1 for axis in axes):
            # fft requires only 1 chunk for the specified axis, so we do rechunk first
            chunks = {validate_axis(in_tensor.ndim, axis): in_tensor.shape[axis] for axis in axes}
            new_chunks = decide_chunk_sizes(in_tensor.shape, chunks, in_tensor.dtype.itemsize)
            in_tensor = in_tensor.rechunk(new_chunks).single_tiles()

        out_chunks = []
        for c in in_tensor.chunks:
            chunk_op = op.copy().reset_key()
            chunk_shape = cls._get_shape(op, c.shape)
            out_chunk = chunk_op.new_chunk([c], shape=chunk_shape, index=c.index)
            out_chunks.append(out_chunk)

        nsplits = [tuple(c.shape[i] for c in out_chunks
                         if all(idx == 0 for j, idx in enumerate(c.index) if j != i))
                   for i in range(len(out_chunks[0].shape))]
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, op.outputs[0].shape,
                                  chunks=out_chunks, nsplits=nsplits)

    def __call__(self, a):
        shape = self._get_shape(self, a.shape)
        return self.new_tensor([a], shape)


class TensorFFTMixin(TensorFFTBaseMixin):
    __slots__ = ()

    @classmethod
    def tile(cls, op):
        return cls._tile_fft(op, [op.axis])


class TensorComplexFFTMixin(TensorFFTMixin):
    @classmethod
    def _get_shape(cls, op, shape):
        new_shape = list(shape)
        if op.n is not None:
            new_shape[op.axis] = op.n
        return tuple(new_shape)


def validate_fft(tensor, axis=-1, norm=None):
    validate_axis(tensor.ndim, axis)
    if norm is not None and norm not in ('ortho',):
        raise ValueError('Invalid norm value {0}, should be None or "ortho"'.format(norm))


class TensorFFTNMixin(TensorFFTBaseMixin):
    @classmethod
    def tile(cls, op):
        return cls._tile_fft(op, op.axes)

    @staticmethod
    def _merge_shape(op, shape):
        new_shape = list(shape)
        if op.shape is not None:
            for ss, axis in izip(op.shape, op.axes):
                new_shape[axis] = ss
        return new_shape


class TensorComplexFFTNMixin(TensorFFTNMixin):
    @classmethod
    def _get_shape(cls, op, shape):
        return tuple(cls._merge_shape(op, shape))


class TensorRealFFTNMixin(TensorFFTNMixin):
    @classmethod
    def _get_shape(cls, op, shape):
        new_shape = cls._merge_shape(op, shape)
        new_shape[op.axes[-1]] = new_shape[op.axes[-1]] // 2 + 1
        return tuple(new_shape)


class TensorRealIFFTNMixin(TensorFFTNMixin):
    @classmethod
    def _get_shape(cls, op, shape):
        new_shape = list(shape)
        new_shape[op.axes[-1]] = 2 * (new_shape[op.axes[-1]] - 1)
        return tuple(cls._merge_shape(op, new_shape))


def validate_fftn(tensor, s=None, axes=None, norm=None):
    if axes is None:
        if s is None:
            axes = tuple(range(tensor.ndim))
        else:
            axes = tuple(range(len(s)))
    else:
        for axis in axes:
            validate_axis(tensor.ndim, axis)
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


class TensorDiscreteFourierTransform(TensorHasInput):
    __slots__ = ()


class TensorBaseFFT(TensorDiscreteFourierTransform):
    _input = KeyField('input')
    _norm = StringField('norm')

    @property
    def norm(self):
        return getattr(self, '_norm', None)


class TensorBaseSingleDimensionFFT(TensorBaseFFT):
    _n = Int64Field('n')
    _axis = Int32Field('axis')

    @property
    def n(self):
        return self._n

    @property
    def axis(self):
        return self._axis


class TensorBaseMultipleDimensionFFT(TensorBaseFFT):
    _shape = ListField('shape', ValueType.int64)
    _axes = ListField('axes', ValueType.int32)

    @property
    def shape(self):
        return self._shape

    @property
    def axes(self):
        return self._axes


class TensorStandardFFT(TensorBaseSingleDimensionFFT):
    pass


class TensorStandardFFTN(TensorBaseMultipleDimensionFFT):
    pass


class TensorFFTShiftBase(TensorHasInput):
    _input = KeyField('input')
    _axes = ListField('axes', ValueType.int32)

    @property
    def axes(self):
        return self._axes


class TensorRealFFT(TensorBaseSingleDimensionFFT):
    pass


class TensorRealFFTN(TensorBaseMultipleDimensionFFT):
    pass


class TensorHermitianFFT(TensorBaseSingleDimensionFFT):
    pass

