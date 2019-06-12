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

from numbers import Number

import numpy as np

from .... import opcodes as OperandDef
from ....serialize import KeyField, AnyField
from ....core import Base, Entity
from ...core import Tensor
from ..utils import broadcast_shape
from ..datasource import tensor as astensor
from .core import TensorOperand, TensorElementWise, filter_inputs


class TensorClip(TensorOperand, TensorElementWise):
    _op_type_ = OperandDef.CLIP

    _a = KeyField('a')
    _a_min = AnyField('a_min')
    _a_max = AnyField('a_max')
    _out = KeyField('out')

    @property
    def a(self):
        return self._a

    @property
    def a_min(self):
        return self._a_min

    @property
    def a_max(self):
        return self._a_max

    @property
    def out(self):
        return getattr(self, '_out', None)

    def _set_inputs(self, inputs):
        super(TensorClip, self)._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        self._a = next(inputs_iter)
        if isinstance(self._a_min, (Base, Entity)):
            self._a_min = next(inputs_iter)
        if isinstance(self._a_max, (Base, Entity)):
            self._a_max = next(inputs_iter)
        if getattr(self, '_out', None) is not None:
            self._out = next(inputs_iter)

    def __call__(self, a, a_min, a_max, out=None):
        a = astensor(a)
        tensors = [a]
        sparse = a.issparse()

        if isinstance(a_min, Number):
            if a_min > 0:
                sparse = False
            a_min_dtype = np.array(a_min).dtype
        else:
            a_min = astensor(a_min)
            tensors.append(a_min)
            if not a_min.issparse():
                sparse = False
            a_min_dtype = a_min.dtype
        self._a_min = a_min

        if isinstance(a_max, Number):
            if a_max < 0:
                sparse = False
            a_max_dtype = np.array(a_max).dtype
        else:
            a_max = astensor(a_max)
            tensors.append(a_max)
            if not a_max.issparse():
                sparse = False
            a_max_dtype = a_max.dtype
        self._a_max = a_max

        if out is not None:
            if isinstance(out, Tensor):
                self._out = out
            else:
                raise TypeError('out should be Tensor object, got {0} instead'.format(type(out)))

        dtype = np.result_type(a.dtype, a_min_dtype, a_max_dtype)
        # check broadcast
        shape = broadcast_shape(*[t.shape for t in tensors])

        setattr(self, '_sparse', sparse)
        inputs = filter_inputs([a, a_min, a_max, out])
        t = self.new_tensor(inputs, shape)

        if out is None:
            setattr(self, '_dtype', dtype)
            return t

        # if `out` is specified, use out's dtype and shape
        out_shape, out_dtype = out.shape, out.dtype

        if t.shape != out_shape:
            t = self.new_tensor(inputs, out_shape)
        setattr(self, '_dtype', out_dtype)

        out.data = t.data
        return out


def clip(a, a_min, a_max, out=None):
    """
    Clip (limit) the values in a tensor.

    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Parameters
    ----------
    a : array_like
        Tensor containing elements to clip.
    a_min : scalar or array_like or `None`
        Minimum value. If `None`, clipping is not performed on lower
        interval edge. Not more than one of `a_min` and `a_max` may be
        `None`.
    a_max : scalar or array_like or `None`
        Maximum value. If `None`, clipping is not performed on upper
        interval edge. Not more than one of `a_min` and `a_max` may be
        `None`. If `a_min` or `a_max` are array_like, then the three
        arrays will be broadcasted to match their shapes.
    out : Tensor, optional
        The results will be placed in this tensor. It may be the input
        array for in-place clipping.  `out` must be of the right shape
        to hold the output.  Its type is preserved.

    Returns
    -------
    clipped_array : Tensor
        An tensor with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.arange(10)
    >>> mt.clip(a, 1, 8).execute()
    array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
    >>> a.execute()
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> mt.clip(a, 3, 6, out=a).execute()
    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
    >>> a = mt.arange(10)
    >>> a.execute()
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> mt.clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8).execute()
    array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])

    """
    op = TensorClip()
    return op(a, a_min, a_max, out=out)
