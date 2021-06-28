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

import numpy as np

from ... import opcodes as OperandDef
from ..datasource import tensor as astensor
from .core import TensorReduction, TensorReductionMixin


class TensorSum(TensorReduction, TensorReductionMixin):
    _op_type_ = OperandDef.SUM
    _func_name = 'sum'

    def __init__(self, axis=None, keepdims=None, combine_size=None, stage=None, **kw):
        stage = self._rewrite_stage(stage)
        super().__init__(_axis=axis, _keepdims=keepdims,
                         _combine_size=combine_size, stage=stage, **kw)


def sum(a, axis=None, dtype=None, out=None, keepdims=None, combine_size=None):
    """
    Sum of tensor elements over a given axis.

    Parameters
    ----------
    a : array_like
        Elements to sum.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input tensor.  If
        axis is negative it counts from the last to the first axis.

        If axis is a tuple of ints, a sum is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    dtype : dtype, optional
        The type of the returned tensor and of the accumulator in which the
        elements are summed.  The dtype of `a` is used by default unless `a`
        has an integer dtype of less precision than the default platform
        integer.  In that case, if `a` is signed then the platform integer
        is used while if `a` is unsigned then an unsigned integer of the
        same precision as the platform integer is used.
    out : Tensor, optional
        Alternative output tensor in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input tensor.

        If the default value is passed, then `keepdims` will not be
        passed through to the `sum` method of sub-classes of
        `Tensor`, however any non-default value will be.  If the
        sub-classes `sum` method does not implement `keepdims` any
        exceptions will be raised.
    combine_size: int, optional
        The number of chunks to combine.

    Returns
    -------
    sum_along_axis : Tensor
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d tensor, or if `axis` is None, a scalar
        is returned.  If an output array is specified, a reference to
        `out` is returned.

    See Also
    --------
    Tensor.sum : Equivalent method.

    cumsum : Cumulative sum of tensor elements.

    trapz : Integration of tensor values using the composite trapezoidal rule.

    mean, average

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    The sum of an empty array is the neutral element 0:

    >>> import mars.tensor as mt

    >>> mt.sum([]).execute()
    0.0

    Examples
    --------
    >>> mt.sum([0.5, 1.5]).execute()
    2.0
    >>> mt.sum([0.5, 0.7, 0.2, 1.5], dtype=mt.int32).execute()
    1
    >>> mt.sum([[0, 1], [0, 5]]).execute()
    6
    >>> mt.sum([[0, 1], [0, 5]], axis=0).execute()
    array([0, 6])
    >>> mt.sum([[0, 1], [0, 5]], axis=1).execute()
    array([1, 5])

    If the accumulator is too small, overflow occurs:

    >>> mt.ones(128, dtype=mt.int8).sum(dtype=mt.int8).execute()
    -128

    """
    a = astensor(a)
    if dtype is None:
        if a.dtype == np.object_:
            dtype = a.dtype
        else:
            dtype = np.empty((1,), dtype=a.dtype).sum().dtype
    else:
        dtype = np.dtype(dtype)
    op = TensorSum(axis=axis, dtype=dtype, keepdims=keepdims, combine_size=combine_size)
    return op(a, out=out)
