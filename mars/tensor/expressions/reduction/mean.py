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

import numpy as np

from .... import opcodes as OperandDef
from ..datasource import tensor as astensor
from .core import TensorReduction, TensorReductionMixin


class TensorMeanChunk(TensorReduction, TensorReductionMixin):
    _op_type_ = OperandDef.MEAN_CHUNK

    def __init__(self, axis=None, dtype=None, keepdims=None, combine_size=None, **kw):
        super(TensorMeanChunk, self).__init__(_axis=axis, _dtype=dtype, _keepdims=keepdims,
                                              _combine_size=combine_size, **kw)


class TensorMeanCombine(TensorReduction, TensorReductionMixin):
    _op_type_ = OperandDef.MEAN_COMBINE

    def __init__(self, axis=None, dtype=None, keepdims=None, combine_size=None, **kw):
        super(TensorMeanCombine, self).__init__(_axis=axis, _dtype=dtype, _keepdims=keepdims,
                                                _combine_size=combine_size, **kw)


class TensorMean(TensorReduction, TensorReductionMixin):
    _op_type_ = OperandDef.MEAN

    def __init__(self, axis=None, dtype=None, keepdims=None, combine_size=None, **kw):
        super(TensorMean, self).__init__(_axis=axis, _dtype=dtype, _keepdims=keepdims,
                                         _combine_size=combine_size, **kw)

    @staticmethod
    def _get_op_types():
        return TensorMeanChunk, TensorMean, TensorMeanCombine


def mean(a, axis=None, dtype=None, out=None, keepdims=None, combine_size=None):
    """
    Compute the arithmetic mean along the specified axis.

    Returns the average of the array elements.  The average is taken over
    the flattened tensor by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    Parameters
    ----------
    a : array_like
        Tensor containing numbers whose mean is desired. If `a` is not an
        tensor, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.

        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the
        input dtype.
    out : Tensor, optional
        Alternate output tensor in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See `doc.ufuncs` for details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input tensor.

        If the default value is passed, then `keepdims` will not be
        passed through to the `mean` method of sub-classes of
        `Tensor`, however any non-default value will be.  If the
        sub-classes `sum` method does not implement `keepdims` any
        exceptions will be raised.
    combine_size: int, optional
        The number of chunks to combine.

    Returns
    -------
    m : Tensor, see dtype parameter above
        If `out=None`, returns a new tensor containing the mean values,
        otherwise a reference to the output array is returned.

    See Also
    --------
    average : Weighted average
    std, var, nanmean, nanstd, nanvar

    Notes
    -----
    The arithmetic mean is the sum of the elements along the axis divided
    by the number of elements.

    Note that for floating-point input, the mean is computed using the
    same precision the input has.  Depending on the input data, this can
    cause the results to be inaccurate, especially for `float32` (see
    example below).  Specifying a higher-precision accumulator using the
    `dtype` keyword can alleviate this issue.

    By default, `float16` results are computed using `float32` intermediates
    for extra precision.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array([[1, 2], [3, 4]])
    >>> mt.mean(a).execute()
    2.5
    >>> mt.mean(a, axis=0).execute()
    array([ 2.,  3.])
    >>> mt.mean(a, axis=1).execute()
    array([ 1.5,  3.5])

    In single precision, `mean` can be inaccurate:

    >>> a = mt.zeros((2, 512*512), dtype=mt.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> mt.mean(a).execute()
    0.54999924

    Computing the mean in float64 is more accurate:

    >>> mt.mean(a, dtype=mt.float64).execute()
    0.55000000074505806

    """
    a = astensor(a)
    if dtype is None:
        dtype = np.mean(np.empty((1,), dtype=a.dtype)).dtype
    op = TensorMean(axis=axis, dtype=dtype, keepdims=keepdims, combine_size=combine_size)
    return op(a, out=out)
