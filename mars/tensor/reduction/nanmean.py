#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import numpy as np

from ... import opcodes as OperandDef
from ..datasource import tensor as astensor
from ..array_utils import as_same_device, device
from .core import TensorReduction, TensorReductionMixin, nannumel
from .mean import TensorMean


class TensorNanMean(TensorReduction, TensorReductionMixin):
    _op_type_ = OperandDef.NANMEAN

    def __init__(self, axis=None, keepdims=None, combine_size=None, stage=None, **kw):
        stage = self._rewrite_stage(stage)
        super().__init__(_axis=axis, _keepdims=keepdims,
                         _combine_size=combine_size, stage=stage, **kw)

    @classmethod
    def execute_map(cls, ctx, op):
        (in_chunk,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        axis = cls.get_axis(op.axis)

        with device(device_id):
            chunk_count = nannumel(in_chunk, axis=axis, dtype=np.int64,
                                   keepdims=bool(op.keepdims))
            chunk_sum = xp.nansum(in_chunk, axis=axis, dtype=op.dtype,
                                  keepdims=bool(op.keepdims))
            ctx[op.outputs[0].key] = (chunk_sum, chunk_count)

    @classmethod
    def execute_agg(cls, ctx, op):
        axis = cls.get_axis(op.axis)

        a = ctx[op.inputs[0].key]
        if not isinstance(a, (list, tuple)):
            (inp,), device_id, xp = as_same_device(
                [a], device=op.device, ret_extra=True)

            with device(device_id):
                ctx[op.outputs[0].key] = xp.nanmean(inp, axis=axis, dtype=op.dtype,
                                                    keepdims=bool(op.keepdims))
        else:
            (_data, _count), device_id, xp = as_same_device(
                a, device=op.device, ret_extra=True)

            with device(device_id):
                chunk_count = xp.sum(_count, axis=axis, dtype=op.dtype,
                                     keepdims=bool(op.keepdims))
                chunk_sum = xp.sum(_data, axis=axis, dtype=op.dtype,
                                   keepdims=bool(op.keepdims))
                ctx[op.outputs[0].key] = xp.true_divide(chunk_sum, chunk_count,
                                                        dtype=op.dtype)

    @classmethod
    def execute_combine(cls, ctx, op):
        TensorMean.execute_combine(ctx, op)


def nanmean(a, axis=None, dtype=None, out=None, keepdims=None, combine_size=None):
    """
    Compute the arithmetic mean along the specified axis, ignoring NaNs.

    Returns the average of the tensor elements.  The average is taken over
    the flattened tensor by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    For all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    a : array_like
        Tensor containing numbers whose mean is desired. If `a` is not an
        tensor, a conversion is attempted.
    axis : int, optional
        Axis along which the means are computed. The default is to compute
        the mean of the flattened tensor.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for inexact inputs, it is the same as the input
        dtype.
    out : Tensor, optional
        Alternate output tensor in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.  See
        `doc.ufuncs` for details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If the value is anything but the default, then
        `keepdims` will be passed through to the `mean` or `sum` methods
        of sub-classes of `Tensor`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.
    combine_size: int, optional
        The number of chunks to combine.

    Returns
    -------
    m : Tensor, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned. Nan is
        returned for slices that contain only NaNs.

    See Also
    --------
    average : Weighted average
    mean : Arithmetic mean taken while not ignoring NaNs
    var, nanvar

    Notes
    -----
    The arithmetic mean is the sum of the non-NaN elements along the axis
    divided by the number of non-NaN elements.

    Note that for floating-point input, the mean is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32`.  Specifying a
    higher-precision accumulator using the `dtype` keyword can alleviate
    this issue.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array([[1, mt.nan], [3, 4]])
    >>> mt.nanmean(a).execute()
    2.6666666666666665
    >>> mt.nanmean(a, axis=0).execute()
    array([ 2.,  4.])
    >>> mt.nanmean(a, axis=1).execute()
    array([ 1.,  3.5])

    """
    a = astensor(a)
    if dtype is None:
        dtype = np.nanmean(np.empty((1,), dtype=a.dtype)).dtype
    op = TensorNanMean(axis=axis, dtype=dtype, keepdims=keepdims, combine_size=combine_size)
    return op(a, out=out)
