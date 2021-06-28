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
from ...serialization.serializables import Int32Field
from ..datasource import tensor as astensor
from ..array_utils import device, as_same_device, get_array_module
from .var import reduce_var_square
from .core import TensorReduction, TensorReductionMixin, nannumel


class TensorNanMoment(TensorReduction, TensorReductionMixin):
    _op_type_ = OperandDef.NANMOMENT

    _moment = Int32Field('moment', default=2)
    _ddof = Int32Field('ddof')

    def __init__(self, axis=None, keepdims=None, moment=None, ddof=None,
                 combine_size=None, stage=None, **kw):
        stage = self._rewrite_stage(stage)
        if moment is not None:
            kw['_moment'] = moment
        super().__init__(_axis=axis, _keepdims=keepdims, _ddof=ddof,
                         _combine_size=combine_size, stage=stage, **kw)

    @property
    def moment(self):
        return getattr(self, '_moment', 2)

    @property
    def ddof(self):
        return self._ddof

    @classmethod
    def execute_agg(cls, ctx, op):
        axis = cls.get_axis(op.axis)
        dtype = op.dtype

        (_data, _count, _var_square), device_id, xp = as_same_device(
            ctx[op.inputs[0].key], device=op.device, ret_extra=True)

        with device(device_id):
            chunk_count = xp.nansum(_count, axis=axis, dtype=np.int64,
                                    keepdims=True)
            chunk_sum = xp.nansum(_data, axis=axis, dtype=dtype, keepdims=True)
            avg = xp.true_divide(chunk_sum, chunk_count, dtype=dtype)
            avg_diff = xp.true_divide(_data, _count, dtype=dtype) - avg
            var_square = reduce_var_square(_var_square, avg_diff, _count, op, axis, xp.nansum)

            ctx[op.outputs[0].key] = xp.true_divide(
                var_square,
                xp.nansum(chunk_count, axis=axis, dtype=dtype, keepdims=bool(op.keepdims)) - op.ddof,
                dtype=dtype)

    @classmethod
    def execute_map(cls, ctx, op):
        (in_chunk,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        axis = cls.get_axis(op.axis)
        moment = op.moment
        dtype = op.dtype
        empty = get_array_module(in_chunk, nosparse=True).empty

        with device(device_id):
            chunk_count = nannumel(in_chunk, axis=axis, dtype=np.int64, keepdims=bool(op.keepdims))
            chunk_sum = xp.nansum(in_chunk, axis=axis, dtype=dtype, keepdims=bool(op.keepdims))
            avg = xp.true_divide(chunk_sum, chunk_count)
            var_square = empty(chunk_count.shape + (moment - 1,), dtype=dtype)
            for i in range(2, moment + 1):
                var_square[..., i - 2] = xp.nansum((in_chunk - avg) ** i, axis=axis, dtype=dtype,
                                                   keepdims=bool(op.keepdims))
            ctx[op.outputs[0].key] = (chunk_sum, chunk_count, var_square)

    @classmethod
    def execute_combine(cls, ctx, op):
        axis = cls.get_axis(op.axis)
        moment = op.moment
        dtype = op.dtype

        (_data, _count, _var_square), device_id, xp = as_same_device(
            ctx[op.inputs[0].key], device=op.device, ret_extra=True)
        empty = get_array_module(_data, nosparse=True).empty

        with device(device_id):
            chunk_count = xp.nansum(_count, axis=axis, dtype=np.int64, keepdims=bool(op.keepdims))
            chunk_sum = xp.nansum(_data, axis=axis, dtype=dtype, keepdims=bool(op.keepdims))
            avg = xp.true_divide(chunk_sum, chunk_count, dtype=dtype)
            avg_diff = xp.true_divide(_data, _count, dtype=dtype) - avg
            var_square = empty(chunk_count.shape + (moment - 1,), dtype=dtype)

            for m in range(2, moment + 1):
                var_square[..., m - 2] = reduce_var_square(_var_square, avg_diff, _count, op, axis, xp.nansum)

            ctx[op.outputs[0].key] = (chunk_sum, chunk_count, var_square)


class TensorNanVar(TensorReduction, TensorReductionMixin):
    _op_type_ = OperandDef.NANVAR

    _ddof = Int32Field('ddof')

    def __new__(cls, *args, **kwargs):
        if kwargs.get('stage') is not None:
            return TensorNanMoment(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, axis=None, dtype=None, keepdims=None, ddof=0, combine_size=None, **kw):
        super().__init__(_axis=axis, _dtype=dtype, _keepdims=keepdims, _ddof=ddof,
                         _combine_size=combine_size, **kw)

    @property
    def ddof(self):
        return self._ddof

    def _get_op_kw(self):
        kw = dict()
        kw['ddof'] = self.ddof
        return kw

    @classmethod
    def execute(cls, ctx, op):
        axis = cls.get_axis(op.axis)
        (in_chunk,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            ctx[op.outputs[0].key] = xp.nanvar(in_chunk, axis=axis, dtype=op.dtype, ddof=op.ddof,
                                               keepdims=bool(op.keepdims))


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=None, combine_size=None):
    """
    Compute the variance along the specified axis, while ignoring NaNs.

    Returns the variance of the tensor elements, a measure of the spread of
    a distribution.  The variance is computed for the flattened tensor by
    default, otherwise over the specified axis.

    For all-NaN slices or slices with zero degrees of freedom, NaN is
    returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    a : array_like
        Tensor containing numbers whose variance is desired.  If `a` is not a
        tensor, a conversion is attempted.
    axis : int, optional
        Axis along which the variance is computed.  The default is to compute
        the variance of the flattened array.
    dtype : data-type, optional
        Type to use in computing the variance.  For tensors of integer type
        the default is `float32`; for tensors of float types it is the same as
        the tensor type.
    out : Tensor, optional
        Alternate output tensor in which to place the result.  It must have
        the same shape as the expected output, but the type is cast if
        necessary.
    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of non-NaN
        elements. By default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
    combine_size: int, optional
        The number of chunks to combine.


    Returns
    -------
    variance : Tensor, see dtype parameter above
        If `out` is None, return a new tensor containing the variance,
        otherwise return a reference to the output tensor. If ddof is >= the
        number of non-NaN elements in a slice or the slice contains only
        NaNs, then the result for that slice is NaN.

    See Also
    --------
    std : Standard deviation
    mean : Average
    var : Variance while not ignoring NaNs
    nanstd, nanmean

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.,  ``var = mean(abs(x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used
    instead.  In standard statistical practice, ``ddof=1`` provides an
    unbiased estimator of the variance of a hypothetical infinite
    population.  ``ddof=0`` provides a maximum likelihood estimate of the
    variance for normally distributed variables.

    Note that for complex numbers, the absolute value is taken before
    squaring, so that the result is always real and nonnegative.

    For floating-point input, the variance is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32` (see example
    below).  Specifying a higher-accuracy accumulator using the ``dtype``
    keyword can alleviate this issue.

    For this function to work on sub-classes of Tensor, they must define
    `sum` with the kwarg `keepdims`

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array([[1, mt.nan], [3, 4]])
    >>> mt.nanvar(a).execute()
    1.5555555555555554
    >>> mt.nanvar(a, axis=0).execute()
    array([ 1.,  0.])
    >>> mt.nanvar(a, axis=1).execute()
    array([ 0.,  0.25])

    """
    a = astensor(a)
    if dtype is None:
        dtype = np.nanvar(np.ones((1,), dtype=a.dtype)).dtype
    op = TensorNanVar(axis=axis, dtype=dtype, keepdims=keepdims, ddof=ddof, combine_size=combine_size)
    return op(a, out=out)
