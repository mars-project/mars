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

from ..datasource import tensor as astensor
from ..base.broadcast_to import broadcast_to
from ..base.swapaxes import swapaxes


def average(a, axis=None, weights=None, returned=False):
    """
    Compute the weighted average along the specified axis.

    Parameters
    ----------
    a : array_like
        Tensor containing data to be averaged. If `a` is not a tensor, a
        conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to average `a`.  The default,
        axis=None, will average over all of the elements of the input tensor.
        If axis is negative it counts from the last to the first axis.

        If axis is a tuple of ints, averaging is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    weights : array_like, optional
        A tensor of weights associated with the values in `a`. Each value in
        `a` contributes to the average according to its associated weight.
        The weights tensor can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
    returned : bool, optional
        Default is `False`. If `True`, the tuple (`average`, `sum_of_weights`)
        is returned, otherwise only the average is returned.
        If `weights=None`, `sum_of_weights` is equivalent to the number of
        elements over which the average is taken.


    Returns
    -------
    average, [sum_of_weights] : tensor_type or double
        Return the average along the specified axis. When returned is `True`,
        return a tuple with the average as the first element and the sum
        of the weights as the second element. The return type is `Float`
        if `a` is of integer type, otherwise it is of the same type as `a`.
        `sum_of_weights` is of the same type as `average`.

    Raises
    ------
    ZeroDivisionError
        When all weights along axis are zero. See `numpy.ma.average` for a
        version robust to this type of error.
    TypeError
        When the length of 1D `weights` is not the same as the shape of `a`
        along axis.

    See Also
    --------
    mean

    Examples
    --------
    >>> import mars.tensor as mt

    >>> data = list(range(1,5))
    >>> data
    [1, 2, 3, 4]
    >>> mt.average(data).execute()
    2.5
    >>> mt.average(range(1,11), weights=range(10,0,-1)).execute()
    4.0

    >>> data = mt.arange(6).reshape((3,2))
    >>> data.execute()
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> mt.average(data, axis=1, weights=[1./4, 3./4]).execute()
    array([ 0.75,  2.75,  4.75])
    >>> mt.average(data, weights=[1./4, 3./4]).execute()
    Traceback (most recent call last):
    ...
    TypeError: Axis must be specified when shapes of a and weights differ.

    """
    from ..arithmetic import truediv, multiply

    a = astensor(a)

    if weights is None:
        avg = a.mean(axis)
        scl = avg.dtype.type(a.size / avg.size)
    else:
        wgt = astensor(weights)

        if issubclass(a.dtype.type, (np.integer, np.bool_)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)

        # sanity checks
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights "
                    "differ.")
            if wgt.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ.")
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis.")

            # setup wgt to broadcast along axis
            wgt = broadcast_to(wgt, (a.ndim - 1) * (1,) + wgt.shape)
            wgt = swapaxes(wgt, -1, axis)

        scl = wgt.sum(axis=axis, dtype=result_dtype)
        with np.errstate(divide='raise'):
            avg = truediv(multiply(a, wgt, dtype=result_dtype).sum(axis), scl)

    if returned:
        if scl.shape != avg.shape:
            scl = broadcast_to(scl, avg.shape)
        return avg, scl
    else:
        return avg
