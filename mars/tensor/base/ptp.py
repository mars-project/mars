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

from ..utils import validate_axis, check_out_param
from ..datasource import tensor as astensor
from ..core import Tensor
from .ravel import ravel


def ptp(a, axis=None, out=None):
    """
    Range of values (maximum - minimum) along an axis.

    The name of the function comes from the acronym for 'peak to peak'.

    Parameters
    ----------
    a : array_like
        Input values.
    axis : int, optional
        Axis along which to find the peaks.  By default, flatten the
        array.
    out : array_like
        Alternative output tensor in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type of the output values will be cast if necessary.

    Returns
    -------
    ptp : Tensor
        A new tensor holding the result, unless `out` was
        specified, in which case a reference to `out` is returned.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.arange(4).reshape((2,2))
    >>> x.execute()
    array([[0, 1],
           [2, 3]])

    >>> mt.ptp(x, axis=0).execute()
    array([2, 2])

    >>> mt.ptp(x, axis=1).execute()
    array([1, 1])

    """
    a = astensor(a)

    if axis is None:
        a = ravel(a)
    else:
        validate_axis(a.ndim, axis)

    t = a.max(axis=axis) - a.min(axis=axis)

    if out is not None:
        if not isinstance(out, Tensor):
            raise TypeError('out should be Tensor object, got {0} instead'.format(type(out)))

        check_out_param(out, t, 'same_kind')
        out.data = t.data
        return out

    return t
