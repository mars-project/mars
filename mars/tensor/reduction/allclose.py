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


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Returns True if two tensors are element-wise equal within a tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    If either array contains one or more NaNs, False is returned.
    Infs are treated as equal if they are in the same place and of the same
    sign in both tensors.

    Parameters
    ----------
    a, b : array_like
        Input tensors to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output tensor.

    Returns
    -------
    allclose : bool
        Returns True if the two tensors are equal within the given
        tolerance; False otherwise.

    See Also
    --------
    isclose, all, any, equal

    Notes
    -----
    If the following equation is element-wise True, then allclose returns
    True.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    The above equation is not symmetric in `a` and `b`, so that
    ``allclose(a, b)`` might be different from ``allclose(b, a)`` in
    some rare cases.

    The comparison of `a` and `b` uses standard broadcasting, which
    means that `a` and `b` need not have the same shape in order for
    ``allclose(a, b)`` to evaluate to True.  The same is true for
    `equal` but not `array_equal`.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.allclose([1e10,1e-7], [1.00001e10,1e-8]).execute()
    False
    >>> mt.allclose([1e10,1e-8], [1.00001e10,1e-9]).execute()
    True
    >>> mt.allclose([1e10,1e-8], [1.0001e10,1e-9]).execute()
    False
    >>> mt.allclose([1.0, mt.nan], [1.0, mt.nan]).execute()
    False
    >>> mt.allclose([1.0, mt.nan], [1.0, mt.nan], equal_nan=True).execute()
    True

    """
    from ..arithmetic.isclose import isclose
    from .all import all

    return all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))
