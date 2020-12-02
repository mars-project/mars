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
from .core import TensorReduction, TensorReductionMixin


class TensorAll(TensorReduction, TensorReductionMixin):
    _op_type_ = OperandDef.ALL
    _func_name = 'all'

    def __init__(self, axis=None, dtype=None, keepdims=None, combine_size=None, stage=None, **kw):
        stage = self._rewrite_stage(stage)
        super().__init__(_axis=axis, _dtype=dtype, _keepdims=keepdims,
                         _combine_size=combine_size, _stage=stage, **kw)


def all(a, axis=None, out=None, keepdims=None, combine_size=None):
    """
    Test whether all array elements along a given axis evaluate to True.

    Parameters
    ----------
    a : array_like
        Input tensor or object that can be converted to a tensor.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical AND reduction is performed.
        The default (`axis` = `None`) is to perform a logical AND over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.

        If this is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : Tensor, optional
        Alternate output tensor in which to place the result.
        It must have the same shape as the expected output and its
        type is preserved (e.g., if ``dtype(out)`` is float, the result
        will consist of 0.0's and 1.0's).  See `doc.ufuncs` (Section
        "Output arguments") for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input tensor.

        If the default value is passed, then `keepdims` will not be
        passed through to the `all` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-classes `sum` method does not implement `keepdims` any
        exceptions will be raised.
    combine_size: int, optional
        The number of chunks to combine.

    Returns
    -------
    all : Tensor, bool
        A new boolean or tensor is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    See Also
    --------
    Tensor.all : equivalent method

    any : Test whether any element along a given axis evaluates to True.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity
    evaluate to `True` because these are not equal to zero.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.all([[True,False],[True,True]]).execute()
    False

    >>> mt.all([[True,False],[True,True]], axis=0).execute()
    array([ True, False])

    >>> mt.all([-1, 4, 5]).execute()
    True

    >>> mt.all([1.0, mt.nan]).execute()
    True

    """
    a = astensor(a)
    if a.dtype == np.object_:
        dtype = a.dtype
    else:
        dtype = np.dtype(bool)
    op = TensorAll(axis=axis, dtype=dtype, keepdims=keepdims, combine_size=combine_size)
    return op(a, out=out)
