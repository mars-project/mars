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


def result_type(*tensors_and_dtypes):
    """
    Returns the type that results from applying the NumPy
    type promotion rules to the arguments.

    Type promotion in Mars works similarly to the rules in languages
    like C++, with some slight differences.  When both scalars and
    arrays are used, the array's type takes precedence and the actual value
    of the scalar is taken into account.

    For example, calculating 3*a, where a is an array of 32-bit floats,
    intuitively should result in a 32-bit float output.  If the 3 is a
    32-bit integer, the NumPy rules indicate it can't convert losslessly
    into a 32-bit float, so a 64-bit float should be the result type.
    By examining the value of the constant, '3', we see that it fits in
    an 8-bit integer, which can be cast losslessly into the 32-bit float.

    Parameters
    ----------
    tensors_and_dtypes : list of tensors and dtypes
        The operands of some operation whose result type is needed.

    Returns
    -------
    out : dtype
        The result type.

    See also
    --------
    dtype, promote_types, min_scalar_type, can_cast

    Notes
    -----
    The specific algorithm used is as follows.

    Categories are determined by first checking which of boolean,
    integer (int/uint), or floating point (float/complex) the maximum
    kind of all the arrays and the scalars are.

    If there are only scalars or the maximum category of the scalars
    is higher than the maximum category of the arrays,
    the data types are combined with :func:`promote_types`
    to produce the return value.

    Otherwise, `min_scalar_type` is called on each array, and
    the resulting data types are all combined with :func:`promote_types`
    to produce the return value.

    The set of int values is not a subset of the uint values for types
    with the same number of bits, something not reflected in
    :func:`min_scalar_type`, but handled as a special case in `result_type`.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.result_type(3, mt.arange(7, dtype='i1'))
    dtype('int8')

    >>> mt.result_type('i4', 'c8')
    dtype('complex128')

    >>> mt.result_type(3.0, -2)
    dtype('float64')
    """
    from ..core import Tensor

    arrays_and_dtypes = [a.dtype if isinstance(a, Tensor) else a
                         for a in tensors_and_dtypes]
    return np.result_type(*arrays_and_dtypes)
