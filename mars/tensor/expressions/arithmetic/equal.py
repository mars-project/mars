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

from .... import operands
from ...core import build_mode
from ..utils import inject_dtype
from ..datasource import tensor as astensor
from .core import TensorCompare, TensorCompareConstant


class TensorEqual(operands.Equal, TensorCompare):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorEqual, self).__init__(_casting=casting, _err=err,
                                          _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def constant_cls(cls):
        return TensorEqConstant


class TensorEqConstant(operands.EqConstant, TensorCompareConstant):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorEqConstant, self).__init__(_casting=casting, _err=err,
                                               _dtype=dtype, _sparse=sparse, **kw)


@inject_dtype(np.bool_)
def equal(x1, x2, out=None, where=None, **kwargs):
    """
    Return (x1 == x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input tensors of the same shape.
    out : Tensor, None, or tuple of Tensor and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : Tensor or bool
        Output tensor of bools, or a single bool if x1 and x2 are scalars.

    See Also
    --------
    not_equal, greater_equal, less_equal, greater, less

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.equal([0, 1, 3], mt.arange(3)).execute()
    array([ True,  True, False])

    What is compared are values, not types. So an int (1) and a tensor of
    length one can evaluate as True:

    >>> mt.equal(1, mt.ones(1))
    array([ True])
    """
    if build_mode().is_build_mode:
        return astensor(x1)._equals(x2)

    op = TensorEqual(**kwargs)
    return op(x1, x2, out=out, where=where)
