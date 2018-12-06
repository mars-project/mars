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

from ...core import Tensor
from ..datasource import tensor as astensor
from .tensordot import tensordot


def inner(a, b, out=None, sparse=None):
    """
    Returns the inner product of a and b for arrays of floating point types.

    Like the generic NumPy equivalent the product sum is over the last dimension
    of a and b. The first argument is not conjugated.

    """
    a, b = astensor(a), astensor(b)
    if a.isscalar() and b.isscalar():
        ret = a * b
    else:
        ret = tensordot(a, b, axes=(-1, -1), sparse=sparse)

    if out is None:
        return ret

    # set to out
    if not isinstance(out, Tensor):
        raise ValueError('`out` must be a Tensor, got {0} instead'.format(type(out)))
    out.data = ret.data
    return out


innerproduct = inner