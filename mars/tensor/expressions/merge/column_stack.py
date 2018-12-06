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

from ..datasource import tensor as astensor
from .concatenate import concatenate


def column_stack(tup):
    """
    Stack 1-D tensors as columns into a 2-D tensor.

    Take a sequence of 1-D tensors and stack them as columns
    to make a single 2-D tensor. 2-D tensors are stacked as-is,
    just like with `hstack`.  1-D tensors are turned into 2-D columns
    first.

    Parameters
    ----------
    tup : sequence of 1-D or 2-D tensors.
        Tensors to stack. All of them must have the same first dimension.

    Returns
    -------
    stacked : 2-D tensor
        The tensor formed by stacking the given tensors.

    See Also
    --------
    stack, hstack, vstack, concatenate

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array((1,2,3))
    >>> b = mt.array((2,3,4))
    >>> mt.column_stack((a,b)).execute()
    array([[1, 2],
           [2, 3],
           [3, 4]])

    """
    from ..datasource import array

    arrays = []
    for a in tup:
        a = astensor(a)
        if a.ndim < 2:
            a = array(a, ndmin=2).T
        arrays.append(a)

    return concatenate(arrays, 1)
