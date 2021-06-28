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

from ...core import ExecutableTuple
from ..datasource import tensor as astensor


def atleast_2d(*tensors):
    """
    View inputs as tensors with at least two dimensions.

    Parameters
    ----------
    tensors1, tensors2, ... : array_like
        One or more array-like sequences.  Non-tensor inputs are converted
        to tensors.  Tensors that already have two or more dimensions are
        preserved.

    Returns
    -------
    res, res2, ... : Tensor
        A tensor, or list of tensors, each with ``a.ndim >= 2``.
        Copies are avoided where possible, and views with two or more
        dimensions are returned.

    See Also
    --------
    atleast_1d, atleast_3d

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.atleast_2d(3.0).execute()
    array([[ 3.]])

    >>> x = mt.arange(3.0)
    >>> mt.atleast_2d(x).execute()
    array([[ 0.,  1.,  2.]])

    >>> mt.atleast_2d(1, [1, 2], [[1, 2]]).execute()
    [array([[1]]), array([[1, 2]]), array([[1, 2]])]

    """
    new_tensors = []
    for x in tensors:
        x = astensor(x)
        if x.ndim == 0:
            x = x[np.newaxis, np.newaxis]
        elif x.ndim == 1:
            x = x[np.newaxis, :]

        new_tensors.append(x)

    if len(new_tensors) == 1:
        return new_tensors[0]
    return ExecutableTuple(new_tensors)
