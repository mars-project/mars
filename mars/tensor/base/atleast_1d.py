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

from ...core import ExecutableTuple
from ..datasource import tensor as astensor


def atleast_1d(*tensors):
    """
    Convert inputs to tensors with at least one dimension.

    Scalar inputs are converted to 1-dimensional tensors, whilst
    higher-dimensional inputs are preserved.

    Parameters
    ----------
    tensors1, tensors2, ... : array_like
        One or more input tensors.

    Returns
    -------
    ret : Tensor
        An tensor, or list of tensors, each with ``a.ndim >= 1``.
        Copies are made only if necessary.

    See Also
    --------
    atleast_2d, atleast_3d

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.atleast_1d(1.0).execute()
    array([ 1.])

    >>> x = mt.arange(9.0).reshape(3,3)
    >>> mt.atleast_1d(x).execute()
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  8.]])
    >>> mt.atleast_1d(x) is x
    True

    >>> mt.atleast_1d(1, [3, 4]).execute()
    [array([1]), array([3, 4])]

    """
    new_tensors = []
    for x in tensors:
        x = astensor(x)
        if x.ndim == 0:
            x = x[np.newaxis]

        new_tensors.append(x)

    if len(new_tensors) == 1:
        return new_tensors[0]
    return ExecutableTuple(new_tensors)
