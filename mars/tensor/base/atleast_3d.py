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

from ..datasource import tensor as astensor


def atleast_3d(*tensors):
    """
    View inputs as tensors with at least three dimensions.

    Parameters
    ----------
    tensors1, tensors2, ... : array_like
        One or more tensor-like sequences.  Non-tensor inputs are converted to
        tensors.  Tensors that already have three or more dimensions are
        preserved.

    Returns
    -------
    res1, res2, ... : Tensor
        A tensor, or list of tensors, each with ``a.ndim >= 3``.  Copies are
        avoided where possible, and views with three or more dimensions are
        returned.  For example, a 1-D tensor of shape ``(N,)`` becomes a view
        of shape ``(1, N, 1)``, and a 2-D tensor of shape ``(M, N)`` becomes a
        view of shape ``(M, N, 1)``.

    See Also
    --------
    atleast_1d, atleast_2d

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.session import new_session

    >>> sess = new_session()

    >>> mt.atleast_3d(3.0).execute()
    array([[[ 3.]]])

    >>> x = mt.arange(3.0)
    >>> mt.atleast_3d(x).shape
    (1, 3, 1)

    >>> x = mt.arange(12.0).reshape(4,3)
    >>> mt.atleast_3d(x).shape
    (4, 3, 1)

    >>> for arr in sess.run(mt.atleast_3d([1, 2], [[1, 2]], [[[1, 2]]])):
    ...     print(arr, arr.shape)
    ...
    [[[1]
      [2]]] (1, 2, 1)
    [[[1]
      [2]]] (1, 2, 1)
    [[[1 2]]] (1, 1, 2)

    """
    new_tensors = []
    for x in tensors:
        x = astensor(x)
        if x.ndim == 0:
            x = x[np.newaxis, np.newaxis, np.newaxis]
        elif x.ndim == 1:
            x = x[np.newaxis, :, np.newaxis]
        elif x.ndim == 2:
            x = x[:, :, None]

        new_tensors.append(x)

    if len(new_tensors) == 1:
        return new_tensors[0]
    return new_tensors
