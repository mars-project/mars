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
from .solve import solve


def inv(a):
    """
    Compute the (multiplicative) inverse of a matrix.
    Given a square matrix `a`, return the matrix `ainv` satisfying
    ``dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])``.
    Parameters
    ----------
    a : (..., M, M) array_like
        Matrix to be inverted.
    Returns
    -------
    ainv : (..., M, M) ndarray or matrix
        (Multiplicative) inverse of the matrix `a`.
    Raises
    ------
    LinAlgError
        If `a` is not square or inversion fails.
    Notes
    -----
    .. versionadded:: 1.8.0
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.
    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = np.array([[1., 2.], [3., 4.]])
    >>> ainv = mt.linalg.inv(a)
    >>> mt.allclose(mt.dot(a, ainv), mt.eye(2)).execute()
    True
    >>> mt.allclose(mt.dot(ainv, a), mt.eye(2)).execute()
    True
    >>> ainv.execute()
    array([[ -2. ,  1. ],
           [ 1.5, -0.5]])
    """
    from ..datasource import eye

    a = astensor(a)
    return solve(a, eye(a.shape[0], chunks=a.params.raw_chunks))
