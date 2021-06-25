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


def calc_svd_shapes(a):
    """
    Calculate output shapes of singular value decomposition.
    Follow the behavior of `numpy`:
    if a's shape is (6, 18), U's shape is (6, 6), s's shape is (6,), V's shape is (6, 18)
    if a's shape is (18, 6), U's shape is (18, 6), s's shape is (6,), V's shape is (6, 6)
    :param a: input tensor
    :return: (U.shape, s.shape, V.shape)
    """
    x, y = a.shape
    if x > y:
        return (x, y), (y,), (y, y)
    else:
        return (x, x), (x,), (x, y)


def svd_flip(u, v, u_based_decision=True):
    """
    Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u : Tensor
        u and v are the output of `linalg.svd` or
        `randomized_svd`, with matching inner dimensions
        so one can compute `mt.dot(u * s, v)`.

    v : Tensor
        u and v are the output of `linalg.svd` or
        `randomized_svd`, with matching inner dimensions
        so one can compute `mt.dot(u * s, v)`.

    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.


    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.

    """
    from ... import tensor as mt

    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = mt.argmax(mt.abs(u), axis=0)
        signs = mt.sign(u[max_abs_cols, np.arange(u.shape[1])])
        u *= signs
        v *= signs[:, mt.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = mt.argmax(mt.abs(v), axis=1)
        signs = mt.sign(v[np.arange(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, mt.newaxis]
    return u, v
