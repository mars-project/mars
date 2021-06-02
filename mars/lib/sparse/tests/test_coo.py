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
import pytest

from mars.lib.sparse.core import issparse
import mars.lib.sparse as mls

DEBUG = True

# cs: coo solution
# ds: dense solution (converted from coo solution using instance method toarray())
# da: dense answer (how to obtain: 1. convert operand from coo to dense using toarray() ; 2. operate using numpy library

c1 = mls.COONDArray(indices=np.asarray([(9, 5, 4), (7, 6, 5), (2, 2, 4), (7, 9, 4), (2, 2, 6)]),
                    data=np.asarray([3, 4, 5, 3, 1]),
                    shape=np.asarray([10, 11, 12]))

c2 = mls.COONDArray(indices=np.asarray([(9, 5, 4), (2, 2, 4), (7, 9, 4), (2, 2, 6), (8, 4, 9)]),
                    data=np.asarray([3, 5, 3, 1, 2]),
                    shape=np.asarray([10, 11, 12]))

c3 = mls.COONDArray(indices=np.asarray([tuple(i - 1 for i in list(ind)) for ind in c1.indices]),
                    data=np.asarray(c1.data),
                    shape=np.asarray(c1.shape))

# s1 = sps.coo_matrix()

# create dense numpy arrays with a similar shape and all zero values
d1 = np.zeros(shape=c1.shape)
d2 = np.zeros(shape=c2.shape)
d3 = np.zeros(shape=c3.shape)

# assign nnz val to the dense numpy array of each instance.
# d stands for dense
for i in range(len(c1.indices)):
    d1[tuple(c1.indices[i])] = c1.data[i]

for i in range(len(c2.indices)):
    d2[tuple(c2.indices[i])] = c2.data[i]

for i in range(len(c3.indices)):
    d3[tuple(c3.indices[i])] = c3.data[i]


def test_coo_creation():
    # assert(mls.issparse(c1))
    # type assertion only. REQUIRE: parameter assertion as well
    s = mls.COONDArray(c1)
    assert (isinstance(s, mls.COONDArray))
    assert (isinstance(s, mls.SparseNDArray))
    assert (mls.issparse(s))
    assert (s.issparse())
    # assert(mls.issparse(c2))


def test_to_array():
    # if issparse(a):
    #     a = a.toarray()
    # # hand-tune <b> && compare <b> with <a>
    # else:
    #     raise ValueError("input array is not sparse")
    nparr1 = c1.toarray()
    nparr2 = c2.toarray()
    np.testing.assert_allclose(nparr1, d1)
    np.testing.assert_allclose(nparr2, d2)


def assertArrayEqual(a, b):
    if issparse(a):
        a = a.toarray()
    else:
        a = np.asarray(a)
    if issparse(b):
        b = b.toarray()
    else:
        b = np.asarray(b)

    try:
        return np.testing.assert_equal(a, b)
    except AssertionError:
        return False


def test_coo_addition():
    # CASE0: SPARSE + SPARSE
    # cs: coo sum
    # cs = c1.__add__(c2)
    cs = c1 + c2
    # ds: dense sum; coo sum.todense()
    ds = cs.toarray()
    # da: dense answer
    da = d1 + d2
    np.testing.assert_allclose(ds, da)
    # dense_result = d1 + d2

    # CASE1: SPARSE + DENSE
    ds = c1 + d2
    # dense answer
    da = d1 + d2
    np.testing.assert_allclose(ds, da)

    const_val = 3

    # CASE2: SPARSE + CONSTANT, increment_all = False
    # const_val = 3
    # cs = c1 + const_val
    # ds = cs.toarray()
    # # dense answer
    # da = np.zeros(shape=c1.shape)
    # for i, v in zip(c1.indices, c1.values):
    #     da[i] = v + const_val
    # np.testing.assert_allclose(ds, da)

    # CASE3: SPARSE + CONSTANT, increment_all = True
    # da = d1 + const_val * np.ones(shape=c1.shape)
    cs = c1 + const_val
    # ds = cs.toarray()
    # NOTE that output type is changed to numpy ndarray from COONDArray given the nature of increment_all.
    # WILL improve usage of memory by adding a new attribute, offset
    ds = cs
    da = np.ones(shape=c1.shape) * const_val
    for i, v in zip(c1.indices, c1.data):
        da[i] += v
    np.testing.assert_allclose(ds, da)

    # CASE4: TypeError <- SPARSE + INCORRECT INPUT
    with pytest.raises(TypeError):
        cs = c1 + [1, 2, 3]
    # assertEqual(cs, None)
    # equivalent to:
    # assertRaises(TypeError, mls.COONDArray.__add__, c1, [1, 2, 3])


def test_coo_subtraction():
    # CASE0: SPARSE <- SPARSE - SPARSE
    cd = c1 - c2

    dd = cd.toarray()
    da = d1 - d2
    np.testing.assert_allclose(dd, da)

    # CASE1: DENSE <- SPARSE - DENSE
    # dense difference
    dd = c1 - d2

    # dense answer
    da = d1 - d2
    np.testing.assert_allclose(dd, da)

    const_val = 3

    # CASE2: DENSE <- SPARSE + CONSANT, increment_all = True
    cd = c1.__sub__(other=const_val)
    ds = cd
    da = np.ones(shape=c1.shape) * const_val * -1
    for i, v in zip(c1.indices, c1.data):
        da[i] += v
    np.testing.assert_allclose(ds, da)

    # CASE4: TypeError <- SPARSE + INCORRECT INPUT
    with pytest.raises(TypeError):
        _ = c1 - [1, 2, 3]  # noqa: F841


def test_coo_copy():
    # coo 1 copy
    c1c = c1.copy()

    # dense 1 copy
    d1c = c1c.toarray()

    np.testing.assert_allclose(d1c, d1)


def test_coo_transpose():
    # ct: coo transpose.
    # ('ct' denotes what is transposed in the coo form.)
    # dt: dense transpose.
    # ('dt' denotes what is transposed in the dense form. )
    # da: dense answer.
    # ('da' denotes the correct answer for the transpose operation)
    # CASE: Axes is None
    ct = c1.transpose()
    dt = ct.toarray()

    da = d1.transpose()
    np.testing.assert_allclose(dt, da)

    # CASE: Axes is a tuple of length two
    ct = c1.transpose((0, 2))
    dt = ct.toarray()

    da = c1.toarray()
    da = np.transpose(da, (2, 1, 0))  # the order of axis after tranposition.
    # INCORRECT: da = c1.toarray().transpose((1, 0))
    np.testing.assert_allclose(dt, da)


def test_coo_mul():
    # CASE: SPARSE * SPARSE
    # coo product
    cp = c1 * c2
    # dense product
    dp = cp.toarray()
    # dense answer
    da = np.multiply(d1, d2)
    np.testing.assert_allclose(dp, da)

    # CASE: SPARSE <- SPARSE * SPARSE, no matching index
    cp = c1 * c3
    dp = cp.toarray()
    da = np.multiply(d1, c3.toarray())
    np.testing.assert_allclose(dp, da)

    # CASE: SPARSE * DENSE

    cp = c1 * d2
    dp = cp.toarray()
    # dense answer
    da = np.multiply(d1, d2)
    np.testing.assert_allclose(dp, da)

    # CASE: SPARSE * CONSTANT
    multiplier = 3
    cp = c1 * multiplier
    dp = cp.toarray()
    da = np.zeros(shape=c1.shape)
    # print("multiplier: ")
    for i, v in zip(c1.indices, c1.data):
        # print(tuple(i))
        # print(da[tuple(i)])
        da[tuple(i)] = v * multiplier
        # print("i: ", i)
        # print("v: ", v)
        # print(da[i])
    # print(dp[np.nonzero(dp)])
    # print(da[np.nonzero(da)])
    np.testing.assert_allclose(dp, da)

    # CASE: SPARSE * CONSTANT, CONSTANT = 0
    cp = c1 * 0
    dp = cp.toarray()
    da = np.zeros(c1.shape)
    np.testing.assert_allclose(dp, da)

    # CASE: SPARSE * CONSTANT, CONSTANT = 1
    cp = c1 * 1
    dp = cp.toarray()
    da = d1
    np.testing.assert_allclose(dp, da)

    # CASE: Sparse * Incorrect Input
    with pytest.raises(TypeError):
        # cp = c1 * {'a': 1, 'b': 2, 'c': 3}
        cp = c1 * [1, 2, 3]
    # assertRaises(TypeError, mls.COONDArray.__mul__, c1, [1, 2, 3])


def test_coo_div():
    # CASE: SPARSE / SPARSE
    # 'ca' denotes the divided in the coo form.
    # 'cx' denoted the divisor in the coo form.
    # 'cq' denoted the quotient in the coo form
    # cq <- ca / cx
    # coo a; coo x; dense a; dense x
    ca = c1
    cx = c2
    da = d1
    dx = d2

    cq = ca / cx
    # cq = cq.toarray()

    with np.errstate(divide='ignore'):
        answer = np.true_divide(da, dx)

    # answer[np.isnan(answer)] = 0
    np.testing.assert_allclose(cq, answer)
    # print(cq[np.nonzero(cq==answer)])
    # print(answer[np.nonzero(cq == answer)])
    # print(answer)
    # return

    # CASE: SPARSE / SPARSE, no matching index
    cq = (c1 / c3)
    answer = d1 / c3.toarray()
    # answer[np.isnan(answer)] = 0
    np.testing.assert_allclose(cq, answer)

    # CASE: SPARSE / DENSE

    ca = c1
    cx = c2
    da = d1
    dx = d2

    result = ca / dx
    # result = result.toarray()

    # with np.errstate(divide='ignore'):
    with np.errstate(divide='ignore', invalid='ignore'):
        answer = np.true_divide(da, dx)

    # answer[np.isnan(answer)] = 0
    np.testing.assert_allclose(result, answer)

    # CASE: SPARSE / CONSTANT:
    ca = c1
    cx = c2
    da = d1
    const_val = 3
    dx = np.ones(shape=d2.shape) * const_val

    result = ca / const_val
    result = result.toarray()

    with np.errstate(divide='ignore'):
        answer = np.true_divide(da, dx)

    # answer[np.isnan(answer)] = 0
    # print(result[np.nonzero(result)])
    # print(answer[np.nonzero(answer)])
    np.testing.assert_allclose(result, answer)

    # CASE: SPARSE / CONSTANT, CONSTANT = 0
    with pytest.raises(TypeError):
        result = c1 / 0

    # CASE: SPARSE / CONSTANT, CONSTANT = 1
    result = (c1 / 1).toarray()
    answer = d1
    np.testing.assert_allclose(result, answer)

    # CASE: SPARSE / INCORRECT TYPE
    with pytest.raises(TypeError):
        result = c1 / [1, 2, 3]


def test_raw():
    raw_c1 = c1.raw
    np.testing.assert_allclose(raw_c1[0], c1.indices)
    np.testing.assert_allclose(raw_c1[1], c1.data)
    np.testing.assert_allclose(raw_c1[2], c1.shape)
