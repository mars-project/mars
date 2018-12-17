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

from mars.lib.sparse.core import issparse
import mars.lib.sparse as mls
from mars.tests.core import TestBase

DEBUG = True

# cs: coo solution
# ds: dense solution (converted from coo solution using instance method toarray())
# da: dense answer (how to obtain: 1. convert operand from coo to dense using toarray() ; 2. operate using numpy library


class Test(TestBase):
    def setUp(self):
        self.c1 = mls.COONDArray(indices=np.asarray([(9, 5, 4), (7, 6, 5), (2, 2, 4), (7, 9, 4), (2, 2, 6)]),
                                 data=np.asarray([3, 4, 5, 3, 1]),
                                 shape=np.asarray([10, 11, 12])
                                 )

        self.c2 = mls.COONDArray(indices=np.asarray([(9, 5, 4), (2, 2, 4), (7, 9, 4), (2, 2, 6), (8, 4, 9)]),
                                 data=np.asarray([3, 5, 3, 1, 2]),
                                 shape=np.asarray([10, 11, 12])
                                 )

        self.c3 = mls.COONDArray(indices=np.asarray([tuple(i - 1 for i in list(ind)) for ind in self.c1.indices]),
                                 data=np.asarray(self.c1.data),
                                 shape=np.asarray(self.c1.shape)
                                 )

        # self.s1 = sps.coo_matrix()

        # create dense numpy arrays with a similar shape and all zero values
        self.d1 = np.zeros(shape=self.c1.shape)
        self.d2 = np.zeros(shape=self.c2.shape)
        self.d3 = np.zeros(shape=self.c3.shape)

        # assign nnz val to the dense numpy array of each instance.
        # d stands for dense
        for i in range(len(self.c1.indices)):
            self.d1[tuple(self.c1.indices[i])] = self.c1.data[i]

        for i in range(len(self.c2.indices)):
            self.d2[tuple(self.c2.indices[i])] = self.c2.data[i]

        for i in range(len(self.c3.indices)):
            self.d3[tuple(self.c3.indices[i])] = self.c3.data[i]

    def testCooCreation(self):
        # self.assert(mls.issparse(self.c1))
        # type assertion only. REQUIRE: parameter assertion as well
        s = mls.COONDArray(self.c1)
        assert (isinstance(s, mls.COONDArray))
        assert (isinstance(s, mls.SparseNDArray))
        assert (mls.issparse(s))
        assert (s.issparse())
        # assert(mls.issparse(self.c2))
    # update to new numpy ndarray
    def testToArray(self):
        # if issparse(a):
        #     a = a.toarray()
        # # hand-tune <b> && compare <b> with <a>
        # else:
        #     raise ValueError("input array is not sparse")
        nparr1 = self.c1.toarray()
        nparr2 = self.c2.toarray()
        np.testing.assert_allclose(nparr1, self.d1)
        np.testing.assert_allclose(nparr2, self.d2)

    def assertArrayEqual(self, a, b):
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

    def testCooAddition(self):
        # CASE0: SPARSE + SPARSE
        # cs: coo sum
        # cs = self.c1.__add__(self.c2)
        cs = self.c1 + self.c2
        # ds: dense sum; coo sum.todense()
        ds = cs.toarray()
        # da: dense answer
        da = self.d1 + self.d2
        np.testing.assert_allclose(ds, da)
        # dense_result = self.d1 + self.d2

        # CASE1: SPARSE + DENSE
        ds = self.c1 + self.d2
        # dense answer
        da = self.d1 + self.d2
        np.testing.assert_allclose(ds, da)

        const_val = 3

        # CASE2: SPARSE + CONSTANT, increment_all = False
        # const_val = 3
        # cs = self.c1 + const_val
        # ds = cs.toarray()
        # # dense answer
        # da = np.zeros(shape=self.c1.shape)
        # for i, v in zip(self.c1.indices, self.c1.values):
        #     da[i] = v + const_val
        # np.testing.assert_allclose(ds, da)

        # CASE3: SPARSE + CONSTANT, increment_all = True
        # da = self.d1 + const_val * np.ones(shape=self.c1.shape)
        cs = self.c1 + const_val
        # ds = cs.toarray()
        # NOTE that output type is changed to numpy ndarray from COONDArray given the nature of increment_all.
        # WILL improve usage of memory by adding a new attribute, offset
        ds = cs
        da = np.ones(shape=self.c1.shape) * const_val
        for i, v in zip(self.c1.indices, self.c1.data):
            da[i] += v
        np.testing.assert_allclose(ds, da)

        # CASE4: TypeError <- SPARSE + INCORRECT INPUT
        with self.assertRaises(TypeError):
            cs = self.c1 + [1, 2, 3]
        # self.assertEqual(cs, None)
        # equivalent to:
        # self.assertRaises(TypeError, mls.COONDArray.__add__, self.c1, [1, 2, 3])

    # see testCooAddition for references of variable naming
    def testCooSubtraction(self):
        # CASE0: SPARSE <- SPARSE - SPARSE
        cd = self.c1 - self.c2

        dd = cd.toarray()
        da = self.d1 - self.d2
        np.testing.assert_allclose(dd, da)

        # CASE1: DENSE <- SPARSE - DENSE
        # dense difference
        dd = self.c1 - self.d2

        # dense answer
        da = self.d1 - self.d2
        np.testing.assert_allclose(dd, da)

        const_val = 3

        # CASE2: DENSE <- SPARSE + CONSANT, increment_all = True
        cd = self.c1.__sub__(other=const_val)
        ds = cd
        da = np.ones(shape=self.c1.shape) * const_val * -1
        for i, v in zip(self.c1.indices, self.c1.data):
            da[i] += v
        np.testing.assert_allclose(ds, da)

        # CASE4: TypeError <- SPARSE + INCORRECT INPUT
        with self.assertRaises(TypeError):
            cs = self.c1 - [1, 2, 3]

    def testCooCopy(self):
        # coo 1 copy
        c1c = self.c1.copy()

        # dense 1 copy
        d1c = c1c.toarray()

        np.testing.assert_allclose(d1c, self.d1)

    def testCooTranspose(self):
        # ct: coo transpose.
        # ('ct' denotes what is transposed in the coo form.)
        # dt: dense transpose.
        # ('dt' denotes what is transposed in the dense form. )
        # da: dense answer.
        # ('da' denotes the correct answer for the transpose operation)
        # CASE: Axes is None
        ct = self.c1.transpose()
        dt = ct.toarray()

        da = self.d1.transpose()
        np.testing.assert_allclose(dt, da)

        # CASE: Axes is a tuple of length two
        ct = self.c1.transpose((0, 2))
        dt = ct.toarray()

        da = self.c1.toarray()
        da = np.transpose(da, (2, 1, 0))  # the order of axis after tranposition.
        # INCORRECT: da = self.c1.toarray().transpose((1, 0))
        np.testing.assert_allclose(dt, da)

    def testCooMul(self):
        # CASE: SPARSE * SPARSE
        # coo product
        cp = self.c1 * self.c2
        # dense product
        dp = cp.toarray()
        # dense answer
        da = np.multiply(self.d1, self.d2)
        np.testing.assert_allclose(dp, da)

        # CASE: SPARSE <- SPARSE * SPARSE, no matching index
        cp = self.c1 * self.c3
        dp = cp.toarray()
        da = np.multiply(self.d1, self.c3.toarray())
        np.testing.assert_allclose(dp, da)

        # CASE: SPARSE * DENSE

        cp = self.c1 * self.d2
        dp = cp.toarray()
        # dense answer
        da = np.multiply(self.d1, self.d2)
        np.testing.assert_allclose(dp, da)

        # CASE: SPARSE * CONSTANT
        multiplier = 3
        cp = self.c1 * multiplier
        dp = cp.toarray()
        da = np.zeros(shape=self.c1.shape)
        # print("multiplier: ")
        for i, v in zip(self.c1.indices, self.c1.data):
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
        cp = self.c1 * 0
        dp = cp.toarray()
        da = np.zeros(self.c1.shape)
        np.testing.assert_allclose(dp, da)

        # CASE: SPARSE * CONSTANT, CONSTANT = 1
        cp = self.c1 * 1
        dp = cp.toarray()
        da = self.d1
        np.testing.assert_allclose(dp, da)

        # CASE: Sparse * Incorrect Input
        with self.assertRaises(TypeError):
            # cp = self.c1 * {'a': 1, 'b': 2, 'c': 3}
            cp = self.c1 * [1, 2, 3]
        # self.assertRaises(TypeError, mls.COONDArray.__mul__, self.c1, [1, 2, 3])

    def testCooDiv(self):
        # CASE: SPARSE / SPARSE
        # 'ca' denotes the divided in the coo form.
        # 'cx' denoted the divisor in the coo form.
        # 'cq' denoted the quotient in the coo form
        # cq <- ca / cx
        # coo a; coo x; dense a; dense x
        ca = self.c1
        cx = self.c2
        da = self.d1
        dx = self.d2

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
        cq = (self.c1 / self.c3)
        answer = self.d1 / self.c3.toarray()
        # answer[np.isnan(answer)] = 0
        np.testing.assert_allclose(cq, answer)

        # CASE: SPARSE / DENSE

        ca = self.c1
        cx = self.c2
        da = self.d1
        dx = self.d2

        result = ca / dx
        # result = result.toarray()

        # with np.errstate(divide='ignore'):
        with np.errstate(divide='ignore', invalid='ignore'):
            answer = np.true_divide(da, dx)

        # answer[np.isnan(answer)] = 0
        np.testing.assert_allclose(result, answer)

        # CASE: SPARSE / CONSTANT:
        ca = self.c1
        cx = self.c2
        da = self.d1
        const_val = 3
        dx = np.ones(shape=self.d2.shape) * const_val

        result = ca / const_val
        result = result.toarray()

        with np.errstate(divide='ignore'):
            answer = np.true_divide(da, dx)

        # answer[np.isnan(answer)] = 0
        # print(result[np.nonzero(result)])
        # print(answer[np.nonzero(answer)])
        np.testing.assert_allclose(result, answer)

        # CASE: SPARSE / CONSTANT, CONSTANT = 0
        with self.assertRaises(TypeError):
            result = self.c1 / 0

        # CASE: SPARSE / CONSTANT, CONSTANT = 1
        result = (self.c1 / 1).toarray()
        answer = self.d1
        np.testing.assert_allclose(result, answer)

        # CASE: SPARSE / INCORRECT TYPE
        with self.assertRaises(TypeError):
            result = self.c1 / [1, 2, 3]

    ################################################
    ########## supplement uncovered lines:##########
    ################################################

    def testRaw(self):
        raw_c1 = self.c1.raw
        np.testing.assert_allclose(raw_c1[0], self.c1.indices)
        np.testing.assert_allclose(raw_c1[1], self.c1.data)
        np.testing.assert_allclose(raw_c1[2], self.c1.shape)
