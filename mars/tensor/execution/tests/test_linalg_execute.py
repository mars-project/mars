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

import unittest

import numpy as np
import scipy.sparse as sps

from mars.tensor.execution.core import Executor
from mars.tensor.expressions.datasource import tensor, diag, ones, arange
from mars.tensor.expressions.linalg import qr, svd, cholesky, norm, lu, \
    solve_triangular, solve, inv, tensordot, dot, inner, vdot, matmul
from mars.tensor.expressions.random import uniform
from mars.lib.sparse.core import issparse


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')

    def testQRExecution(self):
        data = np.random.randn(18, 6)

        a = tensor(data, chunks=(3, 6))
        q, r = qr(a)
        t = q.dot(r)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

        a = tensor(data, chunks=(9, 6))
        q, r = qr(a)
        t = q.dot(r)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

        a = tensor(data, chunks=3)
        q, r = qr(a)
        t = q.dot(r)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

    def testSVDExecution(self):
        data = np.random.randn(18, 6) + 1j * np.random.randn(18, 6)

        a = tensor(data, chunks=(9, 6))
        U, s, V = svd(a)
        t = U.dot(diag(s).dot(V))

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

        a = tensor(data, chunks=(18, 6))
        U, s, V = svd(a)
        t = U.dot(diag(s).dot(V))

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

        a = tensor(data, chunks=(2, 6))
        U, s, V = svd(a)
        t = U.dot(diag(s).dot(V))

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

    def testCholeskyExecution(self):
        data = np.array([[1, -2j], [2j, 5]])

        a = tensor(data, chunks=1)

        L = cholesky(a, lower=True)
        t = L.dot(L.T.conj())

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

        L = cholesky(a, lower=True)
        U = cholesky(a)
        t = L.dot(U)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

        a = tensor(data, chunks=2)

        L = cholesky(a, lower=True)
        U = cholesky(a)
        t = L.dot(U)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

        a = tensor(data, chunks=(1, 2))

        L = cholesky(a, lower=True)
        U = cholesky(a)
        t = L.dot(U)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

    def testLUExecution(self):
        np.random.seed(1)

        data = np.random.randint(1, 10, (6, 6))
        a = tensor(data, chunks=2)

        P, L, U = lu(a)
        t = P.dot(L).dot(U)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

        a = tensor(data, chunks=1)

        P, L, U = lu(a)
        t = P.dot(L).dot(U)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

        a = tensor(data, chunks=(1, 2))

        P, L, U = lu(a)
        t = P.dot(L).dot(U)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

        a = tensor(data, chunks=3)
        P, L, U = lu(a)
        t = P.dot(L).dot(U)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

    def testSolveTriangular(self):
        from mars.tensor import tril, triu
        np.random.seed(1)

        data1 = np.random.randint(1, 10, (20, 20))
        data2 = np.random.randint(1, 10, (20, ))

        A = tensor(data1, chunks=20)
        b = tensor(data2, chunks=20)

        x = solve_triangular(A, b)
        t = triu(A).dot(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data2)

        x = solve_triangular(A, b, lower=True)
        t = tril(A).dot(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data2)

        A = tensor(data1, chunks=10)
        b = tensor(data2, chunks=10)

        x = solve_triangular(A, b)
        t = triu(A).dot(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data2)

        x = solve_triangular(A, b, lower=True)
        t = tril(A).dot(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data2)

        data1 = np.random.randint(1, 10, (10, 10))
        data2 = np.random.randint(1, 10, (10, 5))

        A = tensor(data1, chunks=10)
        b = tensor(data2, chunks=10)

        x = solve_triangular(A, b)
        t = triu(A).dot(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data2)

        x = solve_triangular(A, b, lower=True)
        t = tril(A).dot(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data2)

        A = tensor(data1, chunks=3)
        b = tensor(data2, chunks=3)

        x = solve_triangular(A, b)
        t = triu(A).dot(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data2)

        x = solve_triangular(A, b, lower=True)
        t = tril(A).dot(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data2)

    def testSolve(self):
        import scipy.linalg
        np.random.seed(1)

        data1 = np.random.randint(1, 10, (20, 20))
        data2 = np.random.randint(1, 10, (20, ))

        A = tensor(data1, chunks=5)
        b = tensor(data2, chunks=5)

        x = solve(A, b)

        res = self.executor.execute_tensor(x, concat=True)[0]
        np.testing.assert_allclose(res, scipy.linalg.solve(data1, data2))
        res = self.executor.execute_tensor(A.dot(x), concat=True)[0]
        np.testing.assert_allclose(res, data2)

        data2 = np.random.randint(1, 10, (20, 5))

        A = tensor(data1, chunks=5)
        b = tensor(data2, chunks=5)

        x = solve(A, b)

        res = self.executor.execute_tensor(x, concat=True)[0]
        np.testing.assert_allclose(res, scipy.linalg.solve(data1, data2))
        res = self.executor.execute_tensor(A.dot(x), concat=True)[0]
        np.testing.assert_allclose(res, data2)

        data2 = np.random.randint(1, 10, (20, 20))

        A = tensor(data1, chunks=5)
        b = tensor(data2, chunks=5)

        x = solve(A, b)

        res = self.executor.execute_tensor(x, concat=True)[0]
        np.testing.assert_allclose(res, scipy.linalg.solve(data1, data2))
        res = self.executor.execute_tensor(A.dot(x), concat=True)[0]
        np.testing.assert_allclose(res, data2)

    def testSolveSymPos(self):
        import scipy.linalg
        np.random.seed(1)

        data = np.random.randint(1, 10, (20, 20))
        data_l = np.tril(data)
        data1 = data_l.dot(data_l.T)
        data2 = np.random.randint(1, 10, (20, ))

        A = tensor(data1, chunks=5)
        b = tensor(data2, chunks=5)

        x = solve(A, b, sym_pos=True)

        res = self.executor.execute_tensor(x, concat=True)[0]
        np.testing.assert_allclose(res, scipy.linalg.solve(data1, data2))
        res = self.executor.execute_tensor(A.dot(x), concat=True)[0]
        np.testing.assert_allclose(res, data2)

    def testInv(self):
        import scipy.linalg
        np.random.seed(1)

        data = np.random.randint(1, 10, (20, 20))

        A = tensor(data, chunks=5)
        inv_A = inv(A)

        res = self.executor.execute_tensor(inv_A, concat=True)[0]
        self.assertTrue(np.allclose(res, scipy.linalg.inv(data)))
        res = self.executor.execute_tensor(A.dot(inv_A), concat=True)[0]
        self.assertTrue(np.allclose(res, np.eye(data.shape[0], dtype=float)))

    def testNormExecution(self):
        d = np.arange(9) - 4
        d2 = d.reshape(3, 3)

        ma = [tensor(d, chunks=2), tensor(d, chunks=9),
              tensor(d2, chunks=(2, 3)), tensor(d2, chunks=3)]

        for i, a in enumerate(ma):
            data = d if i < 2 else d2
            for ord in (None, 'nuc', np.inf, -np.inf, 0, 1, -1, 2, -2):
                for axis in (0, 1, (0, 1)):
                    for keepdims in (True, False):
                        try:
                            expected = np.linalg.norm(data, ord=ord, axis=axis, keepdims=keepdims)
                            t = norm(a, ord=ord, axis=axis, keepdims=keepdims)
                            concat = t.ndim > 0
                            res = self.executor.execute_tensor(t, concat=concat)[0]

                            np.testing.assert_allclose(res, expected, atol=.0001)
                        except ValueError:
                            continue

        m = norm(tensor(d))
        expected = self.executor.execute_tensor(m)[0]
        res = np.linalg.norm(d)
        self.assertEqual(expected, res)

        d = uniform(-0.5, 0.5, size=(500, 2), chunks=50)
        inside = (norm(d, axis=1) < 0.5).sum().astype(float)
        t = inside / 500 * 4
        res = self.executor.execute_tensor(t)[0]
        self.assertAlmostEqual(res, 3.14, delta=1)

    def testTensordotExecution(self):
        a_data = np.arange(60).reshape(3, 4, 5)
        a = tensor(a_data, chunks=2)
        b_data = np.arange(24).reshape(4, 3, 2)
        b = tensor(b_data, chunks=2)

        axes = ([1, 0], [0, 1])
        c = tensordot(a, b, axes=axes)
        res = self.executor.execute_tensor(c)
        expected = np.tensordot(a_data, b_data, axes=axes)
        self.assertTrue(np.array_equal(res[0], expected[:2, :]))
        self.assertTrue(np.array_equal(res[1], expected[2:4, :]))
        self.assertTrue(np.array_equal(res[2], expected[4:, :]))

        a = ones((1000, 2000), chunks=500)
        b = ones((2000, 100), chunks=500)
        c = dot(a, b)
        res = self.executor.execute_tensor(c)
        expected = np.dot(np.ones((1000, 2000)), np.ones((2000, 100)))
        self.assertEqual(len(res), 2)
        self.assertTrue(np.array_equal(res[0], expected[:500, :]))
        self.assertTrue(np.array_equal(res[1], expected[500:, :]))

        a = ones((10, 8), chunks=2)
        b = ones((8, 10), chunks=2)
        c = a.dot(b)
        res = self.executor.execute_tensor(c)
        self.assertEqual(len(res), 25)
        for r in res:
            self.assertTrue(np.array_equal(r, np.tile([8], [2, 2])))

        a = ones((500, 500), chunks=500)
        b = ones((500, 100), chunks=500)
        c = a.dot(b)
        res = self.executor.execute_tensor(c)
        self.assertTrue(np.array_equal(res[0], np.tile([500], [500, 100])))

        raw_a = np.random.random((100, 200, 50))
        raw_b = np.random.random((200, 10, 100))
        a = tensor(raw_a, chunks=50)
        b = tensor(raw_b, chunks=33)
        c = tensordot(a, b, axes=((0, 1), (2, 0)))
        res = self.executor.execute_tensor(c, concat=True)
        expected = np.tensordot(raw_a, raw_b, axes=(c.op.a_axes, c.op.b_axes))
        self.assertTrue(np.allclose(res[0], expected))

        a = ones((1000, 2000), chunks=500)
        b = ones((100, 2000), chunks=500)
        c = inner(a, b)
        res = self.executor.execute_tensor(c)
        expected = np.inner(np.ones((1000, 2000)), np.ones((100, 2000)))
        self.assertEqual(len(res), 2)
        self.assertTrue(np.array_equal(res[0], expected[:500, :]))
        self.assertTrue(np.array_equal(res[1], expected[500:, :]))

    def testSparseDotExecution(self):
        a_data = sps.random(5, 9, density=.1)
        b_data = sps.random(9, 10, density=.2)
        a = tensor(a_data, chunks=2)
        b = tensor(b_data, chunks=3)

        c = dot(a, b)

        res = self.executor.execute_tensor(c, concat=True)[0]
        self.assertTrue(issparse(res))
        np.testing.assert_allclose(res.toarray(), a_data.dot(b_data).toarray())

        c2 = dot(a, b, sparse=False)

        res = self.executor.execute_tensor(c2, concat=True)[0]
        self.assertFalse(issparse(res))
        np.testing.assert_allclose(res, a_data.dot(b_data).toarray())

        c3 = tensordot(a, b.T, (-1, -1), sparse=False)

        res = self.executor.execute_tensor(c3, concat=True)[0]
        self.assertFalse(issparse(res))
        np.testing.assert_allclose(res, a_data.dot(b_data).toarray())

        c = inner(a, b.T)

        res = self.executor.execute_tensor(c, concat=True)[0]
        self.assertTrue(issparse(res))
        np.testing.assert_allclose(res.toarray(), a_data.dot(b_data).toarray())

        c = inner(a, b.T, sparse=False)

        res = self.executor.execute_tensor(c, concat=True)[0]
        self.assertFalse(issparse(res))
        np.testing.assert_allclose(res, a_data.dot(b_data).toarray())

    def testVdotExecution(self):
        a_data = np.array([1 + 2j, 3 + 4j])
        b_data = np.array([5 + 6j, 7 + 8j])
        a = tensor(a_data, chunks=1)
        b = tensor(b_data, chunks=1)

        t = vdot(a, b)

        res = self.executor.execute_tensor(t)[0]
        expected = np.vdot(a_data, b_data)
        np.testing.assert_equal(res, expected)

        a_data = np.array([[1, 4], [5, 6]])
        b_data = np.array([[4, 1], [2, 2]])
        a = tensor(a_data, chunks=1)
        b = tensor(b_data, chunks=1)

        t = vdot(a, b)

        res = self.executor.execute_tensor(t)[0]
        expected = np.vdot(a_data, b_data)
        np.testing.assert_equal(res, expected)

    def testMatmulExecution(self):
        data_a = np.random.randn(10, 20)
        data_b = np.random.randn(20)

        a = tensor(data_a, chunks=2)
        b = tensor(data_b, chunks=3)
        c = matmul(a, b)

        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.matmul(data_a, data_b)
        np.testing.assert_allclose(res, expected)

        data_a = np.random.randn(10, 20)
        data_b = np.random.randn(10)

        a = tensor(data_a, chunks=2)
        b = tensor(data_b, chunks=3)
        c = matmul(b, a)

        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.matmul(data_b, data_a)
        np.testing.assert_allclose(res, expected)

        data_a = np.random.randn(15, 1, 20, 30)
        data_b = np.random.randn(1, 11, 30, 20)

        a = tensor(data_a, chunks=12)
        b = tensor(data_b, chunks=13)
        c = matmul(a, b)

        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.matmul(data_a, data_b)
        np.testing.assert_allclose(res, expected, atol=.0001)

        a = arange(2 * 2 * 4, chunks=1).reshape((2, 2, 4))
        b = arange(2 * 2 * 4, chunks=1).reshape((2, 4, 2))
        c = matmul(a, b)

        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.matmul(np.arange(2 * 2 * 4).reshape(2, 2, 4),
                             np.arange(2 * 2 * 4).reshape(2, 4, 2))
        np.testing.assert_allclose(res, expected, atol=.0001)

        data_a = sps.random(10, 20)
        data_b = sps.random(20, 5)

        a = tensor(data_a, chunks=2)
        b = tensor(data_b, chunks=3)
        c = matmul(a, b)

        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.matmul(data_a.toarray(), data_b.toarray())
        np.testing.assert_allclose(res.toarray(), expected)

