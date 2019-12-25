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

import unittest

import numpy as np
import scipy.sparse as sps

from mars.executor import Executor
from mars.tensor.datasource import tensor, diag, ones, arange
from mars.tensor.linalg import qr, svd, cholesky, norm, lu, \
    solve_triangular, solve, inv, tensordot, dot, inner, vdot, matmul, randomized_svd
from mars.tensor.random import uniform
from mars.learn.datasets.samples_generator import make_low_rank_matrix
from mars.lib.sparse import issparse, SparseNDArray


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')

    def testQRExecution(self):
        data = np.random.randn(18, 6)

        a = tensor(data, chunk_size=(3, 6))
        q, r = qr(a)
        t = q.dot(r)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

        a = tensor(data, chunk_size=(9, 6))
        q, r = qr(a)
        t = q.dot(r)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

        a = tensor(data, chunk_size=3)
        q, r = qr(a)
        t = q.dot(r)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

        # test for Short-and-Fat QR
        data = np.random.randn(6, 18)

        a = tensor(data, chunk_size=(6, 9))
        q, r = qr(a, method='sfqr')
        t = q.dot(r)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

        a = tensor(data, chunk_size=(3, 3))
        q, r = qr(a, method='sfqr')
        t = q.dot(r)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

        a = tensor(data, chunk_size=(6, 3))
        q, r = qr(a, method='sfqr')
        t = q.dot(r)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

    def testSVDExecution(self):
        data = np.random.randn(18, 6) + 1j * np.random.randn(18, 6)

        a = tensor(data, chunk_size=(9, 6))
        U, s, V = svd(a)
        t = U.dot(diag(s).dot(V))

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

        a = tensor(data, chunk_size=(18, 6))
        U, s, V = svd(a)
        t = U.dot(diag(s).dot(V))

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

        a = tensor(data, chunk_size=(2, 6))
        U, s, V = svd(a)
        t = U.dot(diag(s).dot(V))

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

        data = np.random.randn(6, 18) + 1j * np.random.randn(6, 18)

        a = tensor(data)
        U, s, V = svd(a)
        t = U.dot(diag(s).dot(V))

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, data))

        # test for matrix of ones
        data = np.ones((20, 10))

        a = tensor(data, chunk_size=10)
        s = svd(a)[1]
        res = self.executor.execute_tensor(s, concat=True)[0]
        expected = np.linalg.svd(a)[1]
        np.testing.assert_array_almost_equal(res, expected)

    def testRandomizedSVDExecution(self):
        n_samples = 100
        n_features = 500
        rank = 5
        k = 10
        for dtype in (np.int32, np.int64, np.float32, np.float64):
            # generate a matrix X of approximate effective rank `rank` and no noise
            # component (very structured signal):
            X = make_low_rank_matrix(n_samples=n_samples, n_features=n_features,
                                     effective_rank=rank, tail_strength=0.0,
                                     random_state=0).astype(dtype, copy=False)
            self.assertEqual(X.shape, (n_samples, n_features))
            dtype = np.dtype(dtype)
            decimal = 5 if dtype == np.float32 else 7

            # compute the singular values of X using the slow exact method
            X_res = self.executor.execute_tensor(X, concat=True)[0]
            U, s, V = np.linalg.svd(X_res, full_matrices=False)

            # Convert the singular values to the specific dtype
            U = U.astype(dtype, copy=False)
            s = s.astype(dtype, copy=False)
            V = V.astype(dtype, copy=False)

            for normalizer in ['auto', 'LU', 'QR']:  # 'none' would not be stable
                # compute the singular values of X using the fast approximate method
                Ua, sa, Va = randomized_svd(
                    X, k, power_iteration_normalizer=normalizer, random_state=0)

                # If the input dtype is float, then the output dtype is float of the
                # same bit size (f32 is not upcast to f64)
                # But if the input dtype is int, the output dtype is float64
                if dtype.kind == 'f':
                    self.assertEqual(Ua.dtype, dtype)
                    self.assertEqual(sa.dtype, dtype)
                    self.assertEqual(Va.dtype, dtype)
                else:
                    self.assertEqual(Ua.dtype, np.float64)
                    self.assertEqual(sa.dtype, np.float64)
                    self.assertEqual(Va.dtype, np.float64)

                self.assertEqual(Ua.shape, (n_samples, k))
                self.assertEqual(sa.shape, (k,))
                self.assertEqual(Va.shape, (k, n_features))

                # ensure that the singular values of both methods are equal up to the
                # real rank of the matrix
                sa_res = self.executor.execute_tensor(sa, concat=True)[0]
                np.testing.assert_almost_equal(s[:k], sa_res, decimal=decimal)

                # check the singular vectors too (while not checking the sign)
                dot_res = self.executor.execute_tensor(dot(Ua, Va), concat=True)[0]
                np.testing.assert_almost_equal(np.dot(U[:, :k], V[:k, :]), dot_res,
                                               decimal=decimal)

    def testCholeskyExecution(self):
        data = np.random.randint(1, 10, (10, 10))
        symmetric_data = data.dot(data.T)

        a = tensor(symmetric_data, chunk_size=5)

        U = cholesky(a)
        t = U.T.dot(U)

        res_u = self.executor.execute_tensor(U, concat=True)[0]
        np.testing.assert_allclose(np.triu(res_u), res_u)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, symmetric_data))

        L = cholesky(a, lower=True)
        U = cholesky(a)
        t = L.dot(U)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, symmetric_data))

        a = tensor(symmetric_data, chunk_size=2)

        L = cholesky(a, lower=True)
        U = cholesky(a)
        t = L.dot(U)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, symmetric_data)

        a = tensor(symmetric_data, chunk_size=(1, 2))

        L = cholesky(a, lower=True)
        U = cholesky(a)
        t = L.dot(U)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, symmetric_data)

        a = tensor(symmetric_data, chunk_size=4)

        L = cholesky(a, lower=True)
        U = cholesky(a)
        t = L.dot(U)

        res_u = self.executor.execute_tensor(U, concat=True)[0]
        np.testing.assert_allclose(np.triu(res_u), res_u)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, symmetric_data)

        a = tensor(symmetric_data, chunk_size=3)

        L = cholesky(a, lower=True)
        U = cholesky(a)
        t = L.dot(U)

        res_u = self.executor.execute_tensor(U, concat=True)[0]
        np.testing.assert_allclose(np.triu(res_u), res_u)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, symmetric_data)

    def testLUExecution(self):
        np.random.seed(1)

        # square matrix
        data = np.random.randint(1, 10, (6, 6))

        a = tensor(data)
        P, L, U = lu(a)

        # check lower and upper triangular matrix
        result_l = self.executor.execute_tensor(L, concat=True)[0]
        result_u = self.executor.execute_tensor(U, concat=True)[0]

        np.testing.assert_allclose(np.tril(result_l), result_l)
        np.testing.assert_allclose(np.triu(result_u), result_u)

        t = P.dot(L).dot(U)
        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

        a = tensor(data, chunk_size=2)
        P, L, U = lu(a)

        # check lower and upper triangular matrix
        result_l = self.executor.execute_tensor(L, concat=True)[0]
        result_u = self.executor.execute_tensor(U, concat=True)[0]

        np.testing.assert_allclose(np.tril(result_l), result_l)
        np.testing.assert_allclose(np.triu(result_u), result_u)

        t = P.dot(L).dot(U)
        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

        a = tensor(data, chunk_size=(2, 3))
        P, L, U = lu(a)

        # check lower and upper triangular matrix
        result_l = self.executor.execute_tensor(L, concat=True)[0]
        result_u = self.executor.execute_tensor(U, concat=True)[0]

        np.testing.assert_allclose(np.tril(result_l), result_l)
        np.testing.assert_allclose(np.triu(result_u), result_u)

        t = P.dot(L).dot(U)
        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

        a = tensor(data, chunk_size=4)
        P, L, U = lu(a)

        # check lower and upper triangular matrix
        result_l = self.executor.execute_tensor(L, concat=True)[0]
        result_u = self.executor.execute_tensor(U, concat=True)[0]

        np.testing.assert_allclose(np.tril(result_l), result_l)
        np.testing.assert_allclose(np.triu(result_u), result_u)

        t = P.dot(L).dot(U)
        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

        # shape[0] > shape[1]
        data = np.random.randint(1, 10, (10, 6))

        a = tensor(data)
        P, L, U = lu(a)

        # check lower and upper triangular matrix
        result_l = self.executor.execute_tensor(L, concat=True)[0]
        result_u = self.executor.execute_tensor(U, concat=True)[0]

        np.testing.assert_allclose(np.tril(result_l), result_l)
        np.testing.assert_allclose(np.triu(result_u), result_u)

        t = P.dot(L).dot(U)
        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

        a = tensor(data, chunk_size=2)
        P, L, U = lu(a)

        # check lower and upper triangular matrix
        result_l = self.executor.execute_tensor(L, concat=True)[0]
        result_u = self.executor.execute_tensor(U, concat=True)[0]

        np.testing.assert_allclose(np.tril(result_l), result_l)
        np.testing.assert_allclose(np.triu(result_u), result_u)

        t = P.dot(L).dot(U)
        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

        a = tensor(data, chunk_size=(2, 3))
        P, L, U = lu(a)

        # check lower and upper triangular matrix
        result_l = self.executor.execute_tensor(L, concat=True)[0]
        result_u = self.executor.execute_tensor(U, concat=True)[0]

        np.testing.assert_allclose(np.tril(result_l), result_l)
        np.testing.assert_allclose(np.triu(result_u), result_u)

        t = P.dot(L).dot(U)
        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

        a = tensor(data, chunk_size=4)
        P, L, U = lu(a)

        # check lower and upper triangular matrix
        result_l, result_u = self.executor.execute_tensors([L, U])

        np.testing.assert_allclose(np.tril(result_l), result_l)
        np.testing.assert_allclose(np.triu(result_u), result_u)

        t = P.dot(L).dot(U)
        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

        # shape[0] < shape[1]
        data = np.random.randint(1, 10, (6, 10))

        a = tensor(data)
        P, L, U = lu(a)

        # check lower and upper triangular matrix
        result_l = self.executor.execute_tensor(L, concat=True)[0]
        result_u = self.executor.execute_tensor(U, concat=True)[0]

        np.testing.assert_allclose(np.tril(result_l), result_l)
        np.testing.assert_allclose(np.triu(result_u), result_u)

        t = P.dot(L).dot(U)
        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

        a = tensor(data, chunk_size=2)
        P, L, U = lu(a)

        # check lower and upper triangular matrix
        result_l = self.executor.execute_tensor(L, concat=True)[0]
        result_u = self.executor.execute_tensor(U, concat=True)[0]

        np.testing.assert_allclose(np.tril(result_l), result_l)
        np.testing.assert_allclose(np.triu(result_u), result_u)

        t = P.dot(L).dot(U)
        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

        a = tensor(data, chunk_size=(2, 3))
        P, L, U = lu(a)

        # check lower and upper triangular matrix
        result_l = self.executor.execute_tensor(L, concat=True)[0]
        result_u = self.executor.execute_tensor(U, concat=True)[0]

        np.testing.assert_allclose(np.tril(result_l), result_l)
        np.testing.assert_allclose(np.triu(result_u), result_u)

        t = P.dot(L).dot(U)
        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

        a = tensor(data, chunk_size=4)
        P, L, U = lu(a)

        # check lower and upper triangular matrix
        result_l, result_u = self.executor.execute_tensors([L, U])

        np.testing.assert_allclose(np.tril(result_l), result_l)
        np.testing.assert_allclose(np.triu(result_u), result_u)

        t = P.dot(L).dot(U)
        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data)

        # test for sparse
        data = sps.csr_matrix([[2, 0, 0, 0, 5, 2],
                               [0, 6, 1, 0, 0, 6],
                               [8, 0, 9, 0, 0, 2],
                               [0, 6, 0, 8, 7, 3],
                               [7, 0, 6, 1, 7, 0],
                               [0, 0, 0, 7, 0, 8]])

        a = tensor(data)
        P, L, U = lu(a)
        result_l = self.executor.execute_tensor(L, concat=True)[0]
        result_u = self.executor.execute_tensor(U, concat=True)[0]

        # check lower and upper triangular matrix
        np.testing.assert_allclose(np.tril(result_l), result_l)
        np.testing.assert_allclose(np.triu(result_u), result_u)
        self.assertIsInstance(result_l, SparseNDArray)
        self.assertIsInstance(result_u, SparseNDArray)

        t = P.dot(L).dot(U)
        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_array_almost_equal(data.A, res)

        a = tensor(data, chunk_size=2)
        P, L, U = lu(a)
        result_l = self.executor.execute_tensor(L, concat=True)[0]
        result_u = self.executor.execute_tensor(U, concat=True)[0]

        # check lower and upper triangular matrix
        np.testing.assert_allclose(np.tril(result_l), result_l)
        np.testing.assert_allclose(np.triu(result_u), result_u)
        self.assertIsInstance(result_l, SparseNDArray)
        self.assertIsInstance(result_u, SparseNDArray)

        t = P.dot(L).dot(U)
        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_array_almost_equal(data.A, res)

        a = tensor(data, chunk_size=(2, 3))
        P, L, U = lu(a)
        result_l = self.executor.execute_tensor(L, concat=True)[0]
        result_u = self.executor.execute_tensor(U, concat=True)[0]

        # check lower and upper triangular matrix
        np.testing.assert_allclose(np.tril(result_l), result_l)
        np.testing.assert_allclose(np.triu(result_u), result_u)
        self.assertIsInstance(result_l, SparseNDArray)
        self.assertIsInstance(result_u, SparseNDArray)

        t = P.dot(L).dot(U)
        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_array_almost_equal(data.A, res)

        a = tensor(data, chunk_size=4)
        P, L, U = lu(a)
        result_l = self.executor.execute_tensor(L, concat=True)[0]
        result_u = self.executor.execute_tensor(U, concat=True)[0]

        # check lower and upper triangular matrix
        np.testing.assert_allclose(np.tril(result_l), result_l)
        np.testing.assert_allclose(np.triu(result_u), result_u)
        self.assertIsInstance(result_l, SparseNDArray)
        self.assertIsInstance(result_u, SparseNDArray)

        t = P.dot(L).dot(U)
        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_array_almost_equal(data.A, res)

    def testSolveTriangular(self):
        from mars.tensor import tril, triu
        np.random.seed(1)

        data1 = np.random.randint(1, 10, (20, 20))
        data2 = np.random.randint(1, 10, (20, ))

        A = tensor(data1, chunk_size=20)
        b = tensor(data2, chunk_size=20)

        x = solve_triangular(A, b)
        t = triu(A).dot(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data2)

        x = solve_triangular(A, b, lower=True)
        t = tril(A).dot(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data2)

        A = tensor(data1, chunk_size=10)
        b = tensor(data2, chunk_size=10)

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

        A = tensor(data1, chunk_size=10)
        b = tensor(data2, chunk_size=10)

        x = solve_triangular(A, b)
        t = triu(A).dot(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data2)

        x = solve_triangular(A, b, lower=True)
        t = tril(A).dot(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data2)

        A = tensor(data1, chunk_size=3)
        b = tensor(data2, chunk_size=3)

        x = solve_triangular(A, b)
        t = triu(A).dot(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data2)

        x = solve_triangular(A, b, lower=True)
        t = tril(A).dot(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, data2)

        # test sparse
        data1 = sps.csr_matrix(np.triu(np.random.randint(1, 10, (10, 10))))
        data2 = np.random.random((10,))

        A = tensor(data1, chunk_size=5)
        b = tensor(data2, chunk_size=5)

        x = solve_triangular(A, b)

        result_x = self.executor.execute_tensor(x, concat=True)[0]
        result_b = data1.dot(result_x)

        self.assertIsInstance(result_x, SparseNDArray)
        np.testing.assert_allclose(result_b, data2)

        data1 = sps.csr_matrix(np.triu(np.random.randint(1, 10, (10, 10))))
        data2 = np.random.random((10, 2))

        A = tensor(data1, chunk_size=5)
        b = tensor(data2, chunk_size=5)

        x = solve_triangular(A, b)

        result_x = self.executor.execute_tensor(x, concat=True)[0]
        result_b = data1.dot(result_x)

        self.assertIsInstance(result_x, SparseNDArray)
        np.testing.assert_allclose(result_b, data2)

    def testSolve(self):
        import scipy.linalg
        np.random.seed(1)

        data1 = np.random.randint(1, 10, (20, 20))
        data2 = np.random.randint(1, 10, (20, ))

        A = tensor(data1, chunk_size=5)
        b = tensor(data2, chunk_size=5)

        x = solve(A, b)

        res = self.executor.execute_tensor(x, concat=True)[0]
        np.testing.assert_allclose(res, scipy.linalg.solve(data1, data2))
        res = self.executor.execute_tensor(A.dot(x), concat=True)[0]
        np.testing.assert_allclose(res, data2)

        data2 = np.random.randint(1, 10, (20, 5))

        A = tensor(data1, chunk_size=5)
        b = tensor(data2, chunk_size=5)

        x = solve(A, b)

        res = self.executor.execute_tensor(x, concat=True)[0]
        np.testing.assert_allclose(res, scipy.linalg.solve(data1, data2))
        res = self.executor.execute_tensor(A.dot(x), concat=True)[0]
        np.testing.assert_allclose(res, data2)

        data2 = np.random.randint(1, 10, (20, 20))

        A = tensor(data1, chunk_size=5)
        b = tensor(data2, chunk_size=5)

        x = solve(A, b)

        res = self.executor.execute_tensor(x, concat=True)[0]
        np.testing.assert_allclose(res, scipy.linalg.solve(data1, data2))
        res = self.executor.execute_tensor(A.dot(x), concat=True)[0]
        np.testing.assert_allclose(res, data2)

        # test for not all chunks are square in matrix A
        data2 = np.random.randint(1, 10, (20,))

        A = tensor(data1, chunk_size=6)
        b = tensor(data2, chunk_size=6)

        x = solve(A, b)

        res = self.executor.execute_tensor(x, concat=True)[0]
        np.testing.assert_allclose(res, scipy.linalg.solve(data1, data2))
        res = self.executor.execute_tensor(A.dot(x), concat=True)[0]
        np.testing.assert_allclose(res, data2)

        A = tensor(data1, chunk_size=(7, 6))
        b = tensor(data2, chunk_size=6)

        x = solve(A, b)

        res = self.executor.execute_tensor(x, concat=True)[0]
        np.testing.assert_allclose(res, scipy.linalg.solve(data1, data2))
        res = self.executor.execute_tensor(A.dot(x), concat=True)[0]
        np.testing.assert_allclose(res, data2)

        # test sparse
        data1 = sps.csr_matrix(np.random.randint(1, 10, (20, 20)))
        data2 = np.random.randint(1, 10, (20, ))

        A = tensor(data1, chunk_size=5)
        b = tensor(data2, chunk_size=5)

        x = solve(A, b)

        res = self.executor.execute_tensor(x, concat=True)[0]
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_allclose(data1.dot(res), data2)

        data2 = np.random.randint(1, 10, (20, 5))

        A = tensor(data1, chunk_size=5)
        b = tensor(data2, chunk_size=5)

        x = solve(A, b)

        res = self.executor.execute_tensor(A.dot(x), concat=True)[0]
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_allclose(res, data2)

        data2 = np.random.randint(1, 10, (20, 20))

        A = tensor(data1, chunk_size=5)
        b = tensor(data2, chunk_size=5)

        x = solve(A, b)

        res = self.executor.execute_tensor(A.dot(x), concat=True)[0]
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_allclose(res, data2)

        # test for not all chunks are square in matrix A
        data2 = np.random.randint(1, 10, (20,))

        A = tensor(data1, chunk_size=6)
        b = tensor(data2, chunk_size=6)

        x = solve(A, b)

        res = self.executor.execute_tensor(A.dot(x), concat=True)[0]
        np.testing.assert_allclose(res, data2)

    def testSolveSymPos(self):
        import scipy.linalg
        np.random.seed(1)

        data = np.random.randint(1, 10, (20, 20))
        data_l = np.tril(data)
        data1 = data_l.dot(data_l.T)
        data2 = np.random.randint(1, 10, (20, ))

        A = tensor(data1, chunk_size=5)
        b = tensor(data2, chunk_size=5)

        x = solve(A, b, sym_pos=True)

        res = self.executor.execute_tensor(x, concat=True)[0]
        np.testing.assert_allclose(res, scipy.linalg.solve(data1, data2))
        res = self.executor.execute_tensor(A.dot(x), concat=True)[0]
        np.testing.assert_allclose(res, data2)

    def testInv(self):
        import scipy.linalg
        np.random.seed(1)

        data = np.random.randint(1, 10, (20, 20))

        A = tensor(data)
        inv_A = inv(A)

        res = self.executor.execute_tensor(inv_A, concat=True)[0]
        self.assertTrue(np.allclose(res, scipy.linalg.inv(data)))
        res = self.executor.execute_tensor(A.dot(inv_A), concat=True)[0]
        self.assertTrue(np.allclose(res, np.eye(data.shape[0], dtype=float)))

        A = tensor(data, chunk_size=5)
        inv_A = inv(A)

        res = self.executor.execute_tensor(inv_A, concat=True)[0]
        self.assertTrue(np.allclose(res, scipy.linalg.inv(data)))
        res = self.executor.execute_tensor(A.dot(inv_A), concat=True)[0]
        self.assertTrue(np.allclose(res, np.eye(data.shape[0], dtype=float)))

        B = A.T.dot(A)
        inv_B = inv(B)
        res = self.executor.execute_tensor(inv_B, concat=True)[0]
        self.assertTrue(np.allclose(res, scipy.linalg.inv(data.T.dot(data))))
        res = self.executor.execute_tensor(B.dot(inv_B), concat=True)[0]
        self.assertTrue(np.allclose(res, np.eye(data.shape[0], dtype=float)))

        # test for not all chunks are square in matrix A
        A = tensor(data, chunk_size=8)
        inv_A = inv(A)

        res = self.executor.execute_tensor(inv_A, concat=True)[0]
        self.assertTrue(np.allclose(res, scipy.linalg.inv(data)))
        res = self.executor.execute_tensor(A.dot(inv_A), concat=True)[0]
        self.assertTrue(np.allclose(res, np.eye(data.shape[0], dtype=float)))

        # test sparse
        data = np.random.randint(1, 10, (20, 20))
        sp_data = sps.csr_matrix(data)

        A = tensor(sp_data, chunk_size=5)
        inv_A = inv(A)

        res = self.executor.execute_tensor(inv_A, concat=True)[0]
        self.assertIsInstance(res, SparseNDArray)
        self.assertTrue(np.allclose(res, scipy.linalg.inv(data)))
        res = self.executor.execute_tensor(A.dot(inv_A), concat=True)[0]
        self.assertTrue(np.allclose(res, np.eye(data.shape[0], dtype=float)))

        # test for not all chunks are square in matrix A
        A = tensor(sp_data, chunk_size=8)
        inv_A = inv(A)

        res = self.executor.execute_tensor(inv_A, concat=True)[0]
        self.assertIsInstance(res, SparseNDArray)
        self.assertTrue(np.allclose(res, scipy.linalg.inv(data)))
        res = self.executor.execute_tensor(A.dot(inv_A), concat=True)[0]
        self.assertTrue(np.allclose(res, np.eye(data.shape[0], dtype=float)))

    def testNormExecution(self):
        d = np.arange(9) - 4
        d2 = d.reshape(3, 3)

        ma = [tensor(d, chunk_size=2), tensor(d, chunk_size=9),
              tensor(d2, chunk_size=(2, 3)), tensor(d2, chunk_size=3)]

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

        d = uniform(-0.5, 0.5, size=(500, 2), chunk_size=50)
        inside = (norm(d, axis=1) < 0.5).sum().astype(float)
        t = inside / 500 * 4
        res = self.executor.execute_tensor(t)[0]
        self.assertAlmostEqual(res, 3.14, delta=1)

    def testTensordotExecution(self):
        size_executor = Executor(sync_provider_type=Executor.SyncProviderType.MOCK)

        a_data = np.arange(60).reshape(3, 4, 5)
        a = tensor(a_data, chunk_size=2)
        b_data = np.arange(24).reshape(4, 3, 2)
        b = tensor(b_data, chunk_size=2)

        axes = ([1, 0], [0, 1])
        c = tensordot(a, b, axes=axes)
        size_res = size_executor.execute_tensor(c, mock=True)
        self.assertEqual(sum(s[0] for s in size_res), c.nbytes)
        self.assertEqual(sum(s[1] for s in size_res), c.nbytes)

        res = self.executor.execute_tensor(c)
        expected = np.tensordot(a_data, b_data, axes=axes)
        self.assertTrue(np.array_equal(res[0], expected[:2, :]))
        self.assertTrue(np.array_equal(res[1], expected[2:4, :]))
        self.assertTrue(np.array_equal(res[2], expected[4:, :]))

        a = ones((1000, 2000), chunk_size=500)
        b = ones((2000, 100), chunk_size=500)
        c = dot(a, b)
        res = self.executor.execute_tensor(c)
        expected = np.dot(np.ones((1000, 2000)), np.ones((2000, 100)))
        self.assertEqual(len(res), 2)
        self.assertTrue(np.array_equal(res[0], expected[:500, :]))
        self.assertTrue(np.array_equal(res[1], expected[500:, :]))

        a = ones((10, 8), chunk_size=2)
        b = ones((8, 10), chunk_size=2)
        c = a.dot(b)
        res = self.executor.execute_tensor(c)
        self.assertEqual(len(res), 25)
        for r in res:
            self.assertTrue(np.array_equal(r, np.tile([8], [2, 2])))

        a = ones((500, 500), chunk_size=500)
        b = ones((500, 100), chunk_size=500)
        c = a.dot(b)
        res = self.executor.execute_tensor(c)
        self.assertTrue(np.array_equal(res[0], np.tile([500], [500, 100])))

        raw_a = np.random.random((100, 200, 50))
        raw_b = np.random.random((200, 10, 100))
        a = tensor(raw_a, chunk_size=50)
        b = tensor(raw_b, chunk_size=33)
        c = tensordot(a, b, axes=((0, 1), (2, 0)))
        res = self.executor.execute_tensor(c, concat=True)
        expected = np.tensordot(raw_a, raw_b, axes=(c.op.a_axes, c.op.b_axes))
        self.assertTrue(np.allclose(res[0], expected))

        a = ones((1000, 2000), chunk_size=500)
        b = ones((100, 2000), chunk_size=500)
        c = inner(a, b)
        res = self.executor.execute_tensor(c)
        expected = np.inner(np.ones((1000, 2000)), np.ones((100, 2000)))
        self.assertEqual(len(res), 2)
        self.assertTrue(np.array_equal(res[0], expected[:500, :]))
        self.assertTrue(np.array_equal(res[1], expected[500:, :]))

        a = ones((100, 100), chunk_size=30)
        b = ones((100, 100), chunk_size=30)
        c = a.dot(b)
        res = self.executor.execute_tensor(c, concat=True)[0]
        np.testing.assert_array_equal(res, np.ones((100, 100)) * 100)

    def testSparseDotSizeExecution(self):
        from mars.tensor.linalg.tensordot import TensorTensorDot
        from mars.executor import register, register_default
        chunk_sizes = dict()
        chunk_nbytes = dict()
        chunk_input_sizes = dict()
        chunk_input_nbytes = dict()

        def execute_size(t):
            def _tensordot_size_recorder(ctx, op):
                TensorTensorDot.estimate_size(ctx, op)

                chunk_key = op.outputs[0].key
                chunk_sizes[chunk_key] = ctx[chunk_key]
                chunk_nbytes[chunk_key] = op.outputs[0].nbytes

                input_sizes = dict((inp.op.key, ctx[inp.key][0]) for inp in op.inputs)
                chunk_input_sizes[chunk_key] = sum(input_sizes.values())
                input_nbytes = dict((inp.op.key, inp.nbytes) for inp in op.inputs)
                chunk_input_nbytes[chunk_key] = sum(input_nbytes.values())

            size_executor = Executor(sync_provider_type=Executor.SyncProviderType.MOCK)
            try:
                chunk_sizes.clear()
                chunk_nbytes.clear()
                chunk_input_sizes.clear()
                chunk_input_nbytes.clear()
                register(TensorTensorDot, size_estimator=_tensordot_size_recorder)
                size_executor.execute_tensor(t, mock=True)
            finally:
                register_default(TensorTensorDot)

        a_data = sps.random(5, 9, density=.1)
        b_data = sps.random(9, 10, density=.2)
        a = tensor(a_data, chunk_size=2)
        b = tensor(b_data, chunk_size=3)

        c = dot(a, b)
        execute_size(c)

        for key in chunk_input_sizes.keys():
            self.assertGreaterEqual(chunk_sizes[key][1], chunk_input_sizes[key])

        c2 = dot(a, b, sparse=False)
        execute_size(c2)

        for key in chunk_input_sizes.keys():
            self.assertEqual(chunk_sizes[key][0], chunk_nbytes[key])
            self.assertEqual(chunk_sizes[key][1], chunk_input_nbytes[key] + chunk_nbytes[key])

    def testSparseDotExecution(self):
        a_data = sps.random(5, 9, density=.1)
        b_data = sps.random(9, 10, density=.2)
        a = tensor(a_data, chunk_size=2)
        b = tensor(b_data, chunk_size=3)

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

        # test vector inner
        a_data = np.random.rand(5)
        b_data = np.random.rand(5)
        a = tensor(a_data, chunk_size=2).tosparse()
        b = tensor(b_data, chunk_size=2).tosparse()

        c = inner(a, b)

        res = self.executor.execute_tensor(c, concat=True)[0]
        self.assertTrue(np.isscalar(res))
        np.testing.assert_allclose(res, np.inner(a_data, b_data))

    def testVdotExecution(self):
        a_data = np.array([1 + 2j, 3 + 4j])
        b_data = np.array([5 + 6j, 7 + 8j])
        a = tensor(a_data, chunk_size=1)
        b = tensor(b_data, chunk_size=1)

        t = vdot(a, b)

        res = self.executor.execute_tensor(t)[0]
        expected = np.vdot(a_data, b_data)
        np.testing.assert_equal(res, expected)

        a_data = np.array([[1, 4], [5, 6]])
        b_data = np.array([[4, 1], [2, 2]])
        a = tensor(a_data, chunk_size=1)
        b = tensor(b_data, chunk_size=1)

        t = vdot(a, b)

        res = self.executor.execute_tensor(t)[0]
        expected = np.vdot(a_data, b_data)
        np.testing.assert_equal(res, expected)

    def testMatmulExecution(self):
        data_a = np.random.randn(10, 20)
        data_b = np.random.randn(20)

        a = tensor(data_a, chunk_size=2)
        b = tensor(data_b, chunk_size=3)
        c = matmul(a, b)

        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.matmul(data_a, data_b)
        np.testing.assert_allclose(res, expected)

        data_a = np.random.randn(10, 20)
        data_b = np.random.randn(10)

        a = tensor(data_a, chunk_size=2)
        b = tensor(data_b, chunk_size=3)
        c = matmul(b, a)

        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.matmul(data_b, data_a)
        np.testing.assert_allclose(res, expected)

        data_a = np.random.randn(15, 1, 20, 30)
        data_b = np.random.randn(1, 11, 30, 20)

        a = tensor(data_a, chunk_size=12)
        b = tensor(data_b, chunk_size=13)
        c = matmul(a, b)

        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.matmul(data_a, data_b)
        np.testing.assert_allclose(res, expected, atol=.0001)

        a = arange(2 * 2 * 4, chunk_size=1).reshape((2, 2, 4))
        b = arange(2 * 2 * 4, chunk_size=1).reshape((2, 4, 2))
        c = matmul(a, b)

        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.matmul(np.arange(2 * 2 * 4).reshape(2, 2, 4),
                             np.arange(2 * 2 * 4).reshape(2, 4, 2))
        np.testing.assert_allclose(res, expected, atol=.0001)

        data_a = sps.random(10, 20)
        data_b = sps.random(20, 5)

        a = tensor(data_a, chunk_size=2)
        b = tensor(data_b, chunk_size=3)
        c = matmul(a, b)

        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.matmul(data_a.toarray(), data_b.toarray())
        np.testing.assert_allclose(res.toarray(), expected)

        # test order
        data_a = np.asfortranarray(np.random.randn(10, 20))
        data_b = np.asfortranarray(np.random.randn(20, 30))

        a = tensor(data_a, chunk_size=12)
        b = tensor(data_b, chunk_size=13)

        c = matmul(a, b)
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.matmul(data_a, data_b)

        np.testing.assert_allclose(res, expected)
        self.assertEqual(res.flags['C_CONTIGUOUS'], expected.flags['C_CONTIGUOUS'])
        self.assertEqual(res.flags['F_CONTIGUOUS'], expected.flags['F_CONTIGUOUS'])

        c = matmul(a, b, order='A')
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.matmul(data_a, data_b, order='A')

        np.testing.assert_allclose(res, expected)
        self.assertEqual(res.flags['C_CONTIGUOUS'], expected.flags['C_CONTIGUOUS'])
        self.assertEqual(res.flags['F_CONTIGUOUS'], expected.flags['F_CONTIGUOUS'])

        c = matmul(a, b, order='C')
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.matmul(data_a, data_b, order='C')

        np.testing.assert_allclose(res, expected)
        self.assertEqual(res.flags['C_CONTIGUOUS'], expected.flags['C_CONTIGUOUS'])
        self.assertEqual(res.flags['F_CONTIGUOUS'], expected.flags['F_CONTIGUOUS'])
