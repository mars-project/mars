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
import scipy.sparse as sps

from ....learn.datasets.samples_generator import make_low_rank_matrix
from ....lib.sparse import issparse, SparseNDArray
from ....utils import ignore_warning
from ...datasource import tensor, diag, ones, arange
from ...random import uniform
from .. import (
    qr,
    svd,
    cholesky,
    norm,
    lu,
    solve_triangular,
    solve,
    inv,
    tensordot,
    dot,
    inner,
    vdot,
    matmul,
    randomized_svd,
)


def test_qr_execution(setup):
    rs = np.random.RandomState(0)
    data = rs.randn(18, 6)

    a = tensor(data, chunk_size=(3, 6))
    q, r = qr(a)
    t = q.dot(r)

    res = t.execute().fetch()
    np.testing.assert_array_almost_equal(res, data)

    a = tensor(data, chunk_size=(9, 6))
    q, r = qr(a)
    t = q.dot(r)

    res = t.execute().fetch()
    np.testing.assert_array_almost_equal(res, data)

    a = tensor(data, chunk_size=3)
    q, r = qr(a)
    t = q.dot(r)

    res = t.execute().fetch()
    np.testing.assert_array_almost_equal(res, data)

    # test for Short-and-Fat QR
    data = rs.randn(6, 18)

    a = tensor(data, chunk_size=(6, 9))
    q, r = qr(a, method="sfqr")
    t = q.dot(r)

    res = t.execute().fetch()
    np.testing.assert_array_almost_equal(res, data)

    a = tensor(data, chunk_size=(3, 3))
    q, r = qr(a, method="sfqr")
    t = q.dot(r)

    res = t.execute().fetch()
    np.testing.assert_array_almost_equal(res, data)

    a = tensor(data, chunk_size=(6, 3))
    q, r = qr(a, method="sfqr")
    t = q.dot(r)

    res = t.execute().fetch()
    np.testing.assert_array_almost_equal(res, data)


def test_svd_execution(setup):
    rs = np.random.RandomState()
    data = rs.randn(18, 6) + 1j * rs.randn(18, 6)

    a = tensor(data, chunk_size=(9, 6))
    U, s, V = svd(a)
    t = U.dot(diag(s).dot(V))

    res = t.execute().fetch()
    np.testing.assert_array_almost_equal(res, data)

    a = tensor(data, chunk_size=(18, 6))
    U, s, V = svd(a)
    t = U.dot(diag(s).dot(V))

    res = t.execute().fetch()
    np.testing.assert_array_almost_equal(res, data)

    a = tensor(data, chunk_size=(2, 6))
    U, s, V = svd(a)
    t = U.dot(diag(s).dot(V))

    res = t.execute().fetch()
    np.testing.assert_array_almost_equal(res, data)

    data = rs.randn(6, 18) + 1j * rs.randn(6, 18)

    a = tensor(data)
    U, s, V = svd(a)
    t = U.dot(diag(s).dot(V))

    res = t.execute().fetch()
    np.testing.assert_array_almost_equal(res, data)

    # test for matrix of ones
    data = np.ones((20, 10))

    a = tensor(data, chunk_size=10)
    s = svd(a)[1]
    res = s.execute().fetch()
    expected = np.linalg.svd(a)[1]
    np.testing.assert_array_almost_equal(res, expected)


def test_randomized_svd_execution(setup):
    n_samples = 100
    n_features = 500
    rank = 5
    k = 10
    for dtype in (np.int64, np.float64):
        # generate a matrix X of approximate effective rank `rank` and no noise
        # component (very structured signal):
        X = make_low_rank_matrix(
            n_samples=n_samples,
            n_features=n_features,
            effective_rank=rank,
            tail_strength=0.0,
            random_state=0,
        ).astype(dtype, copy=False)
        assert X.shape == (n_samples, n_features)
        dtype = np.dtype(dtype)
        decimal = 5 if dtype == np.float32 else 7

        # compute the singular values of X using the slow exact method
        X_res = X.execute().fetch()
        U, s, V = np.linalg.svd(X_res, full_matrices=False)

        # Convert the singular values to the specific dtype
        U = U.astype(dtype, copy=False)
        s = s.astype(dtype, copy=False)
        V = V.astype(dtype, copy=False)

        for normalizer in ["auto", "LU", "QR"]:  # 'none' would not be stable
            # compute the singular values of X using the fast approximate method
            Ua, sa, Va = randomized_svd(
                X, k, n_iter=1, power_iteration_normalizer=normalizer, random_state=0
            )

            # If the input dtype is float, then the output dtype is float of the
            # same bit size (f32 is not upcast to f64)
            # But if the input dtype is int, the output dtype is float64
            if dtype.kind == "f":
                assert Ua.dtype == dtype
                assert sa.dtype == dtype
                assert Va.dtype == dtype
            else:
                assert Ua.dtype == np.float64
                assert sa.dtype == np.float64
                assert Va.dtype == np.float64

            assert Ua.shape == (n_samples, k)
            assert sa.shape == (k,)
            assert Va.shape == (k, n_features)

            # ensure that the singular values of both methods are equal up to the
            # real rank of the matrix
            sa_res = sa.execute().fetch()
            np.testing.assert_almost_equal(s[:k], sa_res, decimal=decimal)

            # check the singular vectors too (while not checking the sign)
            dot_res = dot(Ua, Va).execute().fetch()
            np.testing.assert_almost_equal(
                np.dot(U[:, :k], V[:k, :]), dot_res, decimal=decimal
            )


def test_cholesky_execution(setup):
    rs = np.random.RandomState(0)
    data = rs.randint(1, 10, (10, 10))
    symmetric_data = data.dot(data.T)

    a = tensor(symmetric_data, chunk_size=5)

    U = cholesky(a)
    t = U.T.dot(U)

    res_u = U.execute().fetch()
    np.testing.assert_allclose(np.triu(res_u), res_u)

    res = t.execute().fetch()
    np.testing.assert_allclose(res, symmetric_data)

    L = cholesky(a, lower=True)
    U = cholesky(a)
    t = L.dot(U)

    res = t.execute().fetch()
    np.testing.assert_allclose(res, symmetric_data)

    a = tensor(symmetric_data, chunk_size=5)

    L = cholesky(a, lower=True)
    U = cholesky(a)
    t = L.dot(U)

    res_u = U.execute().fetch()
    np.testing.assert_allclose(np.triu(res_u), res_u)

    res = t.execute().fetch()
    np.testing.assert_allclose(res, symmetric_data)

    a = tensor(symmetric_data, chunk_size=(2, 3))

    L = cholesky(a, lower=True)
    U = cholesky(a)
    t = L.dot(U)

    res = t.execute().fetch()
    np.testing.assert_allclose(res, symmetric_data)


def test_lu_execution(setup):
    rs = np.random.RandomState(0)

    # square matrix
    data = rs.randint(1, 10, (6, 6))

    a = tensor(data)
    P, L, U = lu(a)

    # check lower and upper triangular matrix
    result_l = L.execute().fetch()
    result_u = U.execute().fetch()

    np.testing.assert_allclose(np.tril(result_l), result_l)
    np.testing.assert_allclose(np.triu(result_u), result_u)

    t = P.dot(L).dot(U)
    res = t.execute().fetch()
    np.testing.assert_allclose(res, data)

    a = tensor(data, chunk_size=(3, 4))
    P, L, U = lu(a)

    # check lower and upper triangular matrix
    result_l = L.execute().fetch()
    result_u = U.execute().fetch()

    np.testing.assert_allclose(np.tril(result_l), result_l)
    np.testing.assert_allclose(np.triu(result_u), result_u)

    t = P.dot(L).dot(U)
    res = t.execute().fetch()
    np.testing.assert_allclose(res, data)

    # shape[0] > shape[1]
    data = rs.randint(1, 10, (10, 6))

    a = tensor(data)
    P, L, U = lu(a)

    # check lower and upper triangular matrix
    result_l = L.execute().fetch()
    result_u = U.execute().fetch()

    np.testing.assert_allclose(np.tril(result_l), result_l)
    np.testing.assert_allclose(np.triu(result_u), result_u)

    t = P.dot(L).dot(U)
    res = t.execute().fetch()
    np.testing.assert_allclose(res, data)

    a = tensor(data, chunk_size=5)
    P, L, U = lu(a)

    # check lower and upper triangular matrix
    result_l = L.execute().fetch()
    result_u = U.execute().fetch()

    np.testing.assert_allclose(np.tril(result_l), result_l)
    np.testing.assert_allclose(np.triu(result_u), result_u)

    t = P.dot(L).dot(U)
    res = t.execute().fetch()
    np.testing.assert_allclose(res, data)

    a = tensor(data, chunk_size=(4, 5))
    P, L, U = lu(a)

    # check lower and upper triangular matrix
    result_l = L.execute().fetch()
    result_u = U.execute().fetch()

    np.testing.assert_allclose(np.tril(result_l), result_l)
    np.testing.assert_allclose(np.triu(result_u), result_u)

    t = P.dot(L).dot(U)
    res = t.execute().fetch()
    np.testing.assert_allclose(res, data)

    # shape[0] < shape[1]
    data = rs.randint(1, 10, (6, 10))

    a = tensor(data)
    P, L, U = lu(a)

    # check lower and upper triangular matrix
    result_l = L.execute().fetch()
    result_u = U.execute().fetch()

    np.testing.assert_allclose(np.tril(result_l), result_l)
    np.testing.assert_allclose(np.triu(result_u), result_u)

    t = P.dot(L).dot(U)
    res = t.execute().fetch()
    np.testing.assert_allclose(res, data)

    a = tensor(data, chunk_size=5)
    P, L, U = lu(a)

    # check lower and upper triangular matrix
    result_l = L.execute().fetch()
    result_u = U.execute().fetch()

    np.testing.assert_allclose(np.tril(result_l), result_l)
    np.testing.assert_allclose(np.triu(result_u), result_u)

    t = P.dot(L).dot(U)
    res = t.execute().fetch()
    np.testing.assert_allclose(res, data)

    a = tensor(data, chunk_size=(4, 5))
    P, L, U = lu(a)

    # check lower and upper triangular matrix
    result_l = L.execute().fetch()
    result_u = U.execute().fetch()

    np.testing.assert_allclose(np.tril(result_l), result_l)
    np.testing.assert_allclose(np.triu(result_u), result_u)

    t = P.dot(L).dot(U)
    res = t.execute().fetch()
    np.testing.assert_allclose(res, data)

    # test for sparse
    data = sps.csr_matrix(
        [
            [2, 0, 0, 0, 5, 2],
            [0, 6, 1, 0, 0, 6],
            [8, 0, 9, 0, 0, 2],
            [0, 6, 0, 8, 7, 3],
            [7, 0, 6, 1, 7, 0],
            [0, 0, 0, 7, 0, 8],
        ]
    )

    a = tensor(data)
    P, L, U = lu(a)
    result_l = L.execute().fetch()
    result_u = U.execute().fetch()

    # check lower and upper triangular matrix
    np.testing.assert_allclose(np.tril(result_l), result_l)
    np.testing.assert_allclose(np.triu(result_u), result_u)
    assert isinstance(result_l, SparseNDArray)
    assert isinstance(result_u, SparseNDArray)

    t = P.dot(L).dot(U)
    res = t.execute().fetch()
    np.testing.assert_array_almost_equal(data.A, res)

    a = tensor(data, chunk_size=5)
    P, L, U = lu(a)
    result_l = L.execute().fetch()
    result_u = U.execute().fetch()

    # check lower and upper triangular matrix
    np.testing.assert_allclose(np.tril(result_l), result_l)
    np.testing.assert_allclose(np.triu(result_u), result_u)
    assert isinstance(result_l, SparseNDArray)
    assert isinstance(result_u, SparseNDArray)

    t = P.dot(L).dot(U)
    res = t.execute().fetch()
    np.testing.assert_array_almost_equal(data.A, res)


def test_solve_triangular(setup):
    from ... import tril, triu

    rs = np.random.RandomState(0)

    data1 = rs.randint(1, 10, (20, 20))
    data2 = rs.randint(1, 10, (20,))

    A = tensor(data1, chunk_size=20)
    b = tensor(data2, chunk_size=20)

    x = solve_triangular(A, b)
    t = triu(A).dot(x)

    res = t.execute().fetch()
    np.testing.assert_allclose(res, data2)

    x = solve_triangular(A, b, lower=True)
    t = tril(A).dot(x)

    res = t.execute().fetch()
    np.testing.assert_allclose(res, data2)

    A = tensor(data1, chunk_size=10)
    b = tensor(data2, chunk_size=10)

    x = solve_triangular(A, b)
    t = triu(A).dot(x)

    res = t.execute().fetch()
    np.testing.assert_allclose(res, data2)

    x = solve_triangular(A, b, lower=True)
    t = tril(A).dot(x)

    res = t.execute().fetch()
    np.testing.assert_allclose(res, data2)

    data1 = rs.randint(1, 10, (10, 10))
    data2 = rs.randint(1, 10, (10, 5))

    A = tensor(data1, chunk_size=10)
    b = tensor(data2, chunk_size=10)

    x = solve_triangular(A, b)
    t = triu(A).dot(x)

    res = t.execute().fetch()
    np.testing.assert_allclose(res, data2)

    x = solve_triangular(A, b, lower=True)
    t = tril(A).dot(x)

    res = t.execute().fetch()
    np.testing.assert_allclose(res, data2)

    # test sparse
    data1 = sps.csr_matrix(np.triu(rs.randint(1, 10, (10, 10))))
    data2 = rs.random((10,))

    A = tensor(data1, chunk_size=5)
    b = tensor(data2, chunk_size=5)

    x = solve_triangular(A, b)

    result_x = x.execute().fetch()
    result_b = data1.dot(result_x)

    assert isinstance(result_x, SparseNDArray)
    np.testing.assert_allclose(result_b, data2)

    data1 = sps.csr_matrix(np.triu(rs.randint(1, 10, (10, 10))))
    data2 = rs.random((10, 2))

    A = tensor(data1, chunk_size=5)
    b = tensor(data2, chunk_size=5)

    x = solve_triangular(A, b)

    result_x = x.execute().fetch()
    result_b = data1.dot(result_x)

    assert isinstance(result_x, SparseNDArray)
    np.testing.assert_allclose(result_b, data2)


def test_solve(setup):
    import scipy.linalg

    rs = np.random.RandomState(0)

    data1 = rs.randint(1, 10, (20, 20))
    data2 = rs.randint(1, 10, (20,))

    A = tensor(data1, chunk_size=10)
    b = tensor(data2, chunk_size=10)

    x = solve(A, b)

    res = x.execute().fetch()
    np.testing.assert_allclose(res, scipy.linalg.solve(data1, data2))
    res = A.dot(x).execute().fetch()
    np.testing.assert_allclose(res, data2)

    data2 = rs.randint(1, 10, (20, 5))

    A = tensor(data1, chunk_size=10)
    b = tensor(data2, chunk_size=10)

    x = solve(A, b)

    res = x.execute().fetch()
    np.testing.assert_allclose(res, scipy.linalg.solve(data1, data2))
    res = A.dot(x).execute().fetch()
    np.testing.assert_allclose(res, data2)

    # test for not all chunks are square in matrix A
    data2 = rs.randint(1, 10, (20,))

    A = tensor(data1, chunk_size=10)
    b = tensor(data2, chunk_size=10)

    x = solve(A, b)

    res = x.execute().fetch()
    np.testing.assert_allclose(res, scipy.linalg.solve(data1, data2))
    res = A.dot(x).execute().fetch()
    np.testing.assert_allclose(res, data2)

    A = tensor(data1, chunk_size=(10, 15))
    b = tensor(data2, chunk_size=10)

    x = solve(A, b)

    res = x.execute().fetch()
    np.testing.assert_allclose(res, scipy.linalg.solve(data1, data2))
    res = A.dot(x).execute().fetch()
    np.testing.assert_allclose(res, data2)

    # test sparse
    data1 = sps.csr_matrix(rs.randint(1, 10, (20, 20)))
    data2 = rs.randint(1, 10, (20,))

    A = tensor(data1, chunk_size=10)
    b = tensor(data2, chunk_size=10)

    x = solve(A, b)

    res = x.execute().fetch()
    assert isinstance(res, SparseNDArray)
    np.testing.assert_allclose(data1.dot(res), data2)

    data2 = rs.randint(1, 10, (20, 5))

    A = tensor(data1, chunk_size=10)
    b = tensor(data2, chunk_size=10)

    x = solve(A, b)

    res = A.dot(x).execute().fetch()
    assert isinstance(res, SparseNDArray)
    np.testing.assert_allclose(res, data2)

    # test for not all chunks are square in matrix A
    data2 = rs.randint(1, 10, (20,))

    A = tensor(data1, chunk_size=10)
    b = tensor(data2, chunk_size=10)

    x = solve(A, b)

    res = A.dot(x).execute().fetch()
    np.testing.assert_allclose(res, data2)


def test_solve_sym_pos(setup):
    import scipy.linalg

    rs = np.random.RandomState(0)

    data = rs.randint(1, 10, (20, 20))
    data_l = np.tril(data)
    data1 = data_l.dot(data_l.T)
    data2 = rs.randint(1, 10, (20,))

    A = tensor(data1, chunk_size=10)
    b = tensor(data2, chunk_size=10)

    x = solve(A, b, sym_pos=True)

    res = x.execute().fetch()
    np.testing.assert_allclose(res, scipy.linalg.solve(data1, data2))
    res = A.dot(x).execute().fetch()
    np.testing.assert_allclose(res, data2)


def test_inv(setup):
    import scipy.linalg

    rs = np.random.RandomState(0)

    data = rs.randint(1, 10, (20, 20))

    A = tensor(data)
    inv_A = inv(A)

    res = inv_A.execute().fetch()
    np.testing.assert_allclose(res, scipy.linalg.inv(data))
    res = A.dot(inv_A).execute().fetch()
    np.testing.assert_array_almost_equal(res, np.eye(data.shape[0], dtype=float))

    A = tensor(data, chunk_size=10)
    inv_A = inv(A)

    res = inv_A.execute().fetch()
    np.testing.assert_allclose(res, scipy.linalg.inv(data))
    res = A.dot(inv_A).execute().fetch()
    np.testing.assert_array_almost_equal(res, np.eye(data.shape[0], dtype=float))

    # test 1 chunk
    A = tensor(data, chunk_size=20)
    inv_A = inv(A)

    res = inv_A.execute().fetch()
    np.testing.assert_allclose(res, scipy.linalg.inv(data))
    res = A.dot(inv_A).execute().fetch()
    np.testing.assert_array_almost_equal(res, np.eye(data.shape[0], dtype=float))

    B = A.T.dot(A)
    inv_B = inv(B)
    res = inv_B.execute().fetch()
    np.testing.assert_array_almost_equal(res, scipy.linalg.inv(data.T.dot(data)))
    res = B.dot(inv_B).execute().fetch()
    np.testing.assert_array_almost_equal(res, np.eye(data.shape[0], dtype=float))

    # test for not all chunks are square in matrix A
    A = tensor(data, chunk_size=8)
    inv_A = inv(A)

    res = inv_A.execute().fetch()
    np.testing.assert_array_almost_equal(res, scipy.linalg.inv(data))
    res = A.dot(inv_A).execute().fetch()
    np.testing.assert_array_almost_equal(res, np.eye(data.shape[0], dtype=float))

    # test sparse
    data = rs.randint(1, 10, (20, 20))
    sp_data = sps.csr_matrix(data)

    A = tensor(sp_data, chunk_size=10)
    inv_A = inv(A)

    res = inv_A.execute().fetch()
    assert isinstance(res, SparseNDArray)
    np.testing.assert_array_almost_equal(res, scipy.linalg.inv(data))
    res = A.dot(inv_A).execute().fetch()
    np.testing.assert_array_almost_equal(res, np.eye(data.shape[0], dtype=float))

    # test for not all chunks are square in matrix A
    A = tensor(sp_data, chunk_size=12)
    inv_A = inv(A)

    res = inv_A.execute().fetch()
    assert isinstance(res, SparseNDArray)
    np.testing.assert_array_almost_equal(res, scipy.linalg.inv(data))
    res = A.dot(inv_A).execute().fetch()
    np.testing.assert_array_almost_equal(res, np.eye(data.shape[0], dtype=float))


@ignore_warning
def test_norm_execution(setup):
    d = np.arange(9) - 4
    d2 = d.reshape(3, 3)

    ma = [tensor(d, chunk_size=2), tensor(d2, chunk_size=(2, 3))]

    for i, a in enumerate(ma):
        data = d if i < 1 else d2
        for ord in (None, "nuc", np.inf, -np.inf, 0, 1, -1, 2, -2):
            for axis in (0, 1, (0, 1), -1):
                for keepdims in (True, False):
                    try:
                        expected = np.linalg.norm(
                            data, ord=ord, axis=axis, keepdims=keepdims
                        )
                        t = norm(a, ord=ord, axis=axis, keepdims=keepdims)
                        res = t.execute().fetch()

                        expected_shape = expected.shape
                        t_shape = t.shape
                        assert expected_shape == t_shape

                        np.testing.assert_allclose(res, expected, atol=0.0001)
                    except ValueError:
                        continue

    m = norm(tensor(d))
    expected = m.execute().fetch()
    res = np.linalg.norm(d)
    assert expected == res

    d = uniform(-0.5, 0.5, size=(5000, 2), chunk_size=1000)
    inside = (norm(d, axis=1) < 0.5).sum().astype(float)
    t = inside / 5000 * 4
    res = t.execute().fetch()
    np.testing.assert_almost_equal(3.14, res, decimal=1)

    raw = np.random.RandomState(0).rand(10, 10)
    d = norm(tensor(raw, chunk_size=5))
    expected = d.execute().fetch()
    result = np.linalg.norm(raw)
    np.testing.assert_allclose(expected, result)


def test_tensordot_execution(setup):
    rs = np.random.RandomState(0)
    # size_executor = ExecutorForTest(sync_provider_type=ExecutorForTest.SyncProviderType.MOCK)
    #
    # a_data = np.arange(60).reshape(3, 4, 5)
    # a = tensor(a_data, chunk_size=2)
    # b_data = np.arange(24).reshape(4, 3, 2)
    # b = tensor(b_data, chunk_size=2)
    #
    # axes = ([1, 0], [0, 1])
    # c = tensordot(a, b, axes=axes)
    # size_res = size_executor.execute_tensor(c, mock=True)
    # assert sum(s[0] for s in size_res) == c.nbytes
    # assert sum(s[1] for s in size_res) == c.nbytes

    a = ones((100, 200), chunk_size=50)
    b = ones((200, 10), chunk_size=50)
    c = dot(a, b)
    res = c.execute().fetch()
    expected = np.dot(np.ones((100, 200)), np.ones((200, 10)))
    np.testing.assert_array_equal(res, expected)

    a = ones((10, 8), chunk_size=4)
    b = ones((8, 10), chunk_size=4)
    c = a.dot(b)
    res = c.execute().fetch()
    np.testing.assert_array_equal(res, np.tile([8], [10, 10]))

    a = ones((500, 500), chunk_size=500)
    b = ones((500, 100), chunk_size=500)
    c = a.dot(b)
    res = c.execute().fetch()
    np.testing.assert_array_equal(res, np.tile([500], [500, 100]))

    raw_a = rs.random((100, 200, 50))
    raw_b = rs.random((200, 10, 100))
    a = tensor(raw_a, chunk_size=50)
    b = tensor(raw_b, chunk_size=33)
    c = tensordot(a, b, axes=((0, 1), (2, 0)))
    res = c.execute().fetch()
    expected = np.tensordot(raw_a, raw_b, axes=(c.op.a_axes, c.op.b_axes))
    np.testing.assert_array_almost_equal(res, expected)

    a = ones((100, 200), chunk_size=50)
    b = ones((10, 200), chunk_size=50)
    c = inner(a, b)
    res = c.execute().fetch()
    expected = np.inner(np.ones((100, 200)), np.ones((10, 200)))
    np.testing.assert_array_equal(res, expected)

    a = ones((100, 100), chunk_size=30)
    b = ones((100, 100), chunk_size=30)
    c = a.dot(b)
    res = c.execute().fetch()
    np.testing.assert_array_equal(res, np.ones((100, 100)) * 100)


# def test_sparse_dot_size_execution():
#     from mars.tensor.linalg.tensordot import TensorTensorDot
#     from mars.executor import register, register_default
#     chunk_sizes = dict()
#     chunk_nbytes = dict()
#     chunk_input_sizes = dict()
#     chunk_input_nbytes = dict()
#
#     def execute_size(t):
#         def _tensordot_size_recorder(ctx, op):
#             TensorTensorDot.estimate_size(ctx, op)
#
#             chunk_key = op.outputs[0].key
#             chunk_sizes[chunk_key] = ctx[chunk_key]
#             chunk_nbytes[chunk_key] = op.outputs[0].nbytes
#
#             input_sizes = dict((inp.op.key, ctx[inp.key][0]) for inp in op.inputs)
#             chunk_input_sizes[chunk_key] = sum(input_sizes.values())
#             input_nbytes = dict((inp.op.key, inp.nbytes) for inp in op.inputs)
#             chunk_input_nbytes[chunk_key] = sum(input_nbytes.values())
#
#         size_executor = ExecutorForTest(sync_provider_type=ExecutorForTest.SyncProviderType.MOCK)
#         try:
#             chunk_sizes.clear()
#             chunk_nbytes.clear()
#             chunk_input_sizes.clear()
#             chunk_input_nbytes.clear()
#             register(TensorTensorDot, size_estimator=_tensordot_size_recorder)
#             size_executor.execute_tensor(t, mock=True)
#         finally:
#             register_default(TensorTensorDot)
#
#     a_data = sps.random(5, 9, density=.1)
#     b_data = sps.random(9, 10, density=.2)
#     a = tensor(a_data, chunk_size=2)
#     b = tensor(b_data, chunk_size=3)
#
#     c = dot(a, b)
#     execute_size(c)
#
#     for key in chunk_input_sizes.keys():
#         assert chunk_sizes[key][1] >= chunk_input_sizes[key]
#
#     c2 = dot(a, b, sparse=False)
#     execute_size(c2)
#
#     for key in chunk_input_sizes.keys():
#         assert chunk_sizes[key][0] == chunk_nbytes[key]
#         assert chunk_sizes[key][1] == chunk_input_nbytes[key] + chunk_nbytes[key]


def test_sparse_dot_execution(setup):
    rs = np.random.RandomState(0)

    a_data = sps.random(5, 9, density=0.1)
    b_data = sps.random(9, 10, density=0.2)
    a = tensor(a_data, chunk_size=2)
    b = tensor(b_data, chunk_size=3)

    c = dot(a, b)

    res = c.execute().fetch()
    assert issparse(res) is True
    np.testing.assert_allclose(res.toarray(), a_data.dot(b_data).toarray())

    c2 = dot(a, b, sparse=False)

    res = c2.execute().fetch()
    assert issparse(res) is False
    np.testing.assert_allclose(res, a_data.dot(b_data).toarray())

    c3 = tensordot(a, b.T, (-1, -1), sparse=False)

    res = c3.execute().fetch()
    assert issparse(res) is False
    np.testing.assert_allclose(res, a_data.dot(b_data).toarray())

    c = inner(a, b.T)

    res = c.execute().fetch()
    assert issparse(res) is True
    np.testing.assert_allclose(res.toarray(), a_data.dot(b_data).toarray())

    c = inner(a, b.T, sparse=False)

    res = c.execute().fetch()
    assert issparse(res) is False
    np.testing.assert_allclose(res, a_data.dot(b_data).toarray())

    # test vector inner
    a_data = rs.rand(5)
    b_data = rs.rand(5)
    a = tensor(a_data, chunk_size=2).tosparse()
    b = tensor(b_data, chunk_size=2).tosparse()

    c = inner(a, b)

    res = c.execute().fetch()
    assert np.isscalar(res) is True
    np.testing.assert_allclose(res, np.inner(a_data, b_data))


def test_vdot_execution(setup):
    a_data = np.array([1 + 2j, 3 + 4j])
    b_data = np.array([5 + 6j, 7 + 8j])
    a = tensor(a_data, chunk_size=1)
    b = tensor(b_data, chunk_size=1)

    t = vdot(a, b)

    res = t.execute().fetch()
    expected = np.vdot(a_data, b_data)
    np.testing.assert_equal(res, expected)

    a_data = np.array([[1, 4], [5, 6]])
    b_data = np.array([[4, 1], [2, 2]])
    a = tensor(a_data, chunk_size=1)
    b = tensor(b_data, chunk_size=1)

    t = vdot(a, b)

    res = t.execute().fetch()
    expected = np.vdot(a_data, b_data)
    np.testing.assert_equal(res, expected)


def test_matmul_execution(setup):
    rs = np.random.RandomState(0)

    data_a = rs.randn(10, 20)
    data_b = rs.randn(20)

    a = tensor(data_a, chunk_size=5)
    b = tensor(data_b, chunk_size=6)
    c = matmul(a, b)

    res = c.execute().fetch()
    expected = np.matmul(data_a, data_b)
    np.testing.assert_allclose(res, expected)

    data_a = rs.randn(10, 20)
    data_b = rs.randn(10)

    a = tensor(data_a, chunk_size=5)
    b = tensor(data_b, chunk_size=6)
    c = matmul(b, a)

    res = c.execute().fetch()
    expected = np.matmul(data_b, data_a)
    np.testing.assert_allclose(res, expected)

    data_a = rs.randn(15, 1, 20, 30)
    data_b = rs.randn(1, 11, 30, 20)

    a = tensor(data_a, chunk_size=12)
    b = tensor(data_b, chunk_size=13)
    c = matmul(a, b)

    res = c.execute().fetch()
    expected = np.matmul(data_a, data_b)
    np.testing.assert_allclose(res, expected, atol=0.0001)

    a = arange(2 * 2 * 4, chunk_size=1).reshape((2, 2, 4))
    b = arange(2 * 2 * 4, chunk_size=1).reshape((2, 4, 2))
    c = matmul(a, b)

    res = c.execute().fetch()
    expected = np.matmul(
        np.arange(2 * 2 * 4).reshape(2, 2, 4), np.arange(2 * 2 * 4).reshape(2, 4, 2)
    )
    np.testing.assert_allclose(res, expected, atol=0.0001)

    data_a = sps.random(10, 20)
    data_b = sps.random(20, 5)

    a = tensor(data_a, chunk_size=5)
    b = tensor(data_b, chunk_size=6)
    c = matmul(a, b)

    res = c.execute().fetch()
    expected = np.matmul(data_a.toarray(), data_b.toarray())
    np.testing.assert_allclose(res.toarray(), expected)

    # test order
    data_a = np.asfortranarray(rs.randn(10, 20))
    data_b = np.asfortranarray(rs.randn(20, 30))

    a = tensor(data_a, chunk_size=12)
    b = tensor(data_b, chunk_size=13)

    c = matmul(a, b)
    res = c.execute().fetch()
    expected = np.matmul(data_a, data_b)

    np.testing.assert_allclose(res, expected)
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]

    c = matmul(a, b, order="A")
    res = c.execute().fetch()
    expected = np.matmul(data_a, data_b, order="A")

    np.testing.assert_allclose(res, expected)
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]

    c = matmul(a, b, order="C")
    res = c.execute().fetch()
    expected = np.matmul(data_a, data_b, order="C")

    np.testing.assert_allclose(res, expected)
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]
