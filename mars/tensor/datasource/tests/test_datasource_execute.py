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

import tempfile
import shutil
import unittest
import os
import time

import numpy as np
import pandas as pd
try:
    import scipy.sparse as sps
except ImportError:
    sps = None
try:
    import tiledb
except (ImportError, OSError):  # pragma: no cover
    tiledb = None
try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None
try:
    import zarr
except ImportError:  # pragma: no cover
    zarr = None

import mars.tensor as mt
import mars.dataframe as md
from mars.core import get_tiled
from mars.lib.sparse import SparseNDArray
from mars.tests.core import TestBase, ExecutorForTest
from mars.tensor.datasource import tensor, ones_like, zeros, zeros_like, full, full_like, \
    arange, empty, empty_like, diag, diagflat, eye, linspace, meshgrid, indices, \
    triu, tril, from_dataframe, fromtiledb, fromhdf5, fromzarr
from mars.tensor.lib import nd_grid


class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.executor = ExecutorForTest('numpy')

    def testCreateSparseExecution(self):
        mat = sps.csr_matrix([[0, 0, 2], [2, 0, 0]])
        t = tensor(mat, dtype='f8', chunk_size=2)

        res = self.executor.execute_tensor(t)
        self.assertIsInstance(res[0], SparseNDArray)
        self.assertEqual(res[0].dtype, np.float64)
        np.testing.assert_array_equal(res[0].toarray(), mat[..., :2].toarray())
        np.testing.assert_array_equal(res[1].toarray(), mat[..., 2:].toarray())

        t2 = ones_like(t, dtype='f4')

        res = self.executor.execute_tensor(t2)
        expected = sps.csr_matrix([[0, 0, 1], [1, 0, 0]])
        self.assertIsInstance(res[0], SparseNDArray)
        self.assertEqual(res[0].dtype, np.float32)
        np.testing.assert_array_equal(res[0].toarray(), expected[..., :2].toarray())
        np.testing.assert_array_equal(res[1].toarray(), expected[..., 2:].toarray())

        t3 = tensor(np.array([[0, 0, 2], [2, 0, 0]]), chunk_size=2).tosparse()

        res = self.executor.execute_tensor(t3)
        self.assertIsInstance(res[0], SparseNDArray)
        self.assertEqual(res[0].dtype, np.int_)
        np.testing.assert_array_equal(res[0].toarray(), mat[..., :2].toarray())
        np.testing.assert_array_equal(res[1].toarray(), mat[..., 2:].toarray())

        # test missing argument
        t4 = tensor(np.array([[0, 0, 2], [2, 0, 0]]), chunk_size=2).tosparse(missing=2)
        t4 = t4 + 1
        expected = mat.toarray()
        raw = expected.copy()
        expected[raw == 0] += 1
        expected[raw != 0] = 0

        res = self.executor.execute_tensor(t4, concat=True)[0]
        self.assertIsInstance(res, SparseNDArray)
        self.assertEqual(res.dtype, np.int_)
        np.testing.assert_array_equal(res.toarray(), expected)

        # test missing argument that is np.nan
        t5 = tensor(np.array([[np.nan, np.nan, 2], [2, np.nan, -999]]),
                    chunk_size=2).tosparse(missing=[np.nan, -999])
        t5 = (t5 + 1).todense(fill_value=np.nan)
        expected = mat.toarray().astype(float)
        expected[expected != 0] += 1
        expected[expected == 0] = np.nan

        res = self.executor.execute_tensor(t5, concat=True)[0]
        self.assertEqual(res.dtype, np.float64)
        np.testing.assert_array_equal(res, expected)

    def testZerosExecution(self):
        t = zeros((20, 30), dtype='i8', chunk_size=5)

        res = self.executor.execute_tensor(t, concat=True)
        np.testing.assert_array_equal(res[0], np.zeros((20, 30), dtype='i8'))
        self.assertEqual(res[0].dtype, np.int64)

        t2 = zeros_like(t)
        res = self.executor.execute_tensor(t2, concat=True)
        np.testing.assert_array_equal(res[0], np.zeros((20, 30), dtype='i8'))
        self.assertEqual(res[0].dtype, np.int64)

        t = zeros((20, 30), dtype='i4', chunk_size=5, sparse=True)
        res = self.executor.execute_tensor(t, concat=True)

        self.assertEqual(res[0].nnz, 0)

        t = zeros((20, 30), dtype='i8', chunk_size=6, order='F')
        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.zeros((20, 30), dtype='i8', order='F')
        self.assertEqual(res.flags['C_CONTIGUOUS'], expected.flags['C_CONTIGUOUS'])
        self.assertEqual(res.flags['F_CONTIGUOUS'], expected.flags['F_CONTIGUOUS'])

    def testEmptyExecution(self):
        t = empty((20, 30), dtype='i8', chunk_size=5)

        res = self.executor.execute_tensor(t, concat=True)
        self.assertEqual(res[0].shape, (20, 30))
        self.assertEqual(res[0].dtype, np.int64)

        t = empty((20, 30), chunk_size=5)

        res = self.executor.execute_tensor(t, concat=True)
        self.assertEqual(res[0].shape, (20, 30))
        self.assertEqual(res[0].dtype, np.float64)

        t2 = empty_like(t)
        res = self.executor.execute_tensor(t2, concat=True)
        self.assertEqual(res[0].shape, (20, 30))
        self.assertEqual(res[0].dtype, np.float64)

        t = empty((20, 30), dtype='i8', chunk_size=5, order='F')

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.empty((20, 30), dtype='i8', order='F')
        self.assertEqual(res.flags['C_CONTIGUOUS'], expected.flags['C_CONTIGUOUS'])
        self.assertEqual(res.flags['F_CONTIGUOUS'], expected.flags['F_CONTIGUOUS'])

    def testFullExecution(self):
        t = full((2, 2), 1, dtype='f4', chunk_size=1)

        res = self.executor.execute_tensor(t, concat=True)
        np.testing.assert_array_equal(res[0], np.full((2, 2), 1, dtype='f4'))

        t = full((2, 2), [1, 2], dtype='f8', chunk_size=1)

        res = self.executor.execute_tensor(t, concat=True)
        np.testing.assert_array_equal(res[0], np.full((2, 2), [1, 2], dtype='f8'))

        t = full((2, 2), 1, dtype='f4', chunk_size=1, order='F')

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.full((2, 2), 1, dtype='f4', order='F')
        self.assertEqual(res.flags['C_CONTIGUOUS'], expected.flags['C_CONTIGUOUS'])
        self.assertEqual(res.flags['F_CONTIGUOUS'], expected.flags['F_CONTIGUOUS'])

        t2 = full_like(t, 10, order='F')

        res = self.executor.execute_tensor(t2, concat=True)[0]
        expected = np.full((2, 2), 10, dtype='f4', order='F')
        np.testing.assert_array_equal(res, expected)
        self.assertEqual(res.flags['C_CONTIGUOUS'], expected.flags['C_CONTIGUOUS'])
        self.assertEqual(res.flags['F_CONTIGUOUS'], expected.flags['F_CONTIGUOUS'])

    def testArangeExecution(self):
        t = arange(1, 20, 3, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.array_equal(res, np.arange(1, 20, 3)))

        t = arange(1, 20, .3, chunk_size=4)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.arange(1, 20, .3)
        self.assertTrue(np.allclose(res, expected))

        t = arange(1.0, 1.8, .3, chunk_size=4)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.arange(1.0, 1.8, .3)
        self.assertTrue(np.allclose(res, expected))

        t = arange('1066-10-13', '1066-10-31', dtype=np.datetime64, chunk_size=3)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.arange('1066-10-13', '1066-10-31', dtype=np.datetime64)
        self.assertTrue(np.array_equal(res, expected))

    def testDiagExecution(self):
        # 2-d  6 * 6
        a = arange(36, chunk_size=2).reshape(6, 6)

        d = diag(a)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(36).reshape(6, 6))
        np.testing.assert_equal(res, expected)

        d = diag(a, k=1)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(36).reshape(6, 6), k=1)
        np.testing.assert_equal(res, expected)

        d = diag(a, k=3)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(36).reshape(6, 6), k=3)
        np.testing.assert_equal(res, expected)

        d = diag(a, k=-2)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(36).reshape(6, 6), k=-2)
        np.testing.assert_equal(res, expected)

        d = diag(a, k=-5)
        res = self.executor.execute_tensor(d)[0]
        expected = np.diag(np.arange(36).reshape(6, 6), k=-5)
        np.testing.assert_equal(res, expected)

        # 2-d  6 * 6 sparse, no tensor
        a = sps.rand(6, 6, density=.1)

        d = diag(a)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(a.toarray())
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=1)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(a.toarray(), k=1)
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=3)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(a.toarray(), k=3)
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=-2)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(a.toarray(), k=-2)
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=-5)
        res = self.executor.execute_tensor(d)[0]
        expected = np.diag(a.toarray(), k=-5)
        np.testing.assert_equal(res.toarray(), expected)

        # 2-d  6 * 6 sparse, from tensor
        raw_a = sps.rand(6, 6, density=.1)
        a = tensor(raw_a, chunk_size=2)

        d = diag(a)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(raw_a.toarray())
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=1)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(raw_a.toarray(), k=1)
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=3)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(raw_a.toarray(), k=3)
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=-2)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(raw_a.toarray(), k=-2)
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=-5)
        res = self.executor.execute_tensor(d)[0]
        expected = np.diag(raw_a.toarray(), k=-5)
        np.testing.assert_equal(res.toarray(), expected)

        # 2-d  4 * 9
        a = arange(36, chunk_size=2).reshape(4, 9)

        d = diag(a)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(36).reshape(4, 9))
        np.testing.assert_equal(res, expected)

        d = diag(a, k=1)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(36).reshape(4, 9), k=1)
        np.testing.assert_equal(res, expected)

        d = diag(a, k=3)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(36).reshape(4, 9), k=3)
        np.testing.assert_equal(res, expected)

        d = diag(a, k=-2)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(36).reshape(4, 9), k=-2)
        np.testing.assert_equal(res, expected)

        d = diag(a, k=-3)
        res = self.executor.execute_tensor(d)[0]
        expected = np.diag(np.arange(36).reshape(4, 9), k=-3)
        np.testing.assert_equal(res, expected)

        # 2-d  4 * 9 sparse, no tensor
        a = sps.rand(4, 9, density=.1)

        d = diag(a)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(a.toarray())
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=1)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(a.toarray(), k=1)
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=3)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(a.toarray(), k=3)
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=-2)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(a.toarray(), k=-2)
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=-3)
        res = self.executor.execute_tensor(d)[0]
        expected = np.diag(a.toarray(), k=-3)
        np.testing.assert_equal(res.toarray(), expected)

        # 2-d  4 * 9 sparse, from tensor
        raw_a = sps.rand(4, 9, density=.1)
        a = tensor(raw_a, chunk_size=2)

        d = diag(a)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(raw_a.toarray())
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=1)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(raw_a.toarray(), k=1)
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=3)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(raw_a.toarray(), k=3)
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=-2)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(raw_a.toarray(), k=-2)
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=-3)
        res = self.executor.execute_tensor(d)[0]
        expected = np.diag(raw_a.toarray(), k=-3)
        np.testing.assert_equal(res.toarray(), expected)

        # 1-d
        a = arange(5, chunk_size=2)

        d = diag(a)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(5))
        np.testing.assert_equal(res, expected)
        self.assertTrue(res.flags['C_CONTIGUOUS'])
        self.assertFalse(res.flags['F_CONTIGUOUS'])

        d = diag(a, k=1)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(5), k=1)
        np.testing.assert_equal(res, expected)

        d = diag(a, k=3)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(5), k=3)
        np.testing.assert_equal(res, expected)

        d = diag(a, k=-2)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(5), k=-2)
        np.testing.assert_equal(res, expected)

        d = diag(a, k=-3)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(5), k=-3)
        np.testing.assert_equal(res, expected)

        d = diag(a, sparse=True)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(5))
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=1, sparse=True)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(5), k=1)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=2, sparse=True)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(5), k=2)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=-2, sparse=True)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(5), k=-2)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res.toarray(), expected)

        d = diag(a, k=-3, sparse=True)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.diag(np.arange(5), k=-3)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res.toarray(), expected)

    def testDiagflatExecution(self):
        a = diagflat([[1, 2], [3, 4]], chunk_size=1)

        res = self.executor.execute_tensor(a, concat=True)[0]
        expected = np.diagflat([[1, 2], [3, 4]])
        np.testing.assert_equal(res, expected)

        d = tensor([[1, 2], [3, 4]], chunk_size=1)
        a = diagflat(d)

        res = self.executor.execute_tensor(a, concat=True)[0]
        expected = np.diagflat([[1, 2], [3, 4]])
        np.testing.assert_equal(res, expected)

        a = diagflat([1, 2], 1, chunk_size=1)

        res = self.executor.execute_tensor(a, concat=True)[0]
        expected = np.diagflat([1, 2], 1)
        np.testing.assert_equal(res, expected)

        d = tensor([[1, 2]], chunk_size=1)
        a = diagflat(d, 1)

        res = self.executor.execute_tensor(a, concat=True)[0]
        expected = np.diagflat([1, 2], 1)
        np.testing.assert_equal(res, expected)

    def testEyeExecution(self):
        t = eye(5, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5)
        np.testing.assert_equal(res, expected)

        t = eye(5, k=1, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5, k=1)
        np.testing.assert_equal(res, expected)

        t = eye(5, k=2, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5, k=2)
        np.testing.assert_equal(res, expected)

        t = eye(5, k=-1, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5, k=-1)
        np.testing.assert_equal(res, expected)

        t = eye(5, k=-3, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5, k=-3)
        np.testing.assert_equal(res, expected)

        t = eye(5, M=3, k=1, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5, M=3, k=1)
        np.testing.assert_equal(res, expected)

        t = eye(5, M=3, k=-3, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5, M=3, k=-3)
        np.testing.assert_equal(res, expected)

        t = eye(5, M=7, k=1, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5, M=7, k=1)
        np.testing.assert_equal(res, expected)

        t = eye(5, M=8, k=-3, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5, M=8, k=-3)
        np.testing.assert_equal(res, expected)

        t = eye(2, dtype=int)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertEqual(res.dtype, np.int_)

        # test sparse
        t = eye(5, sparse=True, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res.toarray(), expected)

        t = eye(5, k=1, sparse=True, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5, k=1)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res.toarray(), expected)

        t = eye(5, k=2, sparse=True, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5, k=2)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res.toarray(), expected)

        t = eye(5, k=-1, sparse=True, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5, k=-1)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res.toarray(), expected)

        t = eye(5, k=-3, sparse=True, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5, k=-3)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res.toarray(), expected)

        t = eye(5, M=3, k=1, sparse=True, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5, M=3, k=1)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res.toarray(), expected)

        t = eye(5, M=3, k=-3, sparse=True, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5, M=3, k=-3)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res.toarray(), expected)

        t = eye(5, M=7, k=1, sparse=True, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5, M=7, k=1)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res.toarray(), expected)

        t = eye(5, M=8, k=-3, sparse=True, chunk_size=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.eye(5, M=8, k=-3)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res.toarray(), expected)

        t = eye(5, M=9, k=-3, chunk_size=2, order='F')

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(res.flags['C_CONTIGUOUS'])
        self.assertFalse(res.flags['F_CONTIGUOUS'])

    def testLinspaceExecution(self):
        a = linspace(2.0, 9.0, num=11, chunk_size=3)

        res = self.executor.execute_tensor(a, concat=True)[0]
        expected = np.linspace(2.0, 9.0, num=11)
        np.testing.assert_allclose(res, expected)

        a = linspace(2.0, 9.0, num=11, endpoint=False, chunk_size=3)

        res = self.executor.execute_tensor(a, concat=True)[0]
        expected = np.linspace(2.0, 9.0, num=11, endpoint=False)
        np.testing.assert_allclose(res, expected)

        a = linspace(2.0, 9.0, num=11, chunk_size=3, dtype=int)

        res = self.executor.execute_tensor(a, concat=True)[0]
        self.assertEqual(res.dtype, np.int_)

    def testMeshgridExecution(self):
        a = arange(5, chunk_size=2)
        b = arange(6, 12, chunk_size=3)
        c = arange(12, 19, chunk_size=4)

        A, B, C = meshgrid(a, b, c)

        A_res = self.executor.execute_tensor(A, concat=True)[0]
        A_expected = np.meshgrid(np.arange(5), np.arange(6, 12), np.arange(12, 19))[0]
        np.testing.assert_equal(A_res, A_expected)

        B_res = self.executor.execute_tensor(B, concat=True)[0]
        B_expected = np.meshgrid(np.arange(5), np.arange(6, 12), np.arange(12, 19))[1]
        np.testing.assert_equal(B_res, B_expected)

        C_res = self.executor.execute_tensor(C, concat=True)[0]
        C_expected = np.meshgrid(np.arange(5), np.arange(6, 12), np.arange(12, 19))[2]
        np.testing.assert_equal(C_res, C_expected)

        A, B, C = meshgrid(a, b, c, indexing='ij')

        A_res = self.executor.execute_tensor(A, concat=True)[0]
        A_expected = np.meshgrid(np.arange(5), np.arange(6, 12), np.arange(12, 19), indexing='ij')[0]
        np.testing.assert_equal(A_res, A_expected)

        B_res = self.executor.execute_tensor(B, concat=True)[0]
        B_expected = np.meshgrid(np.arange(5), np.arange(6, 12), np.arange(12, 19), indexing='ij')[1]
        np.testing.assert_equal(B_res, B_expected)

        C_res = self.executor.execute_tensor(C, concat=True)[0]
        C_expected = np.meshgrid(np.arange(5), np.arange(6, 12), np.arange(12, 19), indexing='ij')[2]
        np.testing.assert_equal(C_res, C_expected)

        A, B, C = meshgrid(a, b, c, sparse=True)

        A_res = self.executor.execute_tensor(A, concat=True)[0]
        A_expected = np.meshgrid(np.arange(5), np.arange(6, 12), np.arange(12, 19), sparse=True)[0]
        np.testing.assert_equal(A_res, A_expected)

        B_res = self.executor.execute_tensor(B, concat=True)[0]
        B_expected = np.meshgrid(np.arange(5), np.arange(6, 12), np.arange(12, 19), sparse=True)[1]
        np.testing.assert_equal(B_res, B_expected)

        C_res = self.executor.execute_tensor(C, concat=True)[0]
        C_expected = np.meshgrid(np.arange(5), np.arange(6, 12), np.arange(12, 19), sparse=True)[2]
        np.testing.assert_equal(C_res, C_expected)

        A, B, C = meshgrid(a, b, c, indexing='ij', sparse=True)

        A_res = self.executor.execute_tensor(A, concat=True)[0]
        A_expected = np.meshgrid(np.arange(5), np.arange(6, 12), np.arange(12, 19),
                                 indexing='ij', sparse=True)[0]
        np.testing.assert_equal(A_res, A_expected)

        B_res = self.executor.execute_tensor(B, concat=True)[0]
        B_expected = np.meshgrid(np.arange(5), np.arange(6, 12), np.arange(12, 19),
                                 indexing='ij', sparse=True)[1]
        np.testing.assert_equal(B_res, B_expected)

        C_res = self.executor.execute_tensor(C, concat=True)[0]
        C_expected = np.meshgrid(np.arange(5), np.arange(6, 12), np.arange(12, 19),
                                 indexing='ij', sparse=True)[2]
        np.testing.assert_equal(C_res, C_expected)

    def testIndicesExecution(self):
        grid = indices((2, 3), chunk_size=1)

        res = self.executor.execute_tensor(grid, concat=True)[0]
        expected = np.indices((2, 3))
        np.testing.assert_equal(res, expected)

        res = self.executor.execute_tensor(grid[0], concat=True)[0]
        np.testing.assert_equal(res, expected[0])

        res = self.executor.execute_tensor(grid[1], concat=True)[0]
        np.testing.assert_equal(res, expected[1])

    def testTriuExecution(self):
        a = arange(24, chunk_size=2).reshape(2, 3, 4)

        t = triu(a)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.triu(np.arange(24).reshape(2, 3, 4))
        np.testing.assert_equal(res, expected)

        t = triu(a, k=1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.triu(np.arange(24).reshape(2, 3, 4), k=1)
        np.testing.assert_equal(res, expected)

        t = triu(a, k=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.triu(np.arange(24).reshape(2, 3, 4), k=2)
        np.testing.assert_equal(res, expected)

        t = triu(a, k=-1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.triu(np.arange(24).reshape(2, 3, 4), k=-1)
        np.testing.assert_equal(res, expected)

        t = triu(a, k=-2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.triu(np.arange(24).reshape(2, 3, 4), k=-2)
        np.testing.assert_equal(res, expected)

        # test sparse
        a = arange(12, chunk_size=2).reshape(3, 4).tosparse()

        t = triu(a)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.triu(np.arange(12).reshape(3, 4))
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res, expected)

        t = triu(a, k=1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.triu(np.arange(12).reshape(3, 4), k=1)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res, expected)

        t = triu(a, k=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.triu(np.arange(12).reshape(3, 4), k=2)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res, expected)

        t = triu(a, k=-1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.triu(np.arange(12).reshape(3, 4), k=-1)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res, expected)

        t = triu(a, k=-2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.triu(np.arange(12).reshape(3, 4), k=-2)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res, expected)

        raw = np.asfortranarray(np.random.rand(10, 7))
        a = tensor(raw, chunk_size=3)

        t = triu(a, k=-2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.triu(raw, k=-2)
        np.testing.assert_array_equal(res, expected)
        self.assertEqual(res.flags['C_CONTIGUOUS'], expected.flags['C_CONTIGUOUS'])
        self.assertEqual(res.flags['F_CONTIGUOUS'], expected.flags['F_CONTIGUOUS'])

    def testTrilExecution(self):
        a = arange(24, chunk_size=2).reshape(2, 3, 4)

        t = tril(a)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.tril(np.arange(24).reshape(2, 3, 4))
        np.testing.assert_equal(res, expected)

        t = tril(a, k=1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.tril(np.arange(24).reshape(2, 3, 4), k=1)
        np.testing.assert_equal(res, expected)

        t = tril(a, k=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.tril(np.arange(24).reshape(2, 3, 4), k=2)
        np.testing.assert_equal(res, expected)

        t = tril(a, k=-1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.tril(np.arange(24).reshape(2, 3, 4), k=-1)
        np.testing.assert_equal(res, expected)

        t = tril(a, k=-2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.tril(np.arange(24).reshape(2, 3, 4), k=-2)
        np.testing.assert_equal(res, expected)

        a = arange(12, chunk_size=2).reshape(3, 4).tosparse()

        t = tril(a)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.tril(np.arange(12).reshape(3, 4))
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res, expected)

        t = tril(a, k=1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.tril(np.arange(12).reshape(3, 4), k=1)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res, expected)

        t = tril(a, k=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.tril(np.arange(12).reshape(3, 4), k=2)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res, expected)

        t = tril(a, k=-1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.tril(np.arange(12).reshape(3, 4), k=-1)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res, expected)

        t = tril(a, k=-2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.tril(np.arange(12).reshape(3, 4), k=-2)
        self.assertIsInstance(res, SparseNDArray)
        np.testing.assert_equal(res, expected)

    def testIndexTrickExecution(self):
        mgrid = nd_grid()
        t = mgrid[0:5, 0:5]

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.lib.index_tricks.nd_grid()[0:5, 0:5]
        np.testing.assert_equal(res, expected)

        t = mgrid[-1:1:5j]

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.lib.index_tricks.nd_grid()[-1:1:5j]
        np.testing.assert_equal(res, expected)

        ogrid = nd_grid(sparse=True)

        t = ogrid[0:5, 0:5]

        res = [self.executor.execute_tensor(o, concat=True)[0] for o in t]
        expected = np.lib.index_tricks.nd_grid(sparse=True)[0:5, 0:5]
        [np.testing.assert_equal(r, e) for r, e in zip(res, expected)]

    @unittest.skipIf(tiledb is None, 'tiledb not installed')
    def testReadTileDBExecution(self):
        ctx = tiledb.Ctx()

        tempdir = tempfile.mkdtemp()
        try:
            # create TileDB dense array
            dom = tiledb.Domain(
                tiledb.Dim(ctx=ctx, domain=(1, 100), tile=30, dtype=np.int32),
                tiledb.Dim(ctx=ctx, domain=(0, 90), tile=22, dtype=np.int32),
                tiledb.Dim(ctx=ctx, domain=(0, 9), tile=8, dtype=np.int32),
                ctx=ctx,
            )
            schema = tiledb.ArraySchema(ctx=ctx, domain=dom, sparse=False,
                                        attrs=[tiledb.Attr(ctx=ctx, dtype=np.float64)])
            tiledb.DenseArray.create(tempdir, schema)

            expected = np.random.rand(100, 91, 10)
            with tiledb.DenseArray(uri=tempdir, ctx=ctx, mode='w') as arr:
                arr.write_direct(expected)

            a = fromtiledb(tempdir, ctx=ctx)
            result = self.executor.execute_tensor(a, concat=True)[0]

            np.testing.assert_allclose(expected, result)
        finally:
            shutil.rmtree(tempdir)

        tempdir = tempfile.mkdtemp()
        try:
            # create 2-d TileDB sparse array
            dom = tiledb.Domain(
                tiledb.Dim(ctx=ctx, domain=(0, 99), tile=30, dtype=np.int32),
                tiledb.Dim(ctx=ctx, domain=(2, 11), tile=8, dtype=np.int32),
                ctx=ctx,
            )
            schema = tiledb.ArraySchema(ctx=ctx, domain=dom, sparse=True,
                                        attrs=[tiledb.Attr(ctx=ctx, dtype=np.float64)])
            tiledb.SparseArray.create(tempdir, schema)

            expected = sps.rand(100, 10, density=0.01)
            with tiledb.SparseArray(uri=tempdir, ctx=ctx, mode='w') as arr:
                I, J = expected.row, expected.col + 2
                arr[I, J] = {arr.attr(0).name: expected.data}

            a = fromtiledb(tempdir, ctx=ctx)
            result = self.executor.execute_tensor(a, concat=True)[0]

            np.testing.assert_allclose(expected.toarray(), result.toarray())
        finally:
            shutil.rmtree(tempdir)

        tempdir = tempfile.mkdtemp()
        try:
            # create 1-d TileDB sparse array
            dom = tiledb.Domain(
                tiledb.Dim(ctx=ctx, domain=(1, 100), tile=30, dtype=np.int32),
                ctx=ctx,
            )
            schema = tiledb.ArraySchema(ctx=ctx, domain=dom, sparse=True,
                                        attrs=[tiledb.Attr(ctx=ctx, dtype=np.float64)])
            tiledb.SparseArray.create(tempdir, schema)

            expected = sps.rand(1, 100, density=0.05)
            with tiledb.SparseArray(uri=tempdir, ctx=ctx, mode='w') as arr:
                arr[expected.col + 1] = expected.data

            a = fromtiledb(tempdir, ctx=ctx)
            result = self.executor.execute_tensor(a, concat=True)[0]

            np.testing.assert_allclose(expected.toarray()[0], result.toarray())
        finally:
            shutil.rmtree(tempdir)

        tempdir = tempfile.mkdtemp()
        try:
            # create TileDB dense array with column-major
            dom = tiledb.Domain(
                tiledb.Dim(ctx=ctx, domain=(1, 100), tile=30, dtype=np.int32),
                tiledb.Dim(ctx=ctx, domain=(0, 90), tile=22, dtype=np.int32),
                tiledb.Dim(ctx=ctx, domain=(0, 9), tile=8, dtype=np.int32),
                ctx=ctx,
            )
            schema = tiledb.ArraySchema(ctx=ctx, domain=dom, sparse=False, cell_order='F',
                                        attrs=[tiledb.Attr(ctx=ctx, dtype=np.float64)])
            tiledb.DenseArray.create(tempdir, schema)

            expected = np.asfortranarray(np.random.rand(100, 91, 10))
            with tiledb.DenseArray(uri=tempdir, ctx=ctx, mode='w') as arr:
                arr.write_direct(expected)

            a = fromtiledb(tempdir, ctx=ctx)
            result = self.executor.execute_tensor(a, concat=True)[0]

            np.testing.assert_allclose(expected, result)
            self.assertTrue(result.flags['F_CONTIGUOUS'])
            self.assertFalse(result.flags['C_CONTIGUOUS'])
        finally:
            shutil.rmtree(tempdir)

    def testFromDataFrameExecution(self):
        mdf = md.DataFrame({'angle': [0, 3, 4], 'degree': [360, 180, 360]},
                           index=['circle', 'triangle', 'rectangle'])
        tensor_result = self.executor.execute_tensor(from_dataframe(mdf))
        tensor_expected = self.executor.execute_tensor(mt.tensor([[0, 360], [3, 180], [4, 360]]))
        np.testing.assert_equal(tensor_result, tensor_expected)

        # test up-casting
        mdf2 = md.DataFrame({'a': [0.1, 0.2, 0.3], 'b': [1, 2, 3]})
        tensor_result2 = self.executor.execute_tensor(from_dataframe(mdf2))
        np.testing.assert_equal(tensor_result2[0].dtype, np.dtype('float64'))
        tensor_expected2 = self.executor.execute_tensor(mt.tensor([[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]]))
        np.testing.assert_equal(tensor_result2, tensor_expected2)

        raw = [[0.1, 0.2, 0.4], [0.4, 0.7, 0.3]]
        mdf3 = md.DataFrame(raw, columns=list('abc'), chunk_size=2)
        tensor_result3 = self.executor.execute_tensor(from_dataframe(mdf3), concat=True)[0]
        np.testing.assert_array_equal(tensor_result3, np.asarray(raw))
        self.assertTrue(tensor_result3.flags['F_CONTIGUOUS'])
        self.assertFalse(tensor_result3.flags['C_CONTIGUOUS'])

        # test from series
        series = md.Series([1, 2, 3])
        tensor_result = series.to_tensor().execute()
        np.testing.assert_array_equal(tensor_result, np.array([1, 2, 3]))

        series = md.Series(range(10), chunk_size=3)
        tensor_result = series.to_tensor().execute()
        np.testing.assert_array_equal(tensor_result, np.arange(10))

        # test from index
        index = md.Index(pd.MultiIndex.from_tuples([(0, 1), (2, 3), (4, 5)]))
        tensor_result = index.to_tensor(extract_multi_index=True).execute()
        np.testing.assert_array_equal(tensor_result, np.arange(6).reshape((3, 2)))

        index = md.Index(pd.MultiIndex.from_tuples([(0, 1), (2, 3), (4, 5)]))
        tensor_result = index.to_tensor(extract_multi_index=False).execute()
        np.testing.assert_array_equal(tensor_result,
                                      pd.MultiIndex.from_tuples([(0, 1), (2, 3), (4, 5)]).to_series())

    @unittest.skipIf(h5py is None, 'h5py not installed')
    def testReadHDF5Execution(self):
        test_array = np.random.RandomState(0).rand(20, 10)
        group_name = 'test_group'
        dataset_name = 'test_dataset'

        with self.assertRaises(TypeError):
            fromhdf5(object())

        with tempfile.TemporaryDirectory() as d:
            filename = os.path.join(d, f'test_read_{int(time.time())}.hdf5')
            with h5py.File(filename, 'w') as f:
                g = f.create_group(group_name)
                g.create_dataset(dataset_name, chunks=(7, 4), data=test_array)

            # test filename
            r = fromhdf5(filename, group=group_name, dataset=dataset_name)

            result = self.executor.execute_tensor(r, concat=True)[0]
            np.testing.assert_array_equal(result, test_array)
            self.assertEqual(r.extra_params['raw_chunk_size'], (7, 4))

            with self.assertRaises(ValueError):
                fromhdf5(filename)

            with self.assertRaises(ValueError):
                fromhdf5(filename, dataset='non_exist')

            with h5py.File(filename, 'r') as f:
                # test file
                r = fromhdf5(f, group=group_name, dataset=dataset_name)

                result = self.executor.execute_tensor(r, concat=True)[0]
                np.testing.assert_array_equal(result, test_array)

                with self.assertRaises(ValueError):
                    fromhdf5(f)

                with self.assertRaises(ValueError):
                    fromhdf5(f, dataset='non_exist')

                # test dataset
                ds = f[f'{group_name}/{dataset_name}']
                r = fromhdf5(ds)

                result = self.executor.execute_tensor(r, concat=True)[0]
                np.testing.assert_array_equal(result, test_array)

    @unittest.skipIf(zarr is None, 'zarr not installed')
    def testReadZarrExecution(self):
        test_array = np.random.RandomState(0).rand(20, 10)
        group_name = 'test_group'
        dataset_name = 'test_dataset'

        with self.assertRaises(TypeError):
            fromzarr(object())

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, f'test_read_{int(time.time())}.zarr')

            group = zarr.group(path)
            arr = group.array(group_name + '/' + dataset_name, test_array, chunks=(7, 4))

            r = fromzarr(arr)

            result = self.executor.execute_tensor(r, concat=True)[0]
            np.testing.assert_array_equal(result, test_array)
            self.assertGreater(len(get_tiled(r).chunks), 1)

            arr = zarr.open_array(f'{path}/{group_name}/{dataset_name}')
            r = fromzarr(arr)

            result = self.executor.execute_tensor(r, concat=True)[0]
            np.testing.assert_array_equal(result, test_array)
            self.assertGreater(len(get_tiled(r).chunks), 1)

            r = fromzarr(path, group=group_name, dataset=dataset_name)

            result = self.executor.execute_tensor(r, concat=True)[0]
            np.testing.assert_array_equal(result, test_array)
            self.assertGreater(len(get_tiled(r).chunks), 1)

            r = fromzarr(path + '/' + group_name + '/' + dataset_name)

            result = self.executor.execute_tensor(r, concat=True)[0]
            np.testing.assert_array_equal(result, test_array)
            self.assertGreater(len(get_tiled(r).chunks), 1)
