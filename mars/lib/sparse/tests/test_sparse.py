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

from mars.tests.core import TestBase
from mars.lib.sparse import SparseNDArray, SparseVector, SparseMatrix
from mars.lib.sparse.core import issparse
import mars.lib.sparse as mls


class Test(TestBase):
    def setUp(self):
        self.s1 = sps.csr_matrix([[1, 0, 1], [0, 0, 1]])
        self.s2 = sps.csr_matrix([[0, 1, 1], [1, 0, 1]])
        self.v1_data = np.random.rand(3)
        self.v1 = sps.csr_matrix(self.v1_data)
        self.v2_data = np.random.rand(2)
        self.v2 = sps.csr_matrix(self.v2_data)
        self.d1 = np.array([1, 2, 3])

    def testSparseCreation(self):
        s = SparseNDArray(self.s1)
        self.assertEqual(s.ndim, 2)
        self.assertIsInstance(s, SparseMatrix)
        self.assertArrayEqual(s.toarray(), self.s1.A)
        self.assertArrayEqual(s.todense(), self.s1.A)

        v = SparseNDArray(self.v1, shape=(3,))
        self.assertTrue(s.ndim, 1)
        self.assertIsInstance(v, SparseVector)
        self.assertEqual(v.shape, (3,))
        self.assertArrayEqual(v.todense(), self.v1_data)
        self.assertArrayEqual(v.toarray(), self.v1_data)
        self.assertArrayEqual(v, self.v1_data)

    def _nan_equal(self, a, b):
        try:
            np.testing.assert_equal(a, b)
        except AssertionError:
            return False
        return True

    def assertArrayEqual(self, a, b):
        if issparse(a):
            a = a.toarray()
        else:
            a = np.asarray(a)
        if issparse(b):
            b = b.toarray()
        else:
            b = np.asarray(b)
        return self.assertTrue(self._nan_equal(a, b))

    def testSparseAdd(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        self.assertArrayEqual(s1 + s2, self.s1 + self.s2)
        self.assertArrayEqual(s1 + self.d1, self.s1 + self.d1)
        self.assertArrayEqual(self.d1 + s1, self.d1 + self.s1)
        r = sps.csr_matrix(((self.s1.data + 1), self.s1.indices, self.s1.indptr), self.s1.shape)
        self.assertArrayEqual(s1 + 1, r)
        r = sps.csr_matrix(((1 + self.s1.data), self.s1.indices, self.s1.indptr), self.s1.shape)
        self.assertArrayEqual(1 + s1, r)

        # test sparse vector
        v = SparseNDArray(self.v1, shape=(3,))
        self.assertArrayEqual(v + v, self.v1_data + self.v1_data)
        self.assertArrayEqual(v + self.d1, self.v1_data + self.d1)
        self.assertArrayEqual(self.d1 + v, self.d1 + self.v1_data)
        r = sps.csr_matrix(((self.v1.data + 1), self.v1.indices, self.v1.indptr), self.v1.shape)
        self.assertArrayEqual(v + 1, r.toarray().reshape(3))
        r = sps.csr_matrix(((1 + self.v1.data), self.v1.indices, self.v1.indptr), self.v1.shape)
        self.assertArrayEqual(1 + v, r.toarray().reshape(3))

    def testSparseSubtract(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        self.assertArrayEqual(s1 - s2, self.s1 - self.s2)
        self.assertArrayEqual(s1 - self.d1, self.s1 - self.d1)
        self.assertArrayEqual(self.d1 - s1, self.d1 - self.s1)
        r = sps.csr_matrix(((self.s1.data - 1), self.s1.indices, self.s1.indptr), self.s1.shape)
        self.assertArrayEqual(s1 - 1, r)
        r = sps.csr_matrix(((1 - self.s1.data), self.s1.indices, self.s1.indptr), self.s1.shape)
        self.assertArrayEqual(1 - s1, r)

        # test sparse vector
        v = SparseNDArray(self.v1, shape=(3,))
        self.assertArrayEqual(v - v, self.v1_data - self.v1_data)
        self.assertArrayEqual(v - self.d1, self.v1_data - self.d1)
        self.assertArrayEqual(self.d1 - v, self.d1 - self.v1_data)
        r = sps.csr_matrix(((self.v1.data - 1), self.v1.indices, self.v1.indptr), self.v1.shape)
        self.assertArrayEqual(v - 1, r.toarray().reshape(3))
        r = sps.csr_matrix(((1 - self.v1.data), self.v1.indices, self.v1.indptr), self.v1.shape)
        self.assertArrayEqual(1 - v, r.toarray().reshape(3))

    def testSparseMultiply(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        self.assertArrayEqual(s1 * s2, self.s1.multiply(self.s2))
        self.assertArrayEqual(s1 * self.d1, self.s1.multiply(self.d1))
        self.assertArrayEqual(self.d1 * s1, self.s1.multiply(self.d1))
        self.assertArrayEqual(s1 * 2, self.s1 * 2)
        self.assertArrayEqual(2 * s1, self.s1 * 2)

        # test sparse vector
        v = SparseNDArray(self.v1, shape=(3,))
        self.assertArrayEqual(v * v, self.v1_data * self.v1_data)
        self.assertArrayEqual(v * self.d1, self.v1_data * self.d1)
        self.assertArrayEqual(self.d1 * v, self.d1 * self.v1_data)
        r = sps.csr_matrix(((self.v1.data * 1), self.v1.indices, self.v1.indptr), self.v1.shape)
        self.assertArrayEqual(v * 1, r.toarray().reshape(3))
        r = sps.csr_matrix(((1 * self.v1.data), self.v1.indices, self.v1.indptr), self.v1.shape)
        self.assertArrayEqual(1 * v, r.toarray().reshape(3))

    def testSparseDivide(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        self.assertArrayEqual(s1 / s2, self.s1 / self.s2)
        self.assertArrayEqual(s1 / self.d1, self.s1 / self.d1)
        self.assertArrayEqual(self.d1 / s1, self.d1 / self.s1.toarray())
        self.assertArrayEqual(s1 / 2, self.s1 / 2)
        self.assertArrayEqual(2 / s1, 2 / self.s1.toarray())

        # test sparse vector
        v = SparseNDArray(self.v1, shape=(3,))
        self.assertArrayEqual(v / v, self.v1_data / self.v1_data)
        self.assertArrayEqual(v / self.d1, self.v1_data / self.d1)
        self.assertArrayEqual(self.d1 / v, self.d1 / self.v1_data)
        r = sps.csr_matrix(((self.v1.data / 1), self.v1.indices, self.v1.indptr), self.v1.shape)
        self.assertArrayEqual(v / 1, r.toarray().reshape(3))
        r = sps.csr_matrix(((1 / self.v1.data), self.v1.indices, self.v1.indptr), self.v1.shape)
        self.assertArrayEqual(1 / v, r.toarray().reshape(3))

    def testSparseFloorDivide(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        self.assertArrayEqual(s1 // s2, self.s1.toarray() // self.s2.toarray())
        self.assertArrayEqual(s1 // self.d1, self.s1.toarray() // self.d1)
        self.assertArrayEqual(self.d1 // s1, self.d1 // self.s1.toarray())
        self.assertArrayEqual(s1 // 2, self.s1.toarray() // 2)
        self.assertArrayEqual(2 // s1, 2 // self.s1.toarray())

        # test sparse vector
        v = SparseNDArray(self.v1, shape=(3,))
        self.assertArrayEqual(v // v, self.v1_data // self.v1_data)
        self.assertArrayEqual(v // self.d1, self.v1_data // self.d1)
        self.assertArrayEqual(self.d1 // v, self.d1 // self.v1_data)
        r = sps.csr_matrix(((self.v1.data // 1), self.v1.indices, self.v1.indptr), self.v1.shape)
        self.assertArrayEqual(v // 1, r.toarray().reshape(3))
        r = sps.csr_matrix(((1 // self.v1.data), self.v1.indices, self.v1.indptr), self.v1.shape)
        self.assertArrayEqual(1 // v, r.toarray().reshape(3))

    def testSparsePower(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        self.assertArrayEqual(s1 ** s2, self.s1.toarray() ** self.s2.toarray())
        self.assertArrayEqual(s1 ** self.d1, self.s1.toarray() ** self.d1)
        self.assertArrayEqual(self.d1 ** s1, self.d1 ** self.s1.toarray())
        self.assertArrayEqual(s1 ** 2, self.s1.power(2))
        self.assertArrayEqual(2 ** s1, 2 ** self.s1.toarray())

        # test sparse vector
        v = SparseNDArray(self.v1, shape=(3,))
        self.assertArrayEqual(v ** v, self.v1_data ** self.v1_data)
        self.assertArrayEqual(v ** self.d1, self.v1_data ** self.d1)
        self.assertArrayEqual(self.d1 ** v, self.d1 ** self.v1_data)
        r = sps.csr_matrix(((self.v1.data ** 1), self.v1.indices, self.v1.indptr), self.v1.shape)
        self.assertArrayEqual(v ** 1, r.toarray().reshape(3))
        r = sps.csr_matrix(((1 ** self.v1.data), self.v1.indices, self.v1.indptr), self.v1.shape)
        self.assertArrayEqual(1 ** v, r.toarray().reshape(3))

    def testSparseMod(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        self.assertArrayEqual(s1 % s2, self.s1.toarray() % self.s2.toarray())
        self.assertArrayEqual(s1 % self.d1, self.s1.toarray() % self.d1)
        self.assertArrayEqual(self.d1 % s1, self.d1 % self.s1.toarray())
        self.assertArrayEqual(s1 % 2, self.s1.toarray() % 2)
        self.assertArrayEqual(2 % s1, 2 % self.s1.toarray())

        # test sparse vector
        v = SparseNDArray(self.v1, shape=(3,))
        self.assertArrayEqual(v % v, self.v1_data % self.v1_data)
        self.assertArrayEqual(v % self.d1, self.v1_data % self.d1)
        self.assertArrayEqual(self.d1 % v, self.d1 % self.v1_data)
        r = sps.csr_matrix(((self.v1.data % 1), self.v1.indices, self.v1.indptr), self.v1.shape)
        self.assertArrayEqual(v % 1, r.toarray().reshape(3))
        r = sps.csr_matrix(((1 % self.v1.data), self.v1.indices, self.v1.indptr), self.v1.shape)
        self.assertArrayEqual(1 % v, r.toarray().reshape(3))

    def testSparseBin(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)
        v1 = SparseNDArray(self.v1, shape=(3,))

        for method in ('fmod', 'logaddexp', 'logaddexp2', 'equal', 'not_equal',
                       'less', 'less_equal', 'greater', 'greater_equal', 'hypot', 'arctan2'):
            lm, rm = getattr(mls, method), getattr(np, method)
            self.assertArrayEqual(lm(s1, s2), rm(self.s1.toarray(), self.s2.toarray()))
            self.assertArrayEqual(lm(s1, self.d1), rm(self.s1.toarray(), self.d1))
            self.assertArrayEqual(lm(self.d1, s1), rm(self.d1, self.s1.toarray()))
            r1 = sps.csr_matrix((rm(self.s1.data, 2), self.s1.indices, self.s1.indptr), self.s1.shape)
            self.assertArrayEqual(lm(s1, 2), r1)
            r2 = sps.csr_matrix((rm(2, self.s1.data), self.s1.indices, self.s1.indptr), self.s1.shape)
            self.assertArrayEqual(lm(2, s1), r2)

            # test sparse
            self.assertArrayEqual(lm(v1, v1), rm(self.v1_data, self.v1_data))
            self.assertArrayEqual(lm(v1, self.d1), rm(self.v1_data, self.d1))
            self.assertArrayEqual(lm(self.d1, v1), rm(self.d1, self.v1_data))
            self.assertArrayEqual(lm(v1, 2), rm(self.v1_data, 2))
            self.assertArrayEqual(lm(2, v1), rm(2, self.v1_data))

    def testSparseUnary(self):
        s1 = SparseNDArray(self.s1)
        v1 = SparseNDArray(self.v1, shape=(3,))

        for method in ('negative', 'positive', 'absolute', 'abs', 'fabs', 'rint',
                       'sign', 'conj', 'exp', 'exp2', 'log', 'log2', 'log10',
                       'expm1', 'log1p', 'sqrt', 'square', 'cbrt', 'reciprocal',
                       'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
                       'arcsinh', 'arccosh', 'arctanh', 'deg2rad', 'rad2deg',
                       'angle', 'isnan', 'isinf', 'signbit', 'sinc', 'isreal', 'isfinite'):
            lm, rm = getattr(mls, method), getattr(np, method)
            r = sps.csr_matrix((rm(self.s1.data), self.s1.indices, self.s1.indptr), self.s1.shape)
            self.assertArrayEqual(lm(s1), r)
            self.assertArrayEqual(lm(v1), rm(self.v1_data))

    def testSparseDot(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)
        v1 = SparseNDArray(self.v1, shape=(3,))
        v2 = SparseNDArray(self.v2, shape=(2,))

        self.assertArrayEqual(mls.dot(s1, s2.T), self.s1.dot(self.s2.T))
        self.assertArrayEqual(s1.dot(self.d1), self.s1.dot(self.d1))
        self.assertArrayEqual(self.d1.dot(s1.T), self.d1.dot(self.s1.T.toarray()))

        self.assertArrayEqual(mls.tensordot(s1, s2.T, axes=(1, 0)), self.s1.dot(self.s2.T))
        self.assertArrayEqual(mls.tensordot(s1, self.d1, axes=(1, -1)), self.s1.dot(self.d1))
        self.assertArrayEqual(mls.tensordot(self.d1, s1.T, axes=(0, 0)), self.d1.dot(self.s1.T.toarray()))

        self.assertArrayEqual(mls.dot(s1, v1), self.s1.dot(self.v1_data))
        self.assertArrayEqual(mls.dot(s2, v1), self.s2.dot(self.v1_data))
        self.assertArrayEqual(mls.dot(v2, s1), self.v2_data.dot(self.s1.A))
        self.assertArrayEqual(mls.dot(v2, s2), self.v2_data.dot(self.s2.A))
        self.assertArrayEqual(mls.dot(v1, v1), self.v1_data.dot(self.v1_data))
        self.assertArrayEqual(mls.dot(v2, v2), self.v2_data.dot(self.v2_data))

        self.assertArrayEqual(mls.dot(v2, s1, sparse=False), self.v2_data.dot(self.s1.A))
        self.assertArrayEqual(mls.dot(v1, v1, sparse=False), self.v1_data.dot(self.v1_data))

    def testSparseSum(self):
        s1 = SparseNDArray(self.s1)
        v1 = SparseNDArray(self.v1, shape=(3,))
        self.assertEqual(s1.sum(), self.s1.sum())
        np.testing.assert_array_equal(s1.sum(axis=1), np.asarray(self.s1.sum(axis=1)).reshape(2))
        np.testing.assert_array_equal(s1.sum(axis=0), np.asarray(self.s1.sum(axis=0)).reshape(3))
        np.testing.assert_array_equal(v1.sum(), np.asarray(self.v1_data.sum()))

    @unittest.skip
    def testSparseGetitem(self):
        s1 = SparseNDArray(self.s1)
        v1 = SparseVector(self.v1, shape=(3,))
        self.assertEqual(s1[0, 1], self.s1[0, 1])
        self.assertEqual(v1[1], self.v1_data[1])

    def testSparseSetitem(self):
        s1 = SparseNDArray(self.s1.copy())
        s1[1:2, 1] = [2]
        ss1 = self.s1.tolil()
        ss1[1:2, 1] = [2]
        np.testing.assert_array_equal(s1.toarray(), ss1.toarray())

        v1 = SparseVector(self.v1, shape=(3,))
        v1[1:2] = [2]
        vv1 = self.v1_data
        vv1[1:2] = [2]
        np.testing.assert_array_equal(v1.toarray(), vv1)

    def testSparseMaximum(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        np.testing.assert_array_equal(s1.maximum(s2).toarray(), self.s1.maximum(self.s2).toarray())

        v1 = SparseVector(self.v1, shape=(3,))
        np.testing.assert_array_equal(v1.maximum(self.d1), np.maximum(self.v1_data, self.d1))

    def testSparseMinimum(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        np.testing.assert_array_equal(s1.minimum(s2).toarray(), self.s1.minimum(self.s2).toarray())

        v1 = SparseVector(self.v1, shape=(3,))
        np.testing.assert_array_equal(v1.minimum(self.d1), np.minimum(self.v1_data, self.d1))

    def testSparseFillDiagonal(self):
        s1 = sps.random(100, 11, density=0.3, format='csr', random_state=0)

        # fill scalar
        arr = SparseNDArray(s1)
        arr.fill_diagonal(3)

        expected = s1.copy().A
        np.fill_diagonal(expected, 3)

        np.testing.assert_array_equal(arr.toarray(), expected)

        # fill scalar, wrap=True
        arr = SparseNDArray(s1)
        arr.fill_diagonal(3, wrap=True)

        expected = s1.copy().A
        np.fill_diagonal(expected, 3, wrap=True)

        np.testing.assert_array_equal(arr.toarray(), expected)

        # fill list
        arr = SparseNDArray(s1)
        arr.fill_diagonal([1, 2, 3])

        expected = s1.copy().A
        np.fill_diagonal(expected, [1, 2, 3])

        np.testing.assert_array_equal(arr.toarray(), expected)

        # fill list, wrap=True
        arr = SparseNDArray(s1)
        arr.fill_diagonal([1, 2, 3], wrap=True)

        expected = s1.copy().A
        np.fill_diagonal(expected, [1, 2, 3], wrap=True)

        np.testing.assert_array_equal(arr.toarray(), expected)

        # fill long list
        val = np.random.RandomState(0).rand(101)
        arr = SparseNDArray(s1)
        arr.fill_diagonal(val)

        expected = s1.copy().A
        np.fill_diagonal(expected, val)

        np.testing.assert_array_equal(arr.toarray(), expected)

        # fill long list, wrap=True
        val = np.random.RandomState(0).rand(101)
        arr = SparseNDArray(s1)
        arr.fill_diagonal(val, wrap=True)

        expected = s1.copy().A
        np.fill_diagonal(expected, val, wrap=True)

        np.testing.assert_array_equal(arr.toarray(), expected)

        # fill ndarray
        val = np.random.RandomState(0).rand(3, 4)
        arr = SparseNDArray(s1)
        arr.fill_diagonal(val)

        expected = s1.copy().A
        np.fill_diagonal(expected, val)

        np.testing.assert_array_equal(arr.toarray(), expected)

        # fill ndarray, wrap=True
        val = np.random.RandomState(0).rand(3, 4)
        arr = SparseNDArray(s1)
        arr.fill_diagonal(val, wrap=True)

        expected = s1.copy().A
        np.fill_diagonal(expected, val, wrap=True)

        np.testing.assert_array_equal(arr.toarray(), expected)
