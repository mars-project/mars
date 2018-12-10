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

from mars.tests.core import TestBase
from mars.lib.sparse import SparseNDArray
from mars.lib.sparse.core import issparse
import mars.lib.sparse as mls


class Test(TestBase):
    def setUp(self):
        self.s1 = sps.csr_matrix([[1, 0, 1], [0, 0, 1]])
        self.s2 = sps.csr_matrix([[0, 1, 1], [1, 0, 1]])
        self.d1 = np.array([1, 2, 3])

    def testSparseCreation(self):
        s = SparseNDArray(self.s1)
        self.assertEqual(s.ndim, 2)

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
        self.assertArrayEqual(s1 + 1, self.s1.toarray() + 1)
        self.assertArrayEqual(1 + s1, self.s1.toarray() + 1)

    def testSparseSubtract(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        self.assertArrayEqual(s1 - s2, self.s1 - self.s2)
        self.assertArrayEqual(s1 - self.d1, self.s1 - self.d1)
        self.assertArrayEqual(self.d1 - s1, self.d1 - self.s1)
        self.assertArrayEqual(s1 - 1, self.s1.toarray() - 1)
        self.assertArrayEqual(1 - s1, 1 - self.s1.toarray())

    def testSparseMultiply(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        self.assertArrayEqual(s1 * s2, self.s1.multiply(self.s2))
        self.assertArrayEqual(s1 * self.d1, self.s1.multiply(self.d1))
        self.assertArrayEqual(self.d1 * s1, self.s1.multiply(self.d1))
        self.assertArrayEqual(s1 * 2, self.s1 * 2)
        self.assertArrayEqual(2 * s1, self.s1 * 2)

    def testSparseDivide(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        self.assertArrayEqual(s1 / s2, self.s1 / self.s2)
        self.assertArrayEqual(s1 / self.d1, self.s1 / self.d1)
        self.assertArrayEqual(self.d1 / s1, self.d1 / self.s1.toarray())
        self.assertArrayEqual(s1 / 2, self.s1 / 2)
        self.assertArrayEqual(2 / s1, 2 / self.s1.toarray())

    def testSparseFloorDivide(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        self.assertArrayEqual(s1 // s2, self.s1.toarray() // self.s2.toarray())
        self.assertArrayEqual(s1 // self.d1, self.s1.toarray() // self.d1)
        self.assertArrayEqual(self.d1 // s1, self.d1 // self.s1.toarray())
        self.assertArrayEqual(s1 // 2, self.s1.toarray() // 2)
        self.assertArrayEqual(2 // s1, 2 // self.s1.toarray())

    def testSparsePower(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        self.assertArrayEqual(s1 ** s2, self.s1.toarray() ** self.s2.toarray())
        self.assertArrayEqual(s1 ** self.d1, self.s1.toarray() ** self.d1)
        self.assertArrayEqual(self.d1 ** s1, self.d1 ** self.s1.toarray())
        self.assertArrayEqual(s1 ** 2, self.s1.power(2))
        self.assertArrayEqual(2 ** s1, 2 ** self.s1.toarray())

    def testSparseMod(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        self.assertArrayEqual(s1 % s2, self.s1.toarray() % self.s2.toarray())
        self.assertArrayEqual(s1 % self.d1, self.s1.toarray() % self.d1)
        self.assertArrayEqual(self.d1 % s1, self.d1 % self.s1.toarray())
        self.assertArrayEqual(s1 % 2, self.s1.toarray() % 2)
        self.assertArrayEqual(2 % s1, 2 % self.s1.toarray())

    def testSparseBin(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        for method in ('fmod', 'logaddexp', 'logaddexp2', 'equal', 'not_equal',
                       'less', 'less_equal', 'greater', 'greater_equal', 'hypot'):
            lm, rm = getattr(mls, method), getattr(np, method)
            self.assertArrayEqual(lm(s1, s2), rm(self.s1.toarray(), self.s2.toarray()))
            self.assertArrayEqual(lm(s1, self.d1), rm(self.s1.toarray(), self.d1))
            self.assertArrayEqual(lm(self.d1, s1), rm(self.d1, self.s1.toarray()))
            self.assertArrayEqual(lm(s1, 2), rm(self.s1.toarray(), 2))
            self.assertArrayEqual(lm(2, s1), rm(2, self.s1.toarray()))

    def testSparseUnary(self):
        s1 = SparseNDArray(self.s1)

        for method in ('negative', 'positive', 'absolute', 'abs', 'fabs', 'rint',
                       'sign', 'conj', 'exp', 'exp2', 'log', 'log2', 'log10',
                       'expm1', 'log1p', 'sqrt', 'square', 'cbrt', 'reciprocal',
                       'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
                       'arcsinh', 'arccosh', 'arctanh', 'deg2rad', 'rad2deg'):
            lm, rm = getattr(mls, method), getattr(np, method)
            self.assertArrayEqual(lm(s1), rm(self.s1.toarray()))

    def testSparseDot(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        self.assertArrayEqual(mls.dot(s1, s2.T), self.s1.dot(self.s2.T))
        self.assertArrayEqual(s1.dot(self.d1), self.s1.dot(self.d1))
        self.assertArrayEqual(self.d1.dot(s1.T), self.d1.dot(self.s1.T.toarray()))

        self.assertArrayEqual(mls.tensordot(s1, s2.T, axes=(1, 0)), self.s1.dot(self.s2.T))
        self.assertArrayEqual(mls.tensordot(s1, self.d1, axes=(1, -1)), self.s1.dot(self.d1))
        self.assertArrayEqual(mls.tensordot(self.d1, s1.T, axes=(0, 0)), self.d1.dot(self.s1.T.toarray()))

    def testSparseSum(self):
        s1 = SparseNDArray(self.s1)
        self.assertEqual(s1.sum(), self.s1.sum())
        np.testing.assert_array_equal(s1.sum(axis=1), np.asarray(self.s1.sum(axis=1)).reshape(2))

    @unittest.skip
    def testSparseGetitem(self):
        s1 = SparseNDArray(self.s1)
        self.assertEqual(s1[0, 1], self.s1[0, 1])

    def testSparseSetitem(self):
        s1 = SparseNDArray(self.s1.copy())
        s1[1:2, 1] = [2]
        ss1 = self.s1.tolil()
        ss1[1:2, 1] = [2]
        np.testing.assert_array_equal(s1.toarray(), ss1.toarray())

    def testSparseMaximum(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        np.testing.assert_array_equal(s1.maximum(s2).toarray(), self.s1.maximum(self.s2).toarray())

    def testSparseMinimum(self):
        s1 = SparseNDArray(self.s1)
        s2 = SparseNDArray(self.s2)

        np.testing.assert_array_equal(s1.minimum(s2).toarray(), self.s1.minimum(self.s2).toarray())
