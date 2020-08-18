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

from mars.config import option_context
from mars.utils import ignore_warning
from mars.tensor.datasource import ones, tensor, zeros
from mars.tensor.arithmetic import add, cos, truediv, frexp, \
    modf, clip, isclose
from mars.tests.core import require_cupy, ExecutorForTest


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = ExecutorForTest('numpy')

    def _nan_equal(self, a, b):
        try:
            np.testing.assert_equal(a, b)
        except AssertionError:
            return False
        return True

    def testBaseExecution(self):
        arr = ones((10, 8), chunk_size=2)
        arr2 = arr + 1

        res = self.executor.execute_tensor(arr2)

        self.assertTrue((res[0] == np.ones((2, 2)) + 1).all())

        data = np.random.random((10, 8, 3))
        arr = tensor(data, chunk_size=2)
        arr2 = arr + 1

        res = self.executor.execute_tensor(arr2)

        self.assertTrue((res[0] == data[:2, :2, :2] + 1).all())

    def testBaseOrderExecution(self):
        raw = np.asfortranarray(np.random.rand(5, 6))
        arr = tensor(raw, chunk_size=3)

        res = self.executor.execute_tensor(arr + 1, concat=True)[0]
        np.testing.assert_array_equal(res, raw + 1)
        self.assertFalse(res.flags['C_CONTIGUOUS'])
        self.assertTrue(res.flags['F_CONTIGUOUS'])

        res2 = self.executor.execute_tensor(add(arr, 1, order='C'), concat=True)[0]
        np.testing.assert_array_equal(res2, np.add(raw, 1, order='C'))
        self.assertTrue(res2.flags['C_CONTIGUOUS'])
        self.assertFalse(res2.flags['F_CONTIGUOUS'])

    @staticmethod
    def _get_func(op):
        if isinstance(op, str):
            return getattr(np, op)
        return op

    def testUfuncExecution(self):
        from mars.tensor.arithmetic import UNARY_UFUNC, BIN_UFUNC, arccosh, \
            invert, mod, fmod, bitand, bitor, bitxor, lshift, rshift, ldexp

        _sp_unary_ufunc = {arccosh, invert}
        _sp_bin_ufunc = {mod, fmod, bitand, bitor, bitxor, lshift, rshift, ldexp}

        data1 = np.random.random((5, 9, 4))
        data2 = np.random.random((5, 9, 4))
        rand = np.random.random()
        arr1 = tensor(data1, chunk_size=3)
        arr2 = tensor(data2, chunk_size=3)

        _new_unary_ufunc = UNARY_UFUNC - _sp_unary_ufunc
        for func in _new_unary_ufunc:
            res_tensor = func(arr1)
            res = self.executor.execute_tensor(res_tensor, concat=True)
            expected = self._get_func(res_tensor.op._func_name)(data1)
            self.assertTrue(np.allclose(res[0], expected))

        _new_bin_ufunc = BIN_UFUNC - _sp_bin_ufunc
        for func in _new_bin_ufunc:
            res_tensor1 = func(arr1, arr2)
            res_tensor2 = func(arr1, rand)
            res_tensor3 = func(rand, arr1)

            res1 = self.executor.execute_tensor(res_tensor1, concat=True)
            res2 = self.executor.execute_tensor(res_tensor2, concat=True)
            res3 = self.executor.execute_tensor(res_tensor3, concat=True)

            expected1 = self._get_func(res_tensor1.op._func_name)(data1, data2)
            expected2 = self._get_func(res_tensor1.op._func_name)(data1, rand)
            expected3 = self._get_func(res_tensor1.op._func_name)(rand, data1)

            self.assertTrue(np.allclose(res1[0], expected1))
            self.assertTrue(np.allclose(res2[0], expected2))
            self.assertTrue(np.allclose(res3[0], expected3))

        data1 = np.random.randint(2, 10, size=(10, 10, 10))
        data2 = np.random.randint(2, 10, size=(10, 10, 10))
        rand = np.random.randint(1, 10)
        arr1 = tensor(data1, chunk_size=6)
        arr2 = tensor(data2, chunk_size=6)

        for func in _sp_unary_ufunc:
            res_tensor = func(arr1)
            res = self.executor.execute_tensor(res_tensor, concat=True)
            expected = self._get_func(res_tensor.op._func_name)(data1)
            self.assertTrue(np.allclose(res[0], expected))

        for func in _sp_bin_ufunc:
            res_tensor1 = func(arr1, arr2)
            res_tensor2 = func(arr1, rand)
            res_tensor3 = func(rand, arr1)

            res1 = self.executor.execute_tensor(res_tensor1, concat=True)
            res2 = self.executor.execute_tensor(res_tensor2, concat=True)
            res3 = self.executor.execute_tensor(res_tensor3, concat=True)

            expected1 = self._get_func(res_tensor1.op._func_name)(data1, data2)
            expected2 = self._get_func(res_tensor1.op._func_name)(data1, rand)
            expected3 = self._get_func(res_tensor1.op._func_name)(rand, data1)

            self.assertTrue(np.allclose(res1[0], expected1))
            self.assertTrue(np.allclose(res2[0], expected2))
            self.assertTrue(np.allclose(res3[0], expected3))

    @staticmethod
    def _get_sparse_func(op):
        from mars.lib.sparse.core import issparse

        if isinstance(op, str):
            op = getattr(np, op)

        def func(*args):
            new_args = []
            for arg in args:
                if issparse(arg):
                    new_args.append(arg.toarray())
                else:
                    new_args.append(arg)

            return op(*new_args)

        return func

    @staticmethod
    def toarray(x):
        if hasattr(x, 'toarray'):
            return x.toarray()
        return x

    @ignore_warning
    def testSparseUfuncExexution(self):
        from mars.tensor.arithmetic import UNARY_UFUNC, BIN_UFUNC, arccosh, \
            invert, mod, fmod, bitand, bitor, bitxor, lshift, rshift, ldexp

        _sp_unary_ufunc = {arccosh, invert}
        _sp_bin_ufunc = {mod, fmod, bitand, bitor, bitxor, lshift, rshift, ldexp}

        data1 = sps.random(5, 9, density=.1)
        data2 = sps.random(5, 9, density=.2)
        rand = np.random.random()
        arr1 = tensor(data1, chunk_size=3)
        arr2 = tensor(data2, chunk_size=3)

        _new_unary_ufunc = UNARY_UFUNC - _sp_unary_ufunc
        for func in _new_unary_ufunc:
            res_tensor = func(arr1)
            res = self.executor.execute_tensor(res_tensor, concat=True)
            expected = self._get_sparse_func(res_tensor.op._func_name)(data1)
            self._nan_equal(self.toarray(res[0]), expected)

        _new_bin_ufunc = BIN_UFUNC - _sp_bin_ufunc
        for func in _new_bin_ufunc:
            res_tensor1 = func(arr1, arr2)
            res_tensor2 = func(arr1, rand)
            res_tensor3 = func(rand, arr1)

            res1 = self.executor.execute_tensor(res_tensor1, concat=True)
            res2 = self.executor.execute_tensor(res_tensor2, concat=True)
            res3 = self.executor.execute_tensor(res_tensor3, concat=True)

            expected1 = self._get_sparse_func(res_tensor1.op._func_name)(data1, data2)
            expected2 = self._get_sparse_func(res_tensor1.op._func_name)(data1, rand)
            expected3 = self._get_sparse_func(res_tensor1.op._func_name)(rand, data1)

            self._nan_equal(self.toarray(res1[0]), expected1)
            self._nan_equal(self.toarray(res2[0]), expected2)
            self._nan_equal(self.toarray(res3[0]), expected3)

        data1 = np.random.randint(2, 10, size=(10, 10))
        data2 = np.random.randint(2, 10, size=(10, 10))
        rand = np.random.randint(1, 10)
        arr1 = tensor(data1, chunk_size=3).tosparse()
        arr2 = tensor(data2, chunk_size=3).tosparse()

        for func in _sp_unary_ufunc:
            res_tensor = func(arr1)
            res = self.executor.execute_tensor(res_tensor, concat=True)
            expected = self._get_sparse_func(res_tensor.op._func_name)(data1)
            self._nan_equal(self.toarray(res[0]), expected)

        for func in _sp_bin_ufunc:
            res_tensor1 = func(arr1, arr2)
            res_tensor2 = func(arr1, rand)
            res_tensor3 = func(rand, arr1)

            res1 = self.executor.execute_tensor(res_tensor1, concat=True)
            res2 = self.executor.execute_tensor(res_tensor2, concat=True)
            res3 = self.executor.execute_tensor(res_tensor3, concat=True)

            expected1 = self._get_sparse_func(res_tensor1.op._func_name)(data1, data2)
            expected2 = self._get_sparse_func(res_tensor1.op._func_name)(data1, rand)
            expected3 = self._get_sparse_func(res_tensor1.op._func_name)(rand, data1)

            self._nan_equal(self.toarray(res1[0]), expected1)
            self._nan_equal(self.toarray(res2[0]), expected2)
            self._nan_equal(self.toarray(res3[0]), expected3)

    def testAddWithOutExecution(self):
        data1 = np.random.random((5, 9, 4))
        data2 = np.random.random((9, 4))

        arr1 = tensor(data1.copy(), chunk_size=3)
        arr2 = tensor(data2.copy(), chunk_size=3)

        add(arr1, arr2, out=arr1)
        res = self.executor.execute_tensor(arr1, concat=True)[0]
        self.assertTrue(np.array_equal(res, data1 + data2))

        arr1 = tensor(data1.copy(), chunk_size=3)
        arr2 = tensor(data2.copy(), chunk_size=3)

        arr3 = add(arr1, arr2, out=arr1.astype('i4'), casting='unsafe')
        res = self.executor.execute_tensor(arr3, concat=True)[0]
        np.testing.assert_array_equal(res, (data1 + data2).astype('i4'))

        arr1 = tensor(data1.copy(), chunk_size=3)
        arr2 = tensor(data2.copy(), chunk_size=3)

        arr3 = truediv(arr1, arr2, out=arr1, where=arr2 > .5)
        res = self.executor.execute_tensor(arr3, concat=True)[0]
        self.assertTrue(np.array_equal(
            res, np.true_divide(data1, data2, out=data1.copy(), where=data2 > .5)))

        arr1 = tensor(data1.copy(), chunk_size=4)
        arr2 = tensor(data2.copy(), chunk_size=4)

        arr3 = add(arr1, arr2, where=arr1 > .5)
        res = self.executor.execute_tensor(arr3, concat=True)[0]
        expected = np.add(data1, data2, where=data1 > .5)
        self.assertTrue(np.array_equal(res[data1 > .5], expected[data1 > .5]))

        arr1 = tensor(data1.copy(), chunk_size=4)

        arr3 = add(arr1, 1, where=arr1 > .5)
        res = self.executor.execute_tensor(arr3, concat=True)[0]
        expected = np.add(data1, 1, where=data1 > .5)
        self.assertTrue(np.array_equal(res[data1 > .5], expected[data1 > .5]))

        arr1 = tensor(data2.copy(), chunk_size=3)

        arr3 = add(arr1[:5, :], 1, out=arr1[-5:, :])
        res = self.executor.execute_tensor(arr3, concat=True)[0]
        expected = np.add(data2[:5, :], 1)
        self.assertTrue(np.array_equal(res, expected))

    def testFrexpExecution(self):
        data1 = np.random.random((5, 9, 4))

        arr1 = tensor(data1.copy(), chunk_size=3)

        o1, o2 = frexp(arr1)
        o = o1 + o2

        res = self.executor.execute_tensor(o, concat=True)[0]
        expected = sum(np.frexp(data1))
        self.assertTrue(np.allclose(res, expected))

        arr1 = tensor(data1.copy(), chunk_size=3)
        o1 = zeros(data1.shape, chunk_size=3)
        o2 = zeros(data1.shape, dtype='i8', chunk_size=3)
        frexp(arr1, o1, o2)
        o = o1 + o2

        res = self.executor.execute_tensor(o, concat=True)[0]
        expected = sum(np.frexp(data1))
        self.assertTrue(np.allclose(res, expected))

        data1 = sps.random(5, 9, density=.1)

        arr1 = tensor(data1.copy(), chunk_size=3)

        o1, o2 = frexp(arr1)
        o = o1 + o2

        res = self.executor.execute_tensor(o, concat=True)[0]
        expected = sum(np.frexp(data1.toarray()))
        np.testing.assert_equal(res.toarray(), expected)

    def testFrexpOrderExecution(self):
        data1 = np.random.random((5, 9))
        t = tensor(data1, chunk_size=3)

        o1, o2 = frexp(t, order='F')
        res1, res2 = self.executor.execute_tileables([o1, o2])
        expected1, expected2 = np.frexp(data1, order='F')
        np.testing.assert_allclose(res1, expected1)
        self.assertTrue(res1.flags['F_CONTIGUOUS'])
        self.assertFalse(res1.flags['C_CONTIGUOUS'])
        np.testing.assert_allclose(res2, expected2)
        self.assertTrue(res2.flags['F_CONTIGUOUS'])
        self.assertFalse(res2.flags['C_CONTIGUOUS'])

    def testModfExecution(self):
        data1 = np.random.random((5, 9))

        arr1 = tensor(data1.copy(), chunk_size=3)

        o1, o2 = modf(arr1)
        o = o1 + o2

        res = self.executor.execute_tensor(o, concat=True)[0]
        expected = sum(np.modf(data1))
        self.assertTrue(np.allclose(res, expected))

        o1, o2 = modf([0, 3.5])
        o = o1 + o2

        res = self.executor.execute_tensor(o, concat=True)[0]
        expected = sum(np.modf([0, 3.5]))
        self.assertTrue(np.allclose(res, expected))

        arr1 = tensor(data1.copy(), chunk_size=3)
        o1 = zeros(data1.shape, chunk_size=3)
        o2 = zeros(data1.shape, chunk_size=3)
        modf(arr1, o1, o2)
        o = o1 + o2

        res = self.executor.execute_tensor(o, concat=True)[0]
        expected = sum(np.modf(data1))
        self.assertTrue(np.allclose(res, expected))

        data1 = sps.random(5, 9, density=.1)

        arr1 = tensor(data1.copy(), chunk_size=3)

        o1, o2 = modf(arr1)
        o = o1 + o2

        res = self.executor.execute_tensor(o, concat=True)[0]
        expected = sum(np.modf(data1.toarray()))
        np.testing.assert_equal(res.toarray(), expected)

    def testModfOrderExecution(self):
        data1 = np.random.random((5, 9))
        t = tensor(data1, chunk_size=3)

        o1, o2 = modf(t, order='F')
        res1, res2 = self.executor.execute_tileables([o1, o2])
        expected1, expected2 = np.modf(data1, order='F')
        np.testing.assert_allclose(res1, expected1)
        self.assertTrue(res1.flags['F_CONTIGUOUS'])
        self.assertFalse(res1.flags['C_CONTIGUOUS'])
        np.testing.assert_allclose(res2, expected2)
        self.assertTrue(res2.flags['F_CONTIGUOUS'])
        self.assertFalse(res2.flags['C_CONTIGUOUS'])

    def testClipExecution(self):
        a_data = np.arange(10)

        a = tensor(a_data.copy(), chunk_size=3)

        b = clip(a, 1, 8)

        res = self.executor.execute_tensor(b, concat=True)[0]
        expected = np.clip(a_data, 1, 8)
        self.assertTrue(np.array_equal(res, expected))

        a = tensor(a_data.copy(), chunk_size=3)
        clip(a, 3, 6, out=a)

        res = self.executor.execute_tensor(a, concat=True)[0]
        expected = np.clip(a_data, 3, 6)
        self.assertTrue(np.array_equal(res, expected))

        a = tensor(a_data.copy(), chunk_size=3)
        a_min_data = np.random.randint(1, 10, size=(10,))
        a_max_data = np.random.randint(1, 10, size=(10,))
        a_min = tensor(a_min_data)
        a_max = tensor(a_max_data)
        clip(a, a_min, a_max, out=a)

        res = self.executor.execute_tensor(a, concat=True)[0]
        expected = np.clip(a_data, a_min_data, a_max_data)
        self.assertTrue(np.array_equal(res, expected))

        with option_context() as options:
            options.chunk_size = 3

            a = tensor(a_data.copy(), chunk_size=3)
            b = clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)

            res = self.executor.execute_tensor(b, concat=True)[0]
            expected = np.clip(a_data, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)
            self.assertTrue(np.array_equal(res, expected))

            # test sparse clip
            a_data = sps.csr_matrix([[0, 2, 8], [0, 0, -1]])
            a = tensor(a_data, chunk_size=3)
            b_data = sps.csr_matrix([[0, 3, 0], [1, 0, -2]])

            c = clip(a, b_data, 4)

            res = self.executor.execute_tensor(c, concat=True)[0]
            expected = np.clip(a_data.toarray(), b_data.toarray(), 4)
            self.assertTrue(np.array_equal(res.toarray(), expected))

    def testClipOrderExecution(self):
        a_data = np.asfortranarray(np.random.rand(4, 8))

        a = tensor(a_data, chunk_size=3)

        b = clip(a, 0.2, 0.8)

        res = self.executor.execute_tensor(b, concat=True)[0]
        expected = np.clip(a_data, 0.2, 0.8)

        np.testing.assert_allclose(res, expected)
        self.assertTrue(res.flags['F_CONTIGUOUS'])
        self.assertFalse(res.flags['C_CONTIGUOUS'])

    def testAroundExecution(self):
        data = np.random.randn(10, 20)
        x = tensor(data, chunk_size=3)

        t = x.round(2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.around(data, decimals=2)

        np.testing.assert_allclose(res, expected)

        data = sps.random(10, 20, density=.2)
        x = tensor(data, chunk_size=3)

        t = x.round(2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.around(data.toarray(), decimals=2)

        np.testing.assert_allclose(res.toarray(), expected)

    def testAroundOrderExecution(self):
        data = np.asfortranarray(np.random.rand(10, 20))
        x = tensor(data, chunk_size=3)

        t = x.round(2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.around(data, decimals=2)

        np.testing.assert_allclose(res, expected)
        self.assertTrue(res.flags['F_CONTIGUOUS'])
        self.assertFalse(res.flags['C_CONTIGUOUS'])

    def testCosOrderExecution(self):
        data = np.asfortranarray(np.random.rand(3, 5))
        x = tensor(data, chunk_size=2)

        t = cos(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        np.testing.assert_allclose(res, np.cos(data))
        self.assertFalse(res.flags['C_CONTIGUOUS'])
        self.assertTrue(res.flags['F_CONTIGUOUS'])

        t2 = cos(x, order='C')

        res2 = self.executor.execute_tensor(t2, concat=True)[0]
        np.testing.assert_allclose(res2, np.cos(data, order='C'))
        self.assertTrue(res2.flags['C_CONTIGUOUS'])
        self.assertFalse(res2.flags['F_CONTIGUOUS'])

    def testIsCloseExecution(self):
        data = np.array([1.05, 1.0, 1.01, np.nan])
        data2 = np.array([1.04, 1.0, 1.03, np.nan])

        x = tensor(data, chunk_size=2)
        y = tensor(data2, chunk_size=3)

        z = isclose(x, y, atol=.01)

        res = self.executor.execute_tensor(z, concat=True)[0]
        expected = np.isclose(data, data2, atol=.01)
        np.testing.assert_equal(res, expected)

        z = isclose(x, y, atol=.01, equal_nan=True)

        res = self.executor.execute_tensor(z, concat=True)[0]
        expected = np.isclose(data, data2, atol=.01, equal_nan=True)
        np.testing.assert_equal(res, expected)

        # test tensor with scalar
        z = isclose(x, 1.0, atol=.01)
        res = self.executor.execute_tensor(z, concat=True)[0]
        expected = np.isclose(data, 1.0, atol=.01)
        np.testing.assert_equal(res, expected)
        z = isclose(1.0, y, atol=.01)
        res = self.executor.execute_tensor(z, concat=True)[0]
        expected = np.isclose(1.0, data2, atol=.01)
        np.testing.assert_equal(res, expected)
        z = isclose(1.0, 2.0, atol=.01)
        res = self.executor.execute_tensor(z, concat=True)[0]
        expected = np.isclose(1.0, 2.0, atol=.01)
        np.testing.assert_equal(res, expected)

        # test sparse
        data = sps.csr_matrix(np.array([0, 1.0, 1.01, np.nan]))
        data2 = sps.csr_matrix(np.array([0, 1.0, 1.03, np.nan]))

        x = tensor(data, chunk_size=2)
        y = tensor(data2, chunk_size=3)

        z = isclose(x, y, atol=.01)

        res = self.executor.execute_tensor(z, concat=True)[0]
        expected = np.isclose(data.toarray(), data2.toarray(), atol=.01)
        np.testing.assert_equal(res, expected)

        z = isclose(x, y, atol=.01, equal_nan=True)

        res = self.executor.execute_tensor(z, concat=True)[0]
        expected = np.isclose(data.toarray(), data2.toarray(), atol=.01, equal_nan=True)
        np.testing.assert_equal(res, expected)

    @ignore_warning
    def testDtypeExecution(self):
        a = ones((10, 20), dtype='f4', chunk_size=5)

        c = truediv(a, 2, dtype='f8')

        res = self.executor.execute_tensor(c, concat=True)[0]
        self.assertEqual(res.dtype, np.float64)

        c = truediv(a, 0, dtype='f8')
        res = self.executor.execute_tensor(c, concat=True)[0]
        self.assertTrue(np.isinf(res[0, 0]))

        with self.assertRaises(FloatingPointError):
            with np.errstate(divide='raise'):
                c = truediv(a, 0, dtype='f8')
                _ = self.executor.execute_tensor(c, concat=True)[0]  # noqa: F841

    def testSetGetRealExecution(self):
        a_data = np.array([1+2j, 3+4j, 5+6j])
        a = tensor(a_data, chunk_size=2)

        res = self.executor.execute_tensor(a.real, concat=True)[0]
        expected = a_data.real

        np.testing.assert_equal(res, expected)

        a.real = 9

        res = self.executor.execute_tensor(a, concat=True)[0]
        expected = a_data.copy()
        expected.real = 9

        np.testing.assert_equal(res, expected)

        a.real = np.array([9, 8, 7])

        res = self.executor.execute_tensor(a, concat=True)[0]
        expected = a_data.copy()
        expected.real = np.array([9, 8, 7])

        np.testing.assert_equal(res, expected)

        # test sparse
        a_data = np.array([[1+2j, 3+4j, 0], [0, 0, 0]])
        a = tensor(sps.csr_matrix(a_data))

        res = self.executor.execute_tensor(a.real, concat=True)[0].toarray()
        expected = a_data.real

        np.testing.assert_equal(res, expected)

        a.real = 9

        res = self.executor.execute_tensor(a, concat=True)[0].toarray()
        expected = a_data.copy()
        expected.real = 9

        np.testing.assert_equal(res, expected)

        a.real = np.array([9, 8, 7])

        res = self.executor.execute_tensor(a, concat=True)[0].toarray()
        expected = a_data.copy()
        expected.real = np.array([9, 8, 7])

        np.testing.assert_equal(res, expected)

    def testSetGetImagExecution(self):
        a_data = np.array([1+2j, 3+4j, 5+6j])
        a = tensor(a_data, chunk_size=2)

        res = self.executor.execute_tensor(a.imag, concat=True)[0]
        expected = a_data.imag

        np.testing.assert_equal(res, expected)

        a.imag = 9

        res = self.executor.execute_tensor(a, concat=True)[0]
        expected = a_data.copy()
        expected.imag = 9

        np.testing.assert_equal(res, expected)

        a.imag = np.array([9, 8, 7])

        res = self.executor.execute_tensor(a, concat=True)[0]
        expected = a_data.copy()
        expected.imag = np.array([9, 8, 7])

        np.testing.assert_equal(res, expected)

        # test sparse
        a_data = np.array([[1+2j, 3+4j, 0], [0, 0, 0]])
        a = tensor(sps.csr_matrix(a_data))

        res = self.executor.execute_tensor(a.imag, concat=True)[0].toarray()
        expected = a_data.imag

        np.testing.assert_equal(res, expected)

        a.imag = 9

        res = self.executor.execute_tensor(a, concat=True)[0].toarray()
        expected = a_data.copy()
        expected.imag = 9

        np.testing.assert_equal(res, expected)

        a.imag = np.array([9, 8, 7])

        res = self.executor.execute_tensor(a, concat=True)[0].toarray()
        expected = a_data.copy()
        expected.imag = np.array([9, 8, 7])

        np.testing.assert_equal(res, expected)

    @require_cupy
    def testCupyExecution(self):
        a_data = np.random.rand(10, 10)
        b_data = np.random.rand(10, 10)

        a = tensor(a_data, gpu=True, chunk_size=3)
        b = tensor(b_data, gpu=True, chunk_size=3)
        res_binary = self.executor.execute_tensor((a + b), concat=True)[0]
        np.testing.assert_array_equal(res_binary.get(), (a_data + b_data))

        res_unary = self.executor.execute_tensor(cos(a), concat=True)[0]
        np.testing.assert_array_almost_equal(res_unary.get(), np.cos(a_data))
