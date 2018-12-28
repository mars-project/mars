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

from mars.compat import six
from mars.tensor.execution.core import Executor
from mars.tensor.expressions.datasource import ones, tensor, zeros
from mars.tensor.expressions.arithmetic import add, truediv, frexp, \
    modf, clip, isclose
from mars.config import option_context


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')

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

    @staticmethod
    def _get_func(op):
        if isinstance(op, six.string_types):
            return getattr(np, op)
        return op

    def testUfuncExecution(self):
        from mars.tensor.expressions.arithmetic import UNARY_UFUNC, BIN_UFUNC, arccosh, \
            invert, mod, fmod, bitand, bitor, bitxor, lshift, rshift, ldexp
        from mars.tensor.execution.arithmetic import OP_TO_HANDLER

        _sp_unary_ufunc = set([arccosh, invert])
        _sp_bin_ufunc = set([mod, fmod, bitand, bitor, bitxor, lshift, rshift, ldexp])

        data1 = np.random.random((5, 9, 4))
        data2 = np.random.random((5, 9, 4))
        rand = np.random.random()
        arr1 = tensor(data1, chunk_size=3)
        arr2 = tensor(data2, chunk_size=3)

        _new_unary_ufunc = UNARY_UFUNC - _sp_unary_ufunc
        for func in _new_unary_ufunc:
            res_tensor = func(arr1)
            res = self.executor.execute_tensor(res_tensor, concat=True)
            expected = self._get_func(OP_TO_HANDLER[type(res_tensor.op)])(data1)
            self.assertTrue(np.allclose(res[0], expected))

        _new_bin_ufunc = BIN_UFUNC - _sp_bin_ufunc
        for func in _new_bin_ufunc:
            res_tensor1 = func(arr1, arr2)
            res_tensor2 = func(arr1, rand)
            res_tensor3 = func(rand, arr1)

            res1 = self.executor.execute_tensor(res_tensor1, concat=True)
            res2 = self.executor.execute_tensor(res_tensor2, concat=True)
            res3 = self.executor.execute_tensor(res_tensor3, concat=True)

            expected1 = self._get_func(OP_TO_HANDLER[type(res_tensor1.op)])(data1, data2)
            expected2 = self._get_func(OP_TO_HANDLER[type(res_tensor1.op)])(data1, rand)
            expected3 = self._get_func(OP_TO_HANDLER[type(res_tensor1.op)])(rand, data1)

            self.assertTrue(np.allclose(res1[0], expected1))
            self.assertTrue(np.allclose(res2[0], expected2))
            self.assertTrue(np.allclose(res3[0], expected3))

        data1 = np.random.randint(2, 10, size=(10, 10, 10))
        data2 = np.random.randint(2, 10, size=(10, 10, 10))
        rand = np.random.randint(1, 10)
        arr1 = tensor(data1, chunk_size=3)
        arr2 = tensor(data2, chunk_size=3)

        for func in _sp_unary_ufunc:
            res_tensor = func(arr1)
            res = self.executor.execute_tensor(res_tensor, concat=True)
            expected = self._get_func(OP_TO_HANDLER[type(res_tensor.op)])(data1)
            self.assertTrue(np.allclose(res[0], expected))

        for func in _sp_bin_ufunc:
            res_tensor1 = func(arr1, arr2)
            res_tensor2 = func(arr1, rand)
            res_tensor3 = func(rand, arr1)

            res1 = self.executor.execute_tensor(res_tensor1, concat=True)
            res2 = self.executor.execute_tensor(res_tensor2, concat=True)
            res3 = self.executor.execute_tensor(res_tensor3, concat=True)

            expected1 = self._get_func(OP_TO_HANDLER[type(res_tensor1.op)])(data1, data2)
            expected2 = self._get_func(OP_TO_HANDLER[type(res_tensor1.op)])(data1, rand)
            expected3 = self._get_func(OP_TO_HANDLER[type(res_tensor1.op)])(rand, data1)

            self.assertTrue(np.allclose(res1[0], expected1))
            self.assertTrue(np.allclose(res2[0], expected2))
            self.assertTrue(np.allclose(res3[0], expected3))

    @staticmethod
    def _get_sparse_func(op):
        from mars.lib.sparse.core import issparse

        if isinstance(op, six.string_types):
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

    def testSparseUfuncExexution(self):
        from mars.tensor.expressions.arithmetic import UNARY_UFUNC, BIN_UFUNC, arccosh, \
            invert, mod, fmod, bitand, bitor, bitxor, lshift, rshift, ldexp
        from mars.tensor.execution.arithmetic import OP_TO_HANDLER

        _sp_unary_ufunc = set([arccosh, invert])
        _sp_bin_ufunc = set([mod, fmod, bitand, bitor, bitxor, lshift, rshift, ldexp])

        data1 = sps.random(5, 9, density=.1)
        data2 = sps.random(5, 9, density=.2)
        rand = np.random.random()
        arr1 = tensor(data1, chunk_size=3)
        arr2 = tensor(data2, chunk_size=3)

        _new_unary_ufunc = UNARY_UFUNC - _sp_unary_ufunc
        for func in _new_unary_ufunc:
            res_tensor = func(arr1)
            res = self.executor.execute_tensor(res_tensor, concat=True)
            expected = self._get_sparse_func(OP_TO_HANDLER[type(res_tensor.op)])(data1)
            self._nan_equal(self.toarray(res[0]), expected)

        _new_bin_ufunc = BIN_UFUNC - _sp_bin_ufunc
        for func in _new_bin_ufunc:
            res_tensor1 = func(arr1, arr2)
            res_tensor2 = func(arr1, rand)
            res_tensor3 = func(rand, arr1)

            res1 = self.executor.execute_tensor(res_tensor1, concat=True)
            res2 = self.executor.execute_tensor(res_tensor2, concat=True)
            res3 = self.executor.execute_tensor(res_tensor3, concat=True)

            expected1 = self._get_sparse_func(OP_TO_HANDLER[type(res_tensor1.op)])(data1, data2)
            expected2 = self._get_sparse_func(OP_TO_HANDLER[type(res_tensor1.op)])(data1, rand)
            expected3 = self._get_sparse_func(OP_TO_HANDLER[type(res_tensor1.op)])(rand, data1)

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
            expected = self._get_sparse_func(OP_TO_HANDLER[type(res_tensor.op)])(data1)
            self._nan_equal(self.toarray(res[0]), expected)

        for func in _sp_bin_ufunc:
            res_tensor1 = func(arr1, arr2)
            res_tensor2 = func(arr1, rand)
            res_tensor3 = func(rand, arr1)

            res1 = self.executor.execute_tensor(res_tensor1, concat=True)
            res2 = self.executor.execute_tensor(res_tensor2, concat=True)
            res3 = self.executor.execute_tensor(res_tensor3, concat=True)

            expected1 = self._get_sparse_func(OP_TO_HANDLER[type(res_tensor1.op)])(data1, data2)
            expected2 = self._get_sparse_func(OP_TO_HANDLER[type(res_tensor1.op)])(data1, rand)
            expected3 = self._get_sparse_func(OP_TO_HANDLER[type(res_tensor1.op)])(rand, data1)

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

    def testModfExecution(self):
        data1 = np.random.random((5, 9))

        arr1 = tensor(data1.copy(), chunk_size=3)

        o1, o2 = modf(arr1)
        o = o1 + o2

        res = self.executor.execute_tensor(o, concat=True)[0]
        expected = sum(np.modf(data1))
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

        with option_context() as options:
            options.tensor.chunk_size = 3

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
                _ = self.executor.execute_tensor(c, concat=True)[0]

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
