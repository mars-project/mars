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

from mars import operands
from mars.tensor.expressions.datasource import array, ones, tensor, empty
from mars.tensor.expressions.datasource.core import TensorFetch
from mars.tensor.expressions.arithmetic import add, subtract, truediv, log, frexp, around, \
    isclose, isfinite, negative, cos
from mars.tensor.expressions.linalg import matmul
from mars.tensor.core import Tensor, SparseTensor
from mars.core import build_mode
from mars.operands import Add, AddConstant, SubConstant, Log, IscloseConstant, Isclose
from mars.tests.core import calc_shape


class Test(unittest.TestCase):
    def testAdd(self):
        t1 = ones((3, 4), chunk_size=2)
        t2 = ones(4, chunk_size=2)
        t3 = t1 + t2
        k1 = t3.key
        t3.tiles()
        self.assertNotEqual(t3.key, k1)
        self.assertEqual(t3.shape, (3, 4))
        self.assertEqual(len(t3.chunks), 4)
        self.assertEqual(calc_shape(t3), t3.shape)
        self.assertEqual(t3.chunks[0].inputs, [t1.chunks[0].data, t2.chunks[0].data])
        self.assertEqual(t3.chunks[1].inputs, [t1.chunks[1].data, t2.chunks[1].data])
        self.assertEqual(t3.chunks[2].inputs, [t1.chunks[2].data, t2.chunks[0].data])
        self.assertEqual(t3.chunks[3].inputs, [t1.chunks[3].data, t2.chunks[1].data])
        self.assertEqual(t3.op.dtype, np.dtype('f8'))
        self.assertEqual(t3.chunks[0].op.dtype, np.dtype('f8'))
        self.assertEqual(calc_shape(t3.chunks[0]), t3.chunks[0].shape)

        t4 = t1 + 1
        t4.tiles()
        self.assertEqual(t4.shape, (3, 4))
        self.assertEqual(len(t3.chunks), 4)
        self.assertEqual(calc_shape(t4), t4.shape)
        self.assertEqual(t4.chunks[0].inputs, [t1.chunks[0].data])
        self.assertEqual(t4.chunks[0].op.constant[0], 1)
        self.assertEqual(t4.chunks[1].inputs, [t1.chunks[1].data])
        self.assertEqual(t4.chunks[1].op.constant[0], 1)
        self.assertEqual(t4.chunks[2].inputs, [t1.chunks[2].data])
        self.assertEqual(t4.chunks[2].op.constant[0], 1)
        self.assertEqual(t4.chunks[3].inputs, [t1.chunks[3].data])
        self.assertEqual(t4.chunks[3].op.constant[0], 1)
        self.assertEqual(calc_shape(t4.chunks[0]), t4.chunks[0].shape)

        # sparse tests
        t5 = add([1, 2, 3, 4], 1)
        t5.tiles()
        self.assertEqual(t4.chunks[0].inputs, [t1.chunks[0].data])

        t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

        t = t1 + 1
        self.assertFalse(t.issparse())
        self.assertIs(type(t), Tensor)

        t.tiles()
        self.assertFalse(t.chunks[0].op.sparse)

        t = t1 + 0
        self.assertTrue(t.issparse())
        self.assertIs(type(t), SparseTensor)

        t2 = tensor([[1, 0, 0]], chunk_size=2).tosparse()

        t = t1 + t2
        self.assertTrue(t.issparse())
        self.assertIs(type(t), SparseTensor)

        t.tiles()
        self.assertTrue(t.chunks[0].op.sparse)

        t3 = tensor([1, 1, 1], chunk_size=2)
        t = t1 + t3
        self.assertFalse(t.issparse())
        self.assertIs(type(t), Tensor)

        t.tiles()
        self.assertFalse(t.chunks[0].op.sparse)

    def testMultiply(self):
        t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

        t = t1 * 10
        self.assertTrue(t.issparse())
        self.assertIs(type(t), SparseTensor)

        t.tiles()
        self.assertTrue(t.chunks[0].op.sparse)

        t2 = tensor([[1, 0, 0]], chunk_size=2).tosparse()

        t = t1 * t2
        self.assertTrue(t.issparse())
        self.assertIs(type(t), SparseTensor)

        t.tiles()
        self.assertTrue(t.chunks[0].op.sparse)

        t3 = tensor([1, 1, 1], chunk_size=2)
        t = t1 * t3
        self.assertTrue(t.issparse())
        self.assertIs(type(t), SparseTensor)

        t.tiles()
        self.assertTrue(t.chunks[0].op.sparse)

    def testDivide(self):
        t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

        t = t1 / 10
        self.assertTrue(t.issparse())
        self.assertIs(type(t), SparseTensor)

        t.tiles()
        self.assertTrue(t.chunks[0].op.sparse)

        t2 = tensor([[1, 0, 0]], chunk_size=2).tosparse()

        t = t1 / t2
        self.assertFalse(t.issparse())
        self.assertIs(type(t), Tensor)

        t.tiles()
        self.assertFalse(t.chunks[0].op.sparse)

        t3 = tensor([1, 1, 1], chunk_size=2)
        t = t1 / t3
        self.assertFalse(t.issparse())
        self.assertIs(type(t), Tensor)

        t.tiles()
        self.assertFalse(t.chunks[0].op.sparse)

        t = t3 / t1
        self.assertFalse(t.issparse())
        self.assertIs(type(t), Tensor)

        t.tiles()
        self.assertFalse(t.chunks[0].op.sparse)

    def testDatatimeArith(self):
        t1 = array([np.datetime64('2005-02-02'), np.datetime64('2005-02-03')])
        t2 = t1 + np.timedelta64(1)

        self.assertIsInstance(t2.op, AddConstant)

        t3 = t1 - np.datetime64('2005-02-02')

        self.assertIsInstance(t3.op, SubConstant)
        self.assertEqual(t3.dtype,
                         (np.array(['2005-02-02', '2005-02-03'], dtype=np.datetime64) -
                          np.datetime64('2005-02-02')).dtype)

        t1 = array([np.datetime64('2005-02-02'), np.datetime64('2005-02-03')])
        subtract(t1, np.datetime64('2005-02-02'), out=empty(t1.shape, dtype=t3.dtype))

        t1 = array([np.datetime64('2005-02-02'), np.datetime64('2005-02-03')])
        add(t1, np.timedelta64(1, 'D'), out=t1)

    def testAddWithOut(self):
        t1 = ones((3, 4), chunk_size=2)
        t2 = ones(4, chunk_size=2)

        t3 = add(t1, t2, out=t1)

        self.assertIsInstance(t1.op, Add)
        self.assertEqual(t1.op.out.key, t1.op.lhs.key)
        self.assertIs(t3, t1)
        self.assertEqual(t3.shape, (3, 4))
        self.assertEqual(t3.op.lhs.params.raw_chunk_size, 2)
        self.assertIs(t3.op.rhs, t2.data)
        self.assertNotEqual(t3.key, t3.op.lhs.key)
        self.assertEqual(calc_shape(t3), t3.shape)

        t3.tiles()

        self.assertIsInstance(t1.chunks[0].op, Add)
        self.assertEqual(t1.chunks[0].op.out.key, t1.chunks[0].op.lhs.key)
        self.assertEqual(calc_shape(t1.chunks[0]), t1.chunks[0].shape)

        with self.assertRaises(TypeError):
            add(t1, t2, out=1)

        with self.assertRaises(ValueError):
            add(t1, t2, out=t2)

        with self.assertRaises(TypeError):
            truediv(t1, t2, out=t1.astype('i8'))

        t1 = ones((3, 4), chunk_size=2, dtype=float)
        t2 = ones(4, chunk_size=2, dtype=int)

        t3 = add(t2, 1, out=t1)
        self.assertEqual(t3.shape, (3, 4))
        self.assertEqual(t3.dtype, np.float64)

    def testDtypeFromOut(self):
        x = array([-np.inf, 0., np.inf])
        y = array([2, 2, 2])

        t3 = isfinite(x, y)
        self.assertEqual(t3.dtype, y.dtype)

    def testLogWithOut(self):
        t1 = ones((3, 4), chunk_size=2)

        t2 = log(t1, out=t1)

        self.assertIsInstance(t2.op, Log)
        self.assertEqual(t1.op.out.key, t1.op.input.key)
        self.assertIs(t2, t1)
        self.assertEqual(t2.op.input.params.raw_chunk_size, 2)
        self.assertNotEqual(t2.key, t2.op.input.key)

    def testCopyAdd(self):
        t1 = ones((3, 4), chunk_size=2)
        t2 = ones(4, chunk_size=2)
        t3 = t1 + t2
        t3.tiles()

        c = t3.chunks[0]
        inputs = c.op.lhs, TensorFetch().new_chunk(
            c.op.rhs.inputs, c.op.rhs.shape, index=c.op.rhs.index, _key=c.op.rhs.key)
        new_c = c.op.copy().reset_key().new_chunk(inputs, c.shape, _key='new_key')
        self.assertEqual(new_c.key, 'new_key')
        self.assertIs(new_c.inputs[1], new_c.op.rhs)
        self.assertIsInstance(new_c.inputs[1].op, TensorFetch)

    def testCompare(self):
        t1 = ones(4, chunk_size=2) * 2
        t2 = ones(4, chunk_size=2)
        t3 = t1 > t2
        t3.tiles()
        self.assertEqual(len(t3.chunks), 2)
        self.assertIsInstance(t3.op, operands.GreaterThan)

    def testUnifyChunkAdd(self):
        t1 = ones(4, chunk_size=2)
        t2 = ones(1, chunk_size=1)

        t3 = t1 + t2
        t3.tiles()
        self.assertEqual(len(t3.chunks), 2)
        self.assertEqual(t3.chunks[0].inputs[0], t1.chunks[0].data)
        self.assertEqual(t3.chunks[0].inputs[1], t2.chunks[0].data)
        self.assertEqual(t3.chunks[1].inputs[0], t1.chunks[1].data)
        self.assertEqual(t3.chunks[1].inputs[1], t2.chunks[0].data)
        self.assertEqual(calc_shape(t3), t3.shape)

    def testTensordot(self):
        from mars.tensor.expressions.linalg import tensordot, dot, inner

        t1 = ones((3, 4, 6), chunk_size=2)
        t2 = ones((4, 3, 5), chunk_size=2)
        t3 = tensordot(t1, t2, axes=((0, 1), (1, 0)))

        self.assertEqual(t3.shape, (6, 5))
        self.assertEqual(calc_shape(t3), t3.shape)

        t3.tiles()

        self.assertEqual(t3.shape, (6, 5))
        self.assertEqual(calc_shape(t3.chunks[0]), t3.chunks[0].shape)
        self.assertEqual(len(t3.chunks), 9)

        a = ones((10000, 20000), chunk_size=5000)
        b = ones((20000, 1000), chunk_size=5000)

        with self.assertRaises(ValueError):
            tensordot(a, b)

        a = ones(10, chunk_size=2)
        b = ones((10, 20), chunk_size=2)
        c = dot(a, b)
        self.assertEqual(c.shape, (20,))
        self.assertEqual(calc_shape(c), c.shape)
        c.tiles()
        self.assertEqual(calc_shape(c.chunks[0]), c.chunks[0].shape)
        self.assertEqual(c.shape, tuple(sum(s) for s in c.nsplits))

        a = ones((10, 20), chunk_size=2)
        b = ones(20, chunk_size=2)
        c = dot(a, b)
        self.assertEqual(c.shape, (10,))
        self.assertEqual(calc_shape(c), c.shape)
        c.tiles()
        self.assertEqual(calc_shape(c.chunks[0]), c.chunks[0].shape)
        self.assertEqual(c.shape, tuple(sum(s) for s in c.nsplits))

        v = ones((100, 100), chunk_size=10)
        tv = v.dot(v)
        self.assertEqual(tv.shape, (100, 100))
        self.assertEqual(calc_shape(tv), tv.shape)
        tv.tiles()
        self.assertEqual(calc_shape(tv.chunks[0]), tv.chunks[0].shape)
        self.assertEqual(tv.shape, tuple(sum(s) for s in tv.nsplits))

        a = ones((10, 20), chunk_size=2)
        b = ones((30, 20), chunk_size=2)
        c = inner(a, b)
        self.assertEqual(c.shape, (10, 30))
        self.assertEqual(calc_shape(c), c.shape)
        c.tiles()
        self.assertEqual(calc_shape(c.chunks[0]), c.chunks[0].shape)
        self.assertEqual(c.shape, tuple(sum(s) for s in c.nsplits))

    def testDot(self):
        t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()
        t2 = t1.T

        self.assertTrue(t1.dot(t2).issparse())
        self.assertIs(type(t1.dot(t2)), SparseTensor)
        self.assertFalse(t1.dot(t2, sparse=False).issparse())
        self.assertIs(type(t1.dot(t2, sparse=False)), Tensor)
        self.assertEqual(calc_shape(t1.dot(t2)), (t1.dot(t2)).shape)

    def testFrexp(self):
        t1 = ones((3, 4, 5), chunk_size=2)
        op_type = type(t1.op)

        o1, o2 = frexp(t1)

        self.assertIs(o1.op, o2.op)
        self.assertNotEqual(o1.dtype, o2.dtype)
        self.assertEqual(calc_shape(o1), (o1.shape, o2.shape))
        self.assertEqual(calc_shape(o2), (o1.shape, o2.shape))

        o1, o2 = frexp(t1, t1)

        self.assertIs(o1, t1)
        self.assertIsNot(o1.inputs[0], t1)
        self.assertIsInstance(o1.inputs[0].op, op_type)
        self.assertIsNot(o2.inputs[0], t1)
        self.assertEqual(calc_shape(o1), (o1.shape, o2.shape))
        self.assertEqual(calc_shape(o2), (o1.shape, o2.shape))

    def testDtype(self):
        t1 = ones((2, 3), dtype='f4', chunk_size=2)

        t = truediv(t1, 2, dtype='f8')

        self.assertEqual(t.dtype, np.float64)

        with self.assertRaises(TypeError):
            truediv(t1, 2, dtype='i4')

    def testNegative(self):
        t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

        t = negative(t1)
        self.assertTrue(t.issparse())
        self.assertIs(type(t), SparseTensor)
        self.assertEqual(calc_shape(t), t.shape)

        t.tiles()
        self.assertTrue(t.chunks[0].op.sparse)
        self.assertEqual(calc_shape(t.chunks[0]), t.chunks[0].shape)

    def testCos(self):
        t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

        t = cos(t1)
        self.assertFalse(t.issparse())
        self.assertIs(type(t), Tensor)
        self.assertEqual(calc_shape(t), t.shape)

    def testAround(self):
        t1 = ones((2, 3), dtype='f4', chunk_size=2)

        t = around(t1, decimals=3)

        self.assertFalse(t.issparse())
        self.assertEqual(t.op.decimals, 3)
        self.assertEqual(calc_shape(t), t.shape)

        t.tiles()

        self.assertEqual(t.chunks[0].op.decimals, 3)
        self.assertEqual(calc_shape(t.chunks[0]), t.chunks[0].shape)

    def testIsclose(self):
        t1 = ones((2, 3), dtype='f4', chunk_size=2)

        atol = 1e-4
        rtol = 1e-5
        equal_nan = True

        t = isclose(t1, 2, atol=atol, rtol=rtol, equal_nan=equal_nan)

        self.assertIsInstance(t.op, IscloseConstant)
        self.assertEqual(t.op.atol, atol)
        self.assertEqual(t.op.rtol, rtol)
        self.assertEqual(t.op.equal_nan, equal_nan)
        self.assertEqual(calc_shape(t), t.shape)

        t.tiles()

        self.assertIsInstance(t.chunks[0].op, IscloseConstant)
        self.assertEqual(t.chunks[0].op.atol, atol)
        self.assertEqual(t.chunks[0].op.rtol, rtol)
        self.assertEqual(t.chunks[0].op.equal_nan, equal_nan)
        self.assertEqual(calc_shape(t.chunks[0]), t.chunks[0].shape)

        t1 = ones((2, 3), dtype='f4', chunk_size=2)
        t2 = ones((2, 3), dtype='f4', chunk_size=2)

        atol = 1e-4
        rtol = 1e-5
        equal_nan = True

        t = isclose(t1, t2, atol=atol, rtol=rtol, equal_nan=equal_nan)

        self.assertIsInstance(t.op, Isclose)
        self.assertEqual(t.op.atol, atol)
        self.assertEqual(t.op.rtol, rtol)
        self.assertEqual(t.op.equal_nan, equal_nan)
        self.assertEqual(calc_shape(t), t.shape)

        t.tiles()

        self.assertIsInstance(t.chunks[0].op, Isclose)
        self.assertEqual(t.chunks[0].op.atol, atol)
        self.assertEqual(t.chunks[0].op.rtol, rtol)
        self.assertEqual(t.chunks[0].op.equal_nan, equal_nan)
        self.assertEqual(calc_shape(t.chunks[0]), t.chunks[0].shape)

    def testMatmul(self):
        a_data = [[1, 0], [0, 1]]
        b_data = [[4, 1], [2, 2]]

        a = tensor(a_data, chunk_size=1)
        b = tensor(b_data, chunk_size=1)

        t = matmul(a, b)

        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(calc_shape(t), t.shape)
        t.tiles()
        self.assertEqual(t.shape, tuple(sum(s) for s in t.nsplits))
        self.assertEqual(calc_shape(t.chunks[0]), t.chunks[0].shape)

        b_data = [1, 2]
        b = tensor(b_data, chunk_size=1)

        t = matmul(a, b)

        self.assertEqual(t.shape, (2,))
        self.assertEqual(calc_shape(t), t.shape)
        t.tiles()
        self.assertEqual(t.shape, tuple(sum(s) for s in t.nsplits))
        self.assertEqual(calc_shape(t.chunks[0]), t.chunks[0].shape)

        t = matmul(b, a)

        self.assertEqual(t.shape, (2,))
        self.assertEqual(calc_shape(t), t.shape)
        t.tiles()
        self.assertEqual(t.shape, tuple(sum(s) for s in t.nsplits))
        self.assertEqual(calc_shape(t.chunks[0]), t.chunks[0].shape)

        a_data = np.arange(2 * 2 * 4).reshape((2, 2, 4))
        b_data = np.arange(2 * 2 * 4).reshape((2, 4, 2))

        a = tensor(a_data, chunk_size=1)
        b = tensor(b_data, chunk_size=1)

        t = matmul(a, b)

        self.assertEqual(t.shape, (2, 2, 2))
        self.assertEqual(calc_shape(t), t.shape)
        t.tiles()
        self.assertEqual(t.shape, tuple(sum(s) for s in t.nsplits))
        self.assertEqual(calc_shape(t.chunks[0]), t.chunks[0].shape)

        t = matmul(tensor([2j, 3j], chunk_size=1), tensor([2j, 3j], chunk_size=1))

        self.assertEqual(t.shape, ())
        self.assertEqual(calc_shape(t), t.shape)
        t.tiles()
        self.assertEqual(t.shape, tuple(sum(s) for s in t.nsplits))
        self.assertEqual(calc_shape(t.chunks[0]), t.chunks[0].shape)

        with self.assertRaises(ValueError):
            matmul([1, 2], 3)

        with self.assertRaises(ValueError):
            matmul(np.random.randn(2, 3, 4), np.random.randn(3, 4, 3))

        t = matmul(tensor(np.random.randn(2, 3, 4), chunk_size=2),
                   tensor(np.random.randn(3, 1, 4, 3), chunk_size=3))
        self.assertEqual(t.shape, (3, 2, 3, 3))

        v = ones((100, 100), chunk_size=10)
        tv = matmul(v, v)
        self.assertEqual(tv.shape, (100, 100))
        self.assertEqual(calc_shape(tv), tv.shape)
        tv.tiles()
        self.assertEqual(tv.shape, tuple(sum(s) for s in tv.nsplits))
        self.assertEqual(calc_shape(tv.chunks[0]), tv.chunks[0].shape)

    def testGetSetReal(self):
        a_data = np.array([1+2j, 3+4j, 5+6j])
        a = tensor(a_data, chunk_size=2)

        with self.assertRaises(ValueError):
            a.real = [2, 4]

    def testBuildMode(self):
        t1 = ones((2, 3), chunk_size=2)
        self.assertTrue(t1 == 2)

        with build_mode():
            self.assertFalse(t1 == 2)
