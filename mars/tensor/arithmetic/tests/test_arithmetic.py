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

from mars.tensor.datasource import array, ones, tensor, empty
from mars.tensor.fetch import TensorFetch
from mars.tensor.arithmetic import add, subtract, truediv, log, frexp, around, \
    isclose, isfinite, negative, cos, TensorAdd, TensorSubtract, TensorLog, TensorIsclose, TensorGreaterThan
from mars.tensor.linalg import matmul
from mars.tensor.core import Tensor, SparseTensor
from mars.core import build_mode


class Test(unittest.TestCase):
    def testAdd(self):
        t1 = ones((3, 4), chunk_size=2)
        t2 = ones(4, chunk_size=2)
        t3 = t1 + t2
        k1 = t3.key
        self.assertFalse(t3.op.gpu)
        t3.tiles()
        self.assertNotEqual(t3.key, k1)
        self.assertEqual(t3.shape, (3, 4))
        self.assertEqual(len(t3.chunks), 4)
        self.assertEqual(t3.chunks[0].inputs, [t1.chunks[0].data, t2.chunks[0].data])
        self.assertEqual(t3.chunks[1].inputs, [t1.chunks[1].data, t2.chunks[1].data])
        self.assertEqual(t3.chunks[2].inputs, [t1.chunks[2].data, t2.chunks[0].data])
        self.assertEqual(t3.chunks[3].inputs, [t1.chunks[3].data, t2.chunks[1].data])
        self.assertEqual(t3.op.dtype, np.dtype('f8'))
        self.assertEqual(t3.chunks[0].op.dtype, np.dtype('f8'))

        t4 = t1 + 1
        t4.tiles()
        self.assertEqual(t4.shape, (3, 4))
        self.assertEqual(len(t3.chunks), 4)
        self.assertEqual(t4.chunks[0].inputs, [t1.chunks[0].data])
        self.assertEqual(t4.chunks[0].op.rhs, 1)
        self.assertEqual(t4.chunks[1].inputs, [t1.chunks[1].data])
        self.assertEqual(t4.chunks[1].op.rhs, 1)
        self.assertEqual(t4.chunks[2].inputs, [t1.chunks[2].data])
        self.assertEqual(t4.chunks[2].op.rhs, 1)
        self.assertEqual(t4.chunks[3].inputs, [t1.chunks[3].data])
        self.assertEqual(t4.chunks[3].op.rhs, 1)

        t5 = add([1, 2, 3, 4], 1)
        t5.tiles()
        self.assertEqual(t4.chunks[0].inputs, [t1.chunks[0].data])

        t6 = ones((3, 4), chunk_size=2, gpu=True)
        t7 = ones(4, chunk_size=2, gpu=True)
        t8 = t6 + t7
        self.assertTrue(t8.op.gpu)
        t8.tiles()
        self.assertTrue(t8.chunks[0].op.gpu)
        t9 = t6 + t2
        self.assertIsNone(t9.op.gpu)
        t9.tiles()
        self.assertIsNone(t9.chunks[0].op.gpu)

        # sparse tests
        t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

        t = t1 + 1
        self.assertFalse(t.op.gpu)
        self.assertTrue(t.issparse())
        self.assertIs(type(t), SparseTensor)

        t.tiles()
        self.assertTrue(t.chunks[0].op.sparse)

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

    def testAddOrder(self):
        raw_a = np.random.rand(4, 2)
        raw_b = np.asfortranarray(np.random.rand(4, 2))
        t1 = tensor(raw_a)
        t2 = tensor(raw_b)
        out = tensor(raw_b)

        # C + scalar
        self.assertEqual((t1 + 1).flags['C_CONTIGUOUS'], (raw_a + 1).flags['C_CONTIGUOUS'])
        self.assertEqual((t1 + 1).flags['F_CONTIGUOUS'], (raw_a + 1).flags['F_CONTIGUOUS'])
        # C + C
        self.assertEqual((t1 + t1).flags['C_CONTIGUOUS'], (raw_a + raw_a).flags['C_CONTIGUOUS'])
        self.assertEqual((t1 + t1).flags['F_CONTIGUOUS'], (raw_a + raw_a).flags['F_CONTIGUOUS'])
        # F + scalar
        self.assertEqual((t2 + 1).flags['C_CONTIGUOUS'], (raw_b + 1).flags['C_CONTIGUOUS'])
        self.assertEqual((t2 + 1).flags['F_CONTIGUOUS'], (raw_b + 1).flags['F_CONTIGUOUS'])
        # F + F
        self.assertEqual((t2 + t2).flags['C_CONTIGUOUS'], (raw_b + raw_b).flags['C_CONTIGUOUS'])
        self.assertEqual((t2 + t2).flags['F_CONTIGUOUS'], (raw_b + raw_b).flags['F_CONTIGUOUS'])
        # C + F
        self.assertEqual((t1 + t2).flags['C_CONTIGUOUS'], (raw_a + raw_b).flags['C_CONTIGUOUS'])
        self.assertEqual((t1 + t2).flags['F_CONTIGUOUS'], (raw_a + raw_b).flags['F_CONTIGUOUS'])
        # C + C + out
        self.assertEqual(add(t1, t1, out=out).flags['C_CONTIGUOUS'],
                         np.add(raw_a, raw_a, out=np.empty((4, 2), order='F')).flags['C_CONTIGUOUS'])
        self.assertEqual(add(t1, t1, out=out).flags['F_CONTIGUOUS'],
                         np.add(raw_a, raw_a, out=np.empty((4, 2), order='F')).flags['F_CONTIGUOUS'])

        with self.assertRaises(TypeError):
            add(t1, 1, order='B')

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

        self.assertIsInstance(t2.op, TensorAdd)

        t3 = t1 - np.datetime64('2005-02-02')

        self.assertIsInstance(t3.op, TensorSubtract)
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

        self.assertIsInstance(t1.op, TensorAdd)
        self.assertEqual(t1.op.out.key, t1.op.lhs.key)
        self.assertIs(t3, t1)
        self.assertEqual(t3.shape, (3, 4))
        self.assertEqual(t3.op.lhs.extra_params.raw_chunk_size, 2)
        self.assertIs(t3.op.rhs, t2.data)
        self.assertNotEqual(t3.key, t3.op.lhs.key)

        t3.tiles()

        self.assertIsInstance(t1.chunks[0].op, TensorAdd)
        self.assertEqual(t1.chunks[0].op.out.key, t1.chunks[0].op.lhs.key)

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

    def testLogWithOutWhere(self):
        t1 = ones((3, 4), chunk_size=2)

        t2 = log(t1, out=t1)

        self.assertIsInstance(t2.op, TensorLog)
        self.assertEqual(t1.op.out.key, t1.op.input.key)
        self.assertIs(t2, t1)
        self.assertEqual(t2.op.input.extra_params.raw_chunk_size, 2)
        self.assertNotEqual(t2.key, t2.op.input.key)

        t3 = empty((3, 4), chunk_size=2)
        t4 = log(t1, out=t3, where=t1 > 0)
        self.assertIsInstance(t4.op, TensorLog)
        self.assertIs(t4, t3)
        self.assertEqual(t2.op.input.extra_params.raw_chunk_size, 2)
        self.assertNotEqual(t2.key, t2.op.input.key)

    def testCopyAdd(self):
        t1 = ones((3, 4), chunk_size=2)
        t2 = ones(4, chunk_size=2)
        t3 = t1 + t2
        t3.tiles()

        c = t3.chunks[0]
        inputs = c.op.lhs, TensorFetch().new_chunk(
            c.op.rhs.inputs, shape=c.op.rhs.shape, index=c.op.rhs.index, _key=c.op.rhs.key)
        new_c = c.op.copy().reset_key().new_chunk(inputs, shape=c.shape, _key='new_key')
        self.assertEqual(new_c.key, 'new_key')
        self.assertIs(new_c.inputs[1], new_c.op.rhs)
        self.assertIsInstance(new_c.inputs[1].op, TensorFetch)

    def testCompare(self):
        t1 = ones(4, chunk_size=2) * 2
        t2 = ones(4, chunk_size=2)
        t3 = t1 > t2
        t3.tiles()
        self.assertEqual(len(t3.chunks), 2)
        self.assertIsInstance(t3.op, TensorGreaterThan)

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

    def testFrexp(self):
        t1 = ones((3, 4, 5), chunk_size=2)
        t2 = empty((3, 4, 5), dtype=np.float_, chunk_size=2)
        op_type = type(t1.op)

        o1, o2 = frexp(t1)

        self.assertIs(o1.op, o2.op)
        self.assertNotEqual(o1.dtype, o2.dtype)

        o1, o2 = frexp(t1, t1)

        self.assertIs(o1, t1)
        self.assertIsNot(o1.inputs[0], t1)
        self.assertIsInstance(o1.inputs[0].op, op_type)
        self.assertIsNot(o2.inputs[0], t1)

        o1, o2 = frexp(t1, t2, where=t1 > 0)

        op_type = type(t2.op)
        self.assertIs(o1, t2)
        self.assertIsNot(o1.inputs[0], t1)
        self.assertIsInstance(o1.inputs[0].op, op_type)
        self.assertIsNot(o2.inputs[0], t1)

    def testFrexpOrder(self):
        raw1 = np.asfortranarray(np.random.rand(2, 4))
        t = tensor(raw1)
        o1 = tensor(np.random.rand(2, 4))

        o1, o2 = frexp(t, out1=o1)

        self.assertEqual(o1.flags['C_CONTIGUOUS'],
                         np.frexp(raw1, np.empty((2, 4)))[0].flags['C_CONTIGUOUS'])
        self.assertEqual(o1.flags['F_CONTIGUOUS'],
                         np.frexp(raw1, np.empty((2, 4)))[0].flags['F_CONTIGUOUS'])
        self.assertEqual(o2.flags['C_CONTIGUOUS'],
                         np.frexp(raw1)[1].flags['C_CONTIGUOUS'])
        self.assertEqual(o2.flags['F_CONTIGUOUS'],
                         np.frexp(raw1)[1].flags['F_CONTIGUOUS'])

    def testDtype(self):
        t1 = ones((2, 3), dtype='f4', chunk_size=2)

        t = truediv(t1, 2, dtype='f8')

        self.assertEqual(t.dtype, np.float64)

        with self.assertRaises(TypeError):
            truediv(t1, 2, dtype='i4')

    def testNegative(self):
        t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

        t = negative(t1)
        self.assertFalse(t.op.gpu)
        self.assertTrue(t.issparse())
        self.assertIs(type(t), SparseTensor)

        t.tiles()
        self.assertTrue(t.chunks[0].op.sparse)

    def testNegativeOrder(self):
        raw1 = np.random.rand(4, 2)
        raw2 = np.asfortranarray(np.random.rand(4, 2))
        t1 = tensor(raw1)
        t2 = tensor(raw2)
        t3 = tensor(raw1)
        t4 = tensor(raw2)

        # C
        self.assertEqual(negative(t1).flags['C_CONTIGUOUS'], np.negative(raw1).flags['C_CONTIGUOUS'])
        self.assertEqual(negative(t1).flags['F_CONTIGUOUS'], np.negative(raw1).flags['F_CONTIGUOUS'])
        # F
        self.assertEqual(negative(t2).flags['C_CONTIGUOUS'], np.negative(raw2).flags['C_CONTIGUOUS'])
        self.assertEqual(negative(t2).flags['F_CONTIGUOUS'], np.negative(raw2).flags['F_CONTIGUOUS'])
        # C + out
        self.assertEqual(negative(t1, out=t4).flags['C_CONTIGUOUS'],
                         np.negative(raw1, out=np.empty((4, 2), order='F')).flags['C_CONTIGUOUS'])
        self.assertEqual(negative(t1, out=t4).flags['F_CONTIGUOUS'],
                         np.negative(raw1, out=np.empty((4, 2), order='F')).flags['F_CONTIGUOUS'])
        # F + out
        self.assertEqual(negative(t2, out=t3).flags['C_CONTIGUOUS'],
                         np.negative(raw1, out=np.empty((4, 2), order='C')).flags['C_CONTIGUOUS'])
        self.assertEqual(negative(t2, out=t3).flags['F_CONTIGUOUS'],
                         np.negative(raw1, out=np.empty((4, 2), order='C')).flags['F_CONTIGUOUS'])

        with self.assertRaises(TypeError):
            negative(t1, order='B')

    def testCos(self):
        t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

        t = cos(t1)
        self.assertTrue(t.issparse())
        self.assertIs(type(t), SparseTensor)

    def testAround(self):
        t1 = ones((2, 3), dtype='f4', chunk_size=2)

        t = around(t1, decimals=3)

        self.assertFalse(t.issparse())
        self.assertEqual(t.op.decimals, 3)

        t.tiles()

        self.assertEqual(t.chunks[0].op.decimals, 3)

    def testIsclose(self):
        t1 = ones((2, 3), dtype='f4', chunk_size=2)

        atol = 1e-4
        rtol = 1e-5
        equal_nan = True

        t = isclose(t1, 2, atol=atol, rtol=rtol, equal_nan=equal_nan)

        self.assertIsInstance(t.op, TensorIsclose)
        self.assertEqual(t.op.atol, atol)
        self.assertEqual(t.op.rtol, rtol)
        self.assertEqual(t.op.equal_nan, equal_nan)

        t.tiles()

        self.assertIsInstance(t.chunks[0].op, TensorIsclose)
        self.assertEqual(t.chunks[0].op.atol, atol)
        self.assertEqual(t.chunks[0].op.rtol, rtol)
        self.assertEqual(t.chunks[0].op.equal_nan, equal_nan)

        t1 = ones((2, 3), dtype='f4', chunk_size=2)
        t2 = ones((2, 3), dtype='f4', chunk_size=2)

        atol = 1e-4
        rtol = 1e-5
        equal_nan = True

        t = isclose(t1, t2, atol=atol, rtol=rtol, equal_nan=equal_nan)

        self.assertIsInstance(t.op, TensorIsclose)
        self.assertEqual(t.op.atol, atol)
        self.assertEqual(t.op.rtol, rtol)
        self.assertEqual(t.op.equal_nan, equal_nan)

        t.tiles()

        self.assertIsInstance(t.chunks[0].op, TensorIsclose)
        self.assertEqual(t.chunks[0].op.atol, atol)
        self.assertEqual(t.chunks[0].op.rtol, rtol)
        self.assertEqual(t.chunks[0].op.equal_nan, equal_nan)

    def testMatmul(self):
        a_data = [[1, 0], [0, 1]]
        b_data = [[4, 1], [2, 2]]

        a = tensor(a_data, chunk_size=1)
        b = tensor(b_data, chunk_size=1)

        t = matmul(a, b)

        self.assertEqual(t.shape, (2, 2))
        t.tiles()
        self.assertEqual(t.shape, tuple(sum(s) for s in t.nsplits))

        b_data = [1, 2]
        b = tensor(b_data, chunk_size=1)

        t = matmul(a, b)

        self.assertEqual(t.shape, (2,))
        t.tiles()
        self.assertEqual(t.shape, tuple(sum(s) for s in t.nsplits))

        t = matmul(b, a)

        self.assertEqual(t.shape, (2,))
        t.tiles()
        self.assertEqual(t.shape, tuple(sum(s) for s in t.nsplits))

        a_data = np.arange(2 * 2 * 4).reshape((2, 2, 4))
        b_data = np.arange(2 * 2 * 4).reshape((2, 4, 2))

        a = tensor(a_data, chunk_size=1)
        b = tensor(b_data, chunk_size=1)

        t = matmul(a, b)

        self.assertEqual(t.shape, (2, 2, 2))
        t.tiles()
        self.assertEqual(t.shape, tuple(sum(s) for s in t.nsplits))

        t = matmul(tensor([2j, 3j], chunk_size=1), tensor([2j, 3j], chunk_size=1))

        self.assertEqual(t.shape, ())
        t.tiles()
        self.assertEqual(t.shape, tuple(sum(s) for s in t.nsplits))

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
        tv.tiles()
        self.assertEqual(tv.shape, tuple(sum(s) for s in tv.nsplits))

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
