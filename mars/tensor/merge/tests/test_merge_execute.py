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

import numpy as np
import scipy.sparse as sps

from mars.tensor.datasource import tensor, empty, eye, ones, zeros
from mars.tensor import concatenate, stack, hstack, vstack, dstack, column_stack, union1d, array, block
from mars.tests.core import TestBase


class Test(TestBase):
    def setUp(self):
        self.ctx, self.executor = self._create_test_context()

    def testConcatenateExecution(self):
        a_data = np.random.rand(10, 20, 30)
        b_data = np.random.rand(10, 20, 40)
        c_data = np.random.rand(10, 20, 50)

        a = tensor(a_data, chunk_size=5)
        b = tensor(b_data, chunk_size=6)
        c = tensor(c_data, chunk_size=7)

        d = concatenate([a, b, c], axis=-1)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.concatenate([a_data, b_data, c_data], axis=-1)
        self.assertTrue(np.array_equal(res, expected))

        a_data = sps.random(10, 30)
        b_data = sps.rand(10, 40)
        c_data = sps.rand(10, 50)

        a = tensor(a_data, chunk_size=5)
        b = tensor(b_data, chunk_size=6)
        c = tensor(c_data, chunk_size=7)

        d = concatenate([a, b, c], axis=-1)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.concatenate([a_data.A, b_data.A, c_data.A], axis=-1)
        self.assertTrue(np.array_equal(res.toarray(), expected))

    def testStackExecution(self):
        raw = [np.random.randn(3, 4) for _ in range(10)]
        arrs = [tensor(a, chunk_size=3) for a in raw]

        arr2 = stack(arrs)
        res = self.executor.execute_tensor(arr2, concat=True)
        self.assertTrue(np.array_equal(res[0], np.stack(raw)))

        arr3 = stack(arrs, axis=1)
        res = self.executor.execute_tensor(arr3, concat=True)
        self.assertTrue(np.array_equal(res[0], np.stack(raw, axis=1)))

        arr4 = stack(arrs, axis=2)
        res = self.executor.execute_tensor(arr4, concat=True)
        self.assertTrue(np.array_equal(res[0], np.stack(raw, axis=2)))

        raw2 = [np.asfortranarray(np.random.randn(3, 4)) for _ in range(10)]
        arr5 = [tensor(a, chunk_size=3) for a in raw2]

        arr6 = stack(arr5)
        res = self.executor.execute_tensor(arr6, concat=True)[0]
        expected = np.stack(raw2).copy('A')
        np.testing.assert_array_equal(res, expected)
        self.assertEqual(res.flags['C_CONTIGUOUS'], expected.flags['C_CONTIGUOUS'])
        self.assertEqual(res.flags['F_CONTIGUOUS'], expected.flags['F_CONTIGUOUS'])

        arr7 = stack(arr5, out=empty((10, 3, 4), order='F'))
        res = self.executor.execute_tensor(arr7, concat=True)[0]
        expected = np.stack(raw2, out=np.empty((10, 3, 4), order='F')).copy('A')
        np.testing.assert_array_equal(res, expected)
        self.assertEqual(res.flags['C_CONTIGUOUS'], expected.flags['C_CONTIGUOUS'])
        self.assertEqual(res.flags['F_CONTIGUOUS'], expected.flags['F_CONTIGUOUS'])

        # test stack with unknown shapes
        t = tensor(raw[0], chunk_size=3)
        t2 = t[t[:, 0] > 0.0]
        t3 = t2 + 1

        with self.ctx:
            arr8 = stack([t2, t3])
            result = self.executor.execute_tensors([arr8])[0]
            e = raw[0]
            e2 = e[e[:, 0] > 0.0]
            e3 = e2 + 1
            np.testing.assert_array_equal(result, np.stack([e2, e3]))

    def testHStackExecution(self):
        a_data = np.random.rand(10)
        b_data = np.random.rand(20)

        a = tensor(a_data, chunk_size=4)
        b = tensor(b_data, chunk_size=4)

        c = hstack([a, b])
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.hstack([a_data, b_data])
        self.assertTrue(np.array_equal(res, expected))

        a_data = np.random.rand(10, 20)
        b_data = np.random.rand(10, 5)

        a = tensor(a_data, chunk_size=3)
        b = tensor(b_data, chunk_size=4)

        c = hstack([a, b])
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.hstack([a_data, b_data])
        self.assertTrue(np.array_equal(res, expected))

    def testVStackExecution(self):
        a_data = np.random.rand(10)
        b_data = np.random.rand(10)

        a = tensor(a_data, chunk_size=4)
        b = tensor(b_data, chunk_size=4)

        c = vstack([a, b])
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.vstack([a_data, b_data])
        self.assertTrue(np.array_equal(res, expected))

        a_data = np.random.rand(10, 20)
        b_data = np.random.rand(5, 20)

        a = tensor(a_data, chunk_size=3)
        b = tensor(b_data, chunk_size=4)

        c = vstack([a, b])
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.vstack([a_data, b_data])
        self.assertTrue(np.array_equal(res, expected))

    def testDStackExecution(self):
        a_data = np.random.rand(10)
        b_data = np.random.rand(10)

        a = tensor(a_data, chunk_size=4)
        b = tensor(b_data, chunk_size=4)

        c = dstack([a, b])
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.dstack([a_data, b_data])
        self.assertTrue(np.array_equal(res, expected))

        a_data = np.random.rand(10, 20)
        b_data = np.random.rand(10, 20)

        a = tensor(a_data, chunk_size=3)
        b = tensor(b_data, chunk_size=4)

        c = dstack([a, b])
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.dstack([a_data, b_data])
        self.assertTrue(np.array_equal(res, expected))

    def testColumnStackExecution(self):
        a_data = np.array((1, 2, 3))
        b_data = np.array((2, 3, 4))
        a = tensor(a_data, chunk_size=1)
        b = tensor(b_data, chunk_size=2)

        c = column_stack((a, b))
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.column_stack((a_data, b_data))
        np.testing.assert_equal(res, expected)

        a_data = np.random.rand(4, 2, 3)
        b_data = np.random.rand(4, 2, 3)
        a = tensor(a_data, chunk_size=1)
        b = tensor(b_data, chunk_size=2)

        c = column_stack((a, b))
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.column_stack((a_data, b_data))
        np.testing.assert_equal(res, expected)

    def testUnion1dExecution(self):
        rs = np.random.RandomState(0)
        raw1 = rs.random(10)
        raw2 = rs.random(9)

        t1 = tensor(raw1, chunk_size=3)
        t2 = tensor(raw2, chunk_size=4)

        t = union1d(t1, t2, aggregate_size=1)
        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.union1d(raw1, raw2)
        np.testing.assert_array_equal(res, expected)

        t = union1d(t1, t2)
        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.union1d(raw1, raw2)
        np.testing.assert_array_equal(res, expected)

    def testBlockExecution(self):
        with self.assertRaises(TypeError):
            block((1, 2, 3))

        with self.assertRaises(ValueError):
            block([[1, 2], [[3, 4]]])

        with self.assertRaises(ValueError):
            block([])

        # Test large block.
        block([eye(512), eye(512), ones((512, 1))])

        a = eye(2) * 2
        b = eye(3) * 3
        c = block([
            [a, zeros((2, 3))],
            [ones((3, 2)), b]
        ])
        r = self.executor.execute_tensor(c, concat=True)[0]
        expected = array([[2., 0., 0., 0., 0.],
                          [0., 2., 0., 0., 0.],
                          [1., 1., 3., 0., 0.],
                          [1., 1., 0., 3., 0.],
                          [1., 1., 0., 0., 3.]])
        np.testing.assert_array_equal(r, expected)

        # eye with different chunk sizes
        a = eye(5, chunk_size=2) * 2
        b = eye(4, chunk_size=3) * 3
        c = block([
            [a, zeros((5, 4), chunk_size=4)],
            [ones((4, 5), chunk_size=5), b]
        ])
        r = self.executor.execute_tensor(c, concat=True)[0]
        expected = array([[2., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 2., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 2., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 2., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 2., 0., 0., 0., 0.],
                          [1., 1., 1., 1., 1., 3., 0., 0., 0.],
                          [1., 1., 1., 1., 1., 0., 3., 0., 0.],
                          [1., 1., 1., 1., 1., 0., 0., 3., 0.],
                          [1., 1., 1., 1., 1., 0., 0., 0., 3.]])
        np.testing.assert_array_equal(r, expected)

        # hstack([1, 2, 3])
        c = block([1, 2, 3])
        r = self.executor.execute_tensor(c, concat=True)[0]
        expected = array([1, 2, 3])
        np.testing.assert_array_equal(r, expected)

        # hstack([a, b, 10])
        a = array([1, 2, 3])
        b = array([2, 3, 4])
        c = block([a, b, 10])
        r = self.executor.execute_tensor(c, concat=True)[0]
        expected = array([1, 2, 3, 2, 3, 4, 10])
        np.testing.assert_array_equal(r, expected)

        # hstack([a, b, 10]) with different chunk sizes
        a = array([1, 2, 3, 4, 5, 6, 7], chunk_size=3)
        b = array([2, 3, 4, 5], chunk_size=4)
        c = block([a, b, 10])
        r = self.executor.execute_tensor(c, concat=True)[0]
        expected = array([1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 10])
        np.testing.assert_array_equal(r, expected)

        # hstack([A, B])
        A = ones((2, 2), int)
        B = 2 * A
        c = block([A, B])
        r = self.executor.execute_tensor(c, concat=True)[0]
        expected = array([[1, 1, 2, 2],
                          [1, 1, 2, 2]])
        np.testing.assert_array_equal(r, expected)

        # vstack([a, b])
        a = array([1, 2, 3])
        b = array([2, 3, 4])
        c = block([[a], [b]])
        r = self.executor.execute_tensor(c, concat=True)[0]
        expected = array([[1, 2, 3],
                          [2, 3, 4]])
        np.testing.assert_array_equal(r, expected)

        # vstack([a, b]) with different chunk sizes
        a = array([1, 2, 3, 4, 5, 6, 7], chunk_size=5)
        b = array([2, 3, 4, 5, 6, 7, 8], chunk_size=6)
        c = block([[a], [b]])
        r = self.executor.execute_tensor(c, concat=True)[0]
        expected = array([[1, 2, 3, 4, 5, 6, 7],
                          [2, 3, 4, 5, 6, 7, 8]])
        np.testing.assert_array_equal(r, expected)

        # vstack([A, B])
        A = ones((2, 2), int)
        B = 2 * A
        c = block([[A], [B]])
        r = self.executor.execute_tensor(c, concat=True)[0]
        expected = array([[1, 1],
                          [1, 1],
                          [2, 2],
                          [2, 2]])
        np.testing.assert_array_equal(r, expected)

        a = array(0)
        b = array([1])
        # atleast_1d(a)
        c = block([a])
        r = self.executor.execute_tensor(c, concat=True)[0]
        expected = array([0])
        np.testing.assert_array_equal(r, expected)
        # atleast_1d(b)
        c = block([b])
        r = self.executor.execute_tensor(c, concat=True)[0]
        expected = array([1])
        np.testing.assert_array_equal(r, expected)
        # atleast_2d(a)
        c = block([[a]])
        r = self.executor.execute_tensor(c, concat=True)[0]
        expected = array([[0]])
        np.testing.assert_array_equal(r, expected)
        # atleast_2d(b)
        c = block([[b]])
        r = self.executor.execute_tensor(c, concat=True)[0]
        expected = array([[1]])
        np.testing.assert_array_equal(r, expected)

