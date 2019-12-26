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

from mars.tensor.datasource import ones, tensor
from mars.tests.core import TestExecutor


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = TestExecutor('numpy')

    def testReshapeExecution(self):
        x = ones((1, 2, 3), chunk_size=[4, 3, 5])
        y = x.reshape(3, 2)
        res = self.executor.execute_tensor(y)[0]
        self.assertEqual(y.shape, (3, 2))
        np.testing.assert_equal(res, np.ones((3, 2)))

        data = np.random.rand(6, 4)
        x2 = tensor(data, chunk_size=2)
        y2 = x2.reshape(3, 8, order='F')
        res = self.executor.execute_tensor(y2, concat=True)[0]
        expected = data.reshape((3, 8), order='F')
        np.testing.assert_array_equal(res, expected)
        self.assertTrue(res.flags['F_CONTIGUOUS'])
        self.assertFalse(res.flags['C_CONTIGUOUS'])

        data2 = np.asfortranarray(np.random.rand(6, 4))
        x3 = tensor(data2, chunk_size=2)
        y3 = x3.reshape(3, 8)
        res = self.executor.execute_tensor(y3, concat=True)[0]
        expected = data2.reshape((3, 8))
        np.testing.assert_array_equal(res, expected)
        self.assertTrue(res.flags['C_CONTIGUOUS'])
        self.assertFalse(res.flags['F_CONTIGUOUS'])

        data2 = np.asfortranarray(np.random.rand(6, 4))
        x3 = tensor(data2, chunk_size=2)
        y3 = x3.reshape(3, 8, order='F')
        res = self.executor.execute_tensor(y3, concat=True)[0]
        expected = data2.reshape((3, 8), order='F')
        np.testing.assert_array_equal(res, expected)
        self.assertTrue(res.flags['F_CONTIGUOUS'])
        self.assertFalse(res.flags['C_CONTIGUOUS'])

    def testShuffleReshapeExecution(self):
        a = ones((31, 27), chunk_size=10)
        b = a.reshape(27, 31)
        b.op.extra_params['_reshape_with_shuffle'] = True

        res = self.executor.execute_tensor(b, concat=True)[0]
        np.testing.assert_array_equal(res, np.ones((27, 31)))

        b2 = a.reshape(27, 31, order='F')
        b.op.extra_params['_reshape_with_shuffle'] = True
        res = self.executor.execute_tensor(b2)[0]
        self.assertTrue(res.flags['F_CONTIGUOUS'])
        self.assertFalse(res.flags['C_CONTIGUOUS'])

        data = np.random.rand(6, 4)
        x2 = tensor(data, chunk_size=2)
        y2 = x2.reshape(4, 6, order='F')
        y2.op.extra_params['_reshape_with_shuffle'] = True
        res = self.executor.execute_tensor(y2, concat=True)[0]
        expected = data.reshape((4, 6), order='F')
        np.testing.assert_array_equal(res, expected)
        self.assertTrue(res.flags['F_CONTIGUOUS'])
        self.assertFalse(res.flags['C_CONTIGUOUS'])

        data2 = np.asfortranarray(np.random.rand(6, 4))
        x3 = tensor(data2, chunk_size=2)
        y3 = x3.reshape(4, 6)
        y3.op.extra_params['_reshape_with_shuffle'] = True
        res = self.executor.execute_tensor(y3, concat=True)[0]
        expected = data2.reshape((4, 6))
        np.testing.assert_array_equal(res, expected)
        self.assertTrue(res.flags['C_CONTIGUOUS'])
        self.assertFalse(res.flags['F_CONTIGUOUS'])
