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

from mars import tensor as mt
from mars.tensor.lib import nd_grid
from mars.tests.core import TestBase, ExecutorForTest


class Test(TestBase):
    def setUp(self):
        self.executor = ExecutorForTest('numpy')

    def testIndexTricks(self):
        mgrid = nd_grid()
        g = mgrid[0:5, 0:5]
        g.tiles()  # tileable means no loop exists

        ogrid = nd_grid(sparse=True)
        o = ogrid[0:5, 0:5]
        [ob.tiles() for ob in o]  # tilesable means no loop exists

    def testR_(self):
        r = mt.r_[mt.array([1, 2, 3]), 0, 0, mt.array([4, 5, 6])]

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.r_[np.array([1, 2, 3]), 0, 0, np.array([4, 5, 6])]

        np.testing.assert_array_equal(result, expected)

        r = mt.r_[-1:1:6j, [0]*3, 5, 6]

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.r_[-1:1:6j, [0]*3, 5, 6]

        np.testing.assert_array_equal(result, expected)

        r = mt.r_[-1:1:6j]

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.r_[-1:1:6j]

        np.testing.assert_array_equal(result, expected)

        raw = [[0, 1, 2], [3, 4, 5]]
        a = mt.array(raw, chunk_size=2)
        r = mt.r_['-1', a, a]

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.r_['-1', raw, raw]

        np.testing.assert_array_equal(result, expected)

        r = mt.r_['0,2', [1, 2, 3], [4, 5, 6]]

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.r_['0,2', [1, 2, 3], [4, 5, 6]]

        np.testing.assert_array_equal(result, expected)

        r = mt.r_['0,2,0', [1, 2, 3], [4, 5, 6]]

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.r_['0,2,0', [1, 2, 3], [4, 5, 6]]
        np.testing.assert_array_equal(result, expected)

        r = mt.r_['1,2,0', [1, 2, 3], [4, 5, 6]]

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.r_['1,2,0', [1, 2, 3], [4, 5, 6]]
        np.testing.assert_array_equal(result, expected)

        self.assertEqual(len(mt.r_), 0)

        with self.assertRaises(ValueError):
            _ = mt.r_[:3, 'wrong']

    def testC_(self):
        r = mt.c_[mt.array([1, 2, 3]), mt.array([4, 5, 6])]

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.c_[np.array([1, 2, 3]), np.array([4, 5, 6])]
        np.testing.assert_array_equal(result, expected)

        r = mt.c_[mt.array([[1, 2, 3]]), 0, 0, mt.array([[4, 5, 6]])]

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.c_[np.array([[1, 2, 3]]), 0, 0, np.array([[4, 5, 6]])]
        np.testing.assert_array_equal(result, expected)

        r = mt.c_[:3, 1:4]
        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.c_[:3, 1:4]
        np.testing.assert_array_equal(result, expected)
