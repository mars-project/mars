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

from mars.tiles import get_tiled
from mars.tensor.datasource import ones, empty
from mars.tensor.merge import concatenate, stack


class Test(unittest.TestCase):
    def testConcatenate(self):
        a = ones((10, 20, 30), chunk_size=10)
        b = ones((20, 20, 30), chunk_size=20)

        c = concatenate([a, b])
        self.assertEqual(c.shape, (30, 20, 30))

        a = ones((10, 20, 30), chunk_size=10)
        b = ones((10, 20, 40), chunk_size=20)

        c = concatenate([a, b], axis=-1)
        self.assertEqual(c.shape, (10, 20, 70))

        with self.assertRaises(ValueError):
            a = ones((10, 20, 30), chunk_size=10)
            b = ones((20, 30, 30), chunk_size=20)

            concatenate([a, b])

        with self.assertRaises(ValueError):
            a = ones((10, 20, 30), chunk_size=10)
            b = ones((20, 20), chunk_size=20)

            concatenate([a, b])

        a = ones((10, 20, 30), chunk_size=5)
        b = ones((20, 20, 30), chunk_size=10)

        c = concatenate([a, b]).tiles()
        a = get_tiled(a)
        self.assertEqual(c.chunk_shape[0], 4)
        self.assertEqual(c.chunk_shape[1], 4)
        self.assertEqual(c.chunk_shape[2], 6)
        self.assertEqual(c.nsplits, ((5, 5, 10, 10), (5,) * 4, (5,) * 6))
        self.assertEqual(c.cix[0, 0, 0].key, a.cix[0, 0, 0].key)
        self.assertEqual(c.cix[1, 0, 0].key, a.cix[1, 0, 0].key)

    def testStack(self):
        raw_arrs = [ones((3, 4), chunk_size=2) for _ in range(10)]
        arr2 = stack(raw_arrs, axis=0)

        self.assertEqual(arr2.shape, (10, 3, 4))

        arr2 = arr2.tiles()
        self.assertEqual(arr2.nsplits, ((1,) * 10, (2, 1), (2, 2)))

        arr3 = stack(raw_arrs, axis=1)

        self.assertEqual(arr3.shape, (3, 10, 4))

        arr3 = arr3.tiles()
        self.assertEqual(arr3.nsplits, ((2, 1), (1,) * 10, (2, 2)))

        arr4 = stack(raw_arrs, axis=2)

        self.assertEqual(arr4.shape, (3, 4, 10))

        arr4 = arr4.tiles()
        self.assertEqual(arr4.nsplits, ((2, 1), (2, 2), (1,) * 10))

        with self.assertRaises(ValueError):
            raw_arrs2 = [ones((3, 4), chunk_size=2), ones((4, 3), chunk_size=2)]
            stack(raw_arrs2)

        with self.assertRaises(np.AxisError):
            stack(raw_arrs, axis=3)

        arr5 = stack(raw_arrs, -1).tiles()
        self.assertEqual(arr5.nsplits, ((2, 1), (2, 2), (1,) * 10))

        arr6 = stack(raw_arrs, -3).tiles()
        self.assertEqual(arr6.nsplits, ((1,) * 10, (2, 1), (2, 2)))

        with self.assertRaises(np.AxisError):
            stack(raw_arrs, axis=-4)

        with self.assertRaises(TypeError):
            stack(raw_arrs, out=1)

        with self.assertRaises(ValueError):
            stack(raw_arrs, empty((1, 10, 3, 4)))
