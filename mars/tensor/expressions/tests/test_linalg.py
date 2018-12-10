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

import mars.tensor as mt


class Test(unittest.TestCase):
    def testQR(self):
        a = mt.random.rand(9, 6, chunks=(3, 6))
        q, r = mt.linalg.qr(a)

        self.assertEqual(q.shape, (9, 6))
        self.assertEqual(r.shape, (6, 6))

        q.tiles()

        self.assertEqual(len(q.chunks), 3)
        self.assertEqual(len(r.chunks), 1)

    def testNorm(self):
        data = np.random.rand(9, 6)

        a = mt.tensor(data, chunks=(2, 6))

        for ord in (None, 'nuc', np.inf, -np.inf, 0, 1, -1, 2, -2):
            for axis in (0, 1, (0, 1)):
                for keepdims in (True, False):
                    try:
                        res_shape = mt.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims).shape
                        expect_shape = np.linalg.norm(data, ord=ord, axis=axis, keepdims=keepdims).shape
                        self.assertEqual(res_shape, expect_shape)
                    except ValueError:
                        continue

    def testSVD(self):
        a = mt.random.rand(9, 6, chunks=(3, 6))
        U, s, V = mt.linalg.svd(a)

        self.assertEqual(U.shape, (9, 6))
        self.assertEqual(s.shape, (6,))
        self.assertEqual(V.shape, (6, 6))

        U.tiles()
        self.assertEqual(len(U.chunks), 3)
        self.assertEqual(len(s.chunks), 1)
        self.assertEqual(len(V.chunks), 1)

    def testLU(self):
        a = mt.random.randint(1, 10, (6, 6), chunks=3)
        p, l, u = mt.linalg.lu(a)
        l.tiles()

        self.assertEqual(l.shape, (6, 6))
        self.assertEqual(u.shape, (6, 6))
        self.assertEqual(p.shape, (6, 6))

    def testSolve(self):
        a = mt.random.randint(1, 10, (20, 20))
        b = mt.random.randint(1, 10, (20, ))
        x = mt.linalg.solve(a, b).tiles()

        self.assertEqual(x.shape, (20, ))

        a = mt.random.randint(1, 10, (20, 20), chunks=5)
        b = mt.random.randint(1, 10, (20, 3), chunks=5)
        x = mt.linalg.solve(a, b).tiles()

        self.assertEqual(x.shape, (20, 3))

    def testInv(self):
        a = mt.random.randint(1, 10, (20, 20), chunks=4)
        a_inv = mt.linalg.inv(a).tiles()

        self.assertEqual(a_inv.shape, (20, 20))
