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

from mars.executor import Executor
from mars.tensor import ones, add, swapaxes, moveaxis, atleast_1d, atleast_2d, \
    atleast_3d, squeeze, tensor
from mars.session import LocalSession, Session


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')
        local_session = LocalSession()
        local_session._executor = self.executor
        self.session = Session()
        self.session._sess = local_session

    def testDecref(self):
        a = ones((10, 20), chunk_size=5)
        b = a + 1

        b.execute(session=self.session)

        self.assertEqual(len(self.executor.chunk_result), 1)

        del b
        # decref called
        self.assertEqual(len(self.executor.chunk_result), 0)

    def testArrayFunction(self):
        a = ones((10, 20), chunk_size=5)

        # test sum
        self.assertEqual(np.sum(a).execute(), 200)

        # test qr
        q, r = np.linalg.qr(a)
        self.assertTrue(np.allclose(np.dot(q, r), a).execute())

    def testViewDataOnSlice(self):
        data = np.random.rand(10, 20)
        a = tensor(data, chunk_size=6)
        b = a[:5, 5:10]
        b[:3, :3] = 3

        npa = data.copy()
        npb = npa[:5, 5:10]
        npb[:3, :3] = 3

        np.testing.assert_array_equal(b.execute(), npb)
        np.testing.assert_array_equal(a.execute(), npa)

    def testViewDataOnTranspose(self):
        data = np.random.rand(10, 20)
        a = tensor(data, chunk_size=6)
        b = a.T
        add(b, 1, out=b)

        np.testing.assert_array_equal(b.execute(), data.T + 1)
        np.testing.assert_array_equal(a.execute(), data + 1)

    def testViewDataOnSwapaxes(self):
        data = np.random.rand(10, 20)
        a = tensor(data, chunk_size=6)
        b = swapaxes(a, 1, 0)
        a[1] = 10

        npa = data.copy()
        npb = np.swapaxes(npa, 1, 0)
        npa[1] = 10

        np.testing.assert_array_equal(b.execute(), npb)
        np.testing.assert_array_equal(a.execute(), npa)

    def testViewDataOnMoveaxis(self):
        data = np.random.rand(10, 20)
        a = tensor(data, chunk_size=6)
        b = moveaxis(a, 1, 0)
        a[0][1] = 10

        npa = data.copy()
        npb = np.moveaxis(npa, 1, 0)
        npa[0][1] = 10

        np.testing.assert_array_equal(b.execute(), npb)
        np.testing.assert_array_equal(a.execute(), npa)

    def testViewDataOnAtleast1d(self):
        a = atleast_1d(1)
        b = a[:]
        b[0] = 10

        np.testing.assert_array_equal(b.execute(), np.array([10]))
        np.testing.assert_array_equal(a.execute(), np.array([10]))

    def testViewDataOnAtleast2d(self):
        data = np.random.rand(10)
        a = atleast_2d(tensor(data, chunk_size=5))
        b = add(a[:, :5], 1, out=a[:, 5:])

        npa = np.atleast_2d(data.copy())
        npb = np.add(npa[:, :5], 1, out=npa[:, 5:])

        np.testing.assert_array_equal(b.execute(), npb)
        np.testing.assert_array_equal(a.execute(), npa)

    def testViewDataOnAtleast3d(self):
        data = np.random.rand(10, 20)
        a = atleast_3d(tensor(data, chunk_size=5))
        b = a[:, :5, :10][0]
        c = add(b[:4], b[1:], out=a[0, 16:])

        npa = np.atleast_3d(data.copy())
        npb = npa[:, :5, :10][0]
        npc = np.add(npb[:4], npb[1:], out=npa[0, 16:])

        np.testing.assert_array_equal(c.execute(), npc)
        np.testing.assert_array_equal(b.execute(), npb)
        np.testing.assert_array_equal(a.execute(), npa)

    def testViewDataOnSqueeze(self):
        data = np.random.rand(1, 4, 1)
        a = tensor(data, chunk_size=2)
        b = squeeze(a, axis=0)
        b[:3] = 10

        npa = data.copy()
        npb = np.squeeze(npa, axis=0)
        npb[:3] = 10

        np.testing.assert_array_equal(b.execute(), npb)
        np.testing.assert_array_equal(a.execute(), npa)

    def testViewDataOnReshape(self):
        data = np.random.random((3, 4, 5))
        a = tensor(data.copy(), chunk_size=2)
        b = a.reshape((5, 4, 3))
        b[:3] = 10

        npa = data.copy()
        npb = npa.reshape((5, 4, 3))
        npb[:3] = 10

        np.testing.assert_array_equal(b.execute(), npb)
        np.testing.assert_array_equal(a.execute(), npa)

        data = np.random.random((4, 5))
        a2 = tensor(data.copy(), chunk_size=2)
        b2 = a2.reshape((5, 4), order='F')
        b2[:3] = 10

        npa = data.copy()
        npb = npa.reshape((5, 4), order='F')
        npb[:3] = 10

        b2_result = b2.execute()
        np.testing.assert_array_equal(b2_result, npb)
        np.testing.assert_array_equal(a2.execute(), npa)

    def testViewDataOnRavel(self):
        # ravel creates a view
        data = np.random.rand(3, 4, 5)
        a = tensor(data, chunk_size=2)
        b = a.ravel()
        b[:10] = 10

        npa = data.copy()
        npb = npa.ravel()
        npb[:10] = 10

        np.testing.assert_array_equal(b.execute(), npb)
        np.testing.assert_array_equal(a.execute(), npa)

        # flatten creates a copy
        data = np.random.rand(3, 4, 5)
        a = tensor(data, chunk_size=2)
        b = a.flatten()
        b[:10] = 10

        npa = data.copy()
        npb = npa.flatten()
        npb[:10] = 10

        np.testing.assert_array_equal(b.execute(), npb)
        np.testing.assert_array_equal(a.execute(), npa)

    def testCopyAndView(self):
        data = np.random.rand(10, 20)
        a = tensor(data, chunk_size=6)
        b = a.view()
        b[:5] = 10

        npa = data.copy()
        npb = npa.view()
        npb[:5] = 10

        np.testing.assert_array_equal(b.execute(), npb)
        np.testing.assert_array_equal(a.execute(), npa)

        data = np.random.rand(10, 20)
        a = tensor(data.copy(), chunk_size=6)
        b = a.copy()
        b[:5] = 10

        npa = data.copy()
        npb = npa.copy()
        npb[:5] = 10

        np.testing.assert_array_equal(b.execute(), npb)
        np.testing.assert_array_equal(a.execute(), npa)

        a = tensor(data.copy(), chunk_size=6)
        b = a[:5, :4]
        c = b.copy()
        c[0, 0] = 10

        npa = data.copy()
        npb = npa[:5, :4]
        npc = npb.copy()
        npc[0, 0] =10

        np.testing.assert_array_equal(c.execute(), npc)
        np.testing.assert_array_equal(a.execute(), npa)

    def testFlat(self):
        data = np.random.rand(10, 20)
        a = tensor(data, chunk_size=4)
        fl = a.flat
        fl[1: 10] = 10
        b = fl[10: 20]
        b[0: 4] = 20

        npa = data.copy()
        npfl = npa.flat
        npfl[1: 10] = 10
        npb = npfl[10: 20]
        npb[0: 4] = 20

        np.testing.assert_array_equal(b.execute(), npb)
        np.testing.assert_array_equal(a.execute(), npa)
