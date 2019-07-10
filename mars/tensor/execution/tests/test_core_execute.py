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
from mars.tensor import ones, add, swapaxes
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
        a = ones((10, 20), chunk_size=6)
        b = a[:5, 5:10]
        b[:3, :3] = 3

        npa = np.ones((10, 20))
        npb = npa[:5, 5:10]
        npb[:3, :3] = 3

        np.testing.assert_array_equal(b.execute(), npb)
        np.testing.assert_array_equal(a.execute(), npa)

    def testViewDataOnTranspose(self):
        a = ones((10, 20), chunk_size=6)
        b = a.T
        add(b, 1, out=b)

        np.testing.assert_array_equal(b.execute(), np.ones((20, 10)) + 1)
        np.testing.assert_array_equal(a.execute(), np.ones((10, 20)) + 1)

    def testViewDataOnSwapaxes(self):
        a = ones((10, 20), chunk_size=6)
        b = swapaxes(a, 1, 0)
        a[1] = 10

        npa = np.ones((10, 20))
        npb = np.swapaxes(npa, 1, 0)
        npa[1] = 10

        np.testing.assert_array_equal(b.execute(), npb)
        np.testing.assert_array_equal(a.execute(), npa)
