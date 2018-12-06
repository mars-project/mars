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

import numpy as np

import mars.tensor as mt
from mars.tests.core import TestBase


class Test(TestBase):
    def testSessionExecute(self):
        a = mt.random.rand(10, 20)
        res = a.sum().execute()
        self.assertTrue(np.isscalar(res))
        self.assertLess(res, 200)

        data = np.random.random((5, 9))

        # test multiple outputs
        arr1 = mt.tensor(data.copy(), chunks=3)
        result = mt.modf(arr1).execute()
        expected = np.modf(data)

        np.testing.assert_array_equal(result[0], expected[0])
        np.testing.assert_array_equal(result[1], expected[1])

        # test 1 output
        arr2 = mt.tensor(data.copy(), chunks=3)
        result = ((arr2 + 1) * 2).execute()
        expected = (data + 1) * 2

        np.testing.assert_array_equal(result, expected)

        # test multiple outputs, but only execute 1
        arr3 = mt.tensor(data.copy(), chunks=3)
        arrs = mt.split(arr3, 3, axis=1)
        result = arrs[0].execute()
        expected = np.split(data, 3, axis=1)[0]

        np.testing.assert_array_equal(result, expected)
