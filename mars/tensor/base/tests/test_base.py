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
from mars.tensor.base import atleast_1d, atleast_2d, atleast_3d
from mars.tensor.base import transpose, squeeze, moveaxis
from mars.tensor.datasource import ones, tensor


class Test(unittest.TestCase):
    def testCreateView(self):
        arr = ones((10, 20, 30), chunk_size=[4, 3, 5])
        arr2 = transpose(arr)
        self.assertTrue(arr2.op.create_view)

        arr3 = transpose(arr)
        self.assertTrue(arr3.op.create_view)

        arr4 = arr.swapaxes(0, 1)
        self.assertTrue(arr4.op.create_view)

        arr5 = moveaxis(arr, 1, 0)
        self.assertTrue(arr5.op.create_view)

        arr6 = atleast_1d(1)
        self.assertTrue(arr6.op.create_view)

        arr7 = atleast_2d([1, 1])
        self.assertTrue(arr7.op.create_view)

        arr8 = atleast_3d([1, 1])
        self.assertTrue(arr8.op.create_view)

        arr9 = arr[:3, [1, 2, 3]]
        # no view cuz of fancy indexing
        self.assertFalse(arr9.op.create_view)

        arr10 = arr[:3, None, :5]
        self.assertTrue(arr10.op.create_view)

        data = np.array([[[0], [1], [2]]])
        x = tensor(data)

        t = squeeze(x)
        self.assertTrue(t.op.create_view)

        y = x.reshape(3)
        self.assertTrue(y.op.create_view)
