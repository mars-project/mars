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

from mars.tests.core import TestBase
from mars.tensor.datasource import tensor
from mars.tensor import einsum


class Test(TestBase):
    def testEinsum(self):
        data1 = np.random.rand(3, 4, 5)
        data2 = np.random.rand(4, 3, 2)

        t1 = tensor(data1, chunk_size=2)
        t2 = tensor(data2, chunk_size=3)
        t = einsum('ijk, jil -> kl', t1, t2)

        self.assertEqual(t.shape, (5, 2))

        t = t.tiles()
        self.assertEqual(len(t.chunks), 3)

        # multiply(data1, data2)
        data1 = np.random.rand(6, 6)
        data2 = np.random.rand(6, 6)
        t1 = tensor(data1, chunk_size=3)
        t2 = tensor(data2, chunk_size=3)
        t = einsum('..., ...', t1, t2)

        self.assertEqual(t.shape, (6, 6))

        t = t.tiles()
        self.assertEqual(len(t.chunks), 4)

        t = einsum('..., ...', t1, t2, optimize=True)
        self.assertEqual(t.op.optimize, ['einsum_path', (0, 1)])

        # test broadcast
        data1 = np.random.rand(1, 10, 9)
        data2 = np.random.rand(9, 6)
        data3 = np.random.rand(10, 6)
        data4 = np.random.rand(8,)

        t1 = tensor(data1, chunk_size=(1, (5, 5), (3, 3, 3)))
        t2 = tensor(data2, chunk_size=((3, 3, 3), (3, 3)))
        t3 = tensor(data3, chunk_size=((6, 4), (4, 2)))
        t4 = tensor(data4, chunk_size=3)
        t = einsum('ajk,kl,jl,a->a', t1, t2, t3, t4, optimize='')

        self.assertEqual(t.shape, (8,))

        t = t.tiles()
        self.assertEqual(len(t.chunks), 3)

