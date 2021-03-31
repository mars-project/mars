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

from mars.core.operand import OperandStage
from mars.tensor.datasource import ones
from mars.tensor.reshape.reshape import TensorReshape


class Test(unittest.TestCase):
    def testReshape(self):
        a = ones((10, 20, 30), chunk_size=5)
        b = a.reshape(10, 600)

        b = b.tiles()

        self.assertEqual(tuple(sum(s) for s in b.nsplits), (10, 600))

        a = ones((10, 600), chunk_size=5)
        b = a.reshape(10, 30, 20)

        b = b.tiles()

        self.assertEqual(tuple(sum(s) for s in b.nsplits), (10, 30, 20))

        a = ones((10, 600), chunk_size=5)
        a.shape = [10, 30, 20]

        a = a.tiles()

        self.assertEqual(tuple(sum(s) for s in a.nsplits), (10, 30, 20))

        # test reshape unknown shape
        c = a[a > 0]
        d = c.reshape(10, 600)
        self.assertEqual(d.shape, (10, 600))
        d = c.reshape(-1, 10)
        self.assertEqual(len(d.shape), 2)
        self.assertTrue(np.isnan(d.shape[0]))
        self.assertTrue(d.shape[1], 10)

        with self.assertRaises(TypeError):
            a.reshape((10, 30, 20), other_argument=True)

    def testShuffleReshape(self):
        a = ones((31, 27), chunk_size=10)
        b = a.reshape(27, 31)
        b.op.extra_params['_reshape_with_shuffle'] = True

        b = b.tiles()

        self.assertEqual(tuple(sum(s) for s in b.nsplits), (27, 31))
        self.assertIsInstance(b.chunks[0].op, TensorReshape)
        self.assertEqual(b.chunks[0].op.stage, OperandStage.reduce)

        shuffle_map_sample = b.chunks[0].inputs[0].inputs[0]
        self.assertIsInstance(shuffle_map_sample.op, TensorReshape)
        self.assertEqual(shuffle_map_sample.op.stage, OperandStage.map)
