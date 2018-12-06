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


from mars.compat import unittest
from mars.tensor.expressions.datasource import ones
from mars.tensor.expressions.indexing.slice import TensorSlice
from mars.tensor.expressions.rechunk.rechunk import compute_rechunk


class Test(unittest.TestCase):
    def testComputeRechunk(self):
        tensor = ones((12, 8), chunks=((4, 4, 3, 1), (3, 3, 2)))
        tensor.tiles()
        new_tensor = compute_rechunk(tensor, ((9, 2, 1), (2, 1, 4, 1)))

        self.assertEqual(len(new_tensor.chunks), 12)
        self.assertEqual(len(new_tensor.chunks[0].inputs), 3)
        self.assertIsInstance(new_tensor.chunks[0].inputs[0].op, TensorSlice)
        self.assertIs(new_tensor.chunks[0].inputs[0].inputs[0], tensor.chunks[0].data)
        self.assertEqual(new_tensor.chunks[0].inputs[0].op.slices,
                         (slice(None, None, None), slice(None, 2, None)))
        self.assertIs(new_tensor.chunks[0].inputs[1].inputs[0], tensor.chunks[3].data)
        self.assertEqual(new_tensor.chunks[0].inputs[1].op.slices,
                         (slice(None, None, None), slice(None, 2, None)))
        self.assertIs(new_tensor.chunks[0].inputs[2].inputs[0], tensor.chunks[6].data)
        self.assertEqual(new_tensor.chunks[0].inputs[2].op.slices,
                         (slice(None, 1, None), slice(None, 2, None)))
        self.assertIs(new_tensor.chunks[-1].inputs[0], tensor.chunks[-1].data)
        self.assertEqual(new_tensor.chunks[-1].op.slices,
                         (slice(None, None, None), slice(1, None, None)))

    def testRechunk(self):
        tensor = ones((12, 9), chunks=4)
        new_tensor = tensor.rechunk(3)
        new_tensor.tiles()

        self.assertEqual(len(new_tensor.chunks), 12)
        self.assertEqual(new_tensor.chunks[0].inputs[0], tensor.chunks[0].data)
        self.assertEqual(len(new_tensor.chunks[1].inputs), 2)
        self.assertEqual(new_tensor.chunks[1].inputs[0].op.slices,
                         (slice(None, 3, None), slice(3, None, None)))
        self.assertEqual(new_tensor.chunks[1].inputs[1].op.slices,
                         (slice(None, 3, None), slice(None, 2, None)))
        self.assertEqual(len(new_tensor.chunks[-1].inputs), 2)
        self.assertEqual(new_tensor.chunks[-1].inputs[0].op.slices,
                         (slice(1, None, None), slice(2, None, None)))
        self.assertEqual(new_tensor.chunks[-1].inputs[1].op.slices,
                         (slice(1, None, None), slice(None, None, None)))
