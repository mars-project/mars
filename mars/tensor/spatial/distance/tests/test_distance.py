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

from mars.tensor.datasource import tensor
from mars.tensor.spatial import distance


class Test(unittest.TestCase):
    def testPdist(self):
        raw = np.random.rand(100, 10)

        # test 1 chunk
        a = tensor(raw, chunk_size=100)
        dist = distance.pdist(a)
        self.assertEqual(dist.shape, (100 * 99 // 2,))

        dist = dist.tiles()
        self.assertEqual(len(dist.chunks), 1)
        for c in dist.chunks:
            self.assertEqual(c.shape, (dist.shape[0],))

        # test multiple chunks
        a = tensor(raw, chunk_size=15)
        dist = distance.pdist(a, aggregate_size=2)
        self.assertEqual(dist.shape, (100 * 99 // 2,))

        dist = dist.tiles()
        self.assertEqual(len(dist.chunks), 2)
        for c in dist.chunks:
            self.assertEqual(c.shape, (dist.shape[0] // 2,))

        # X can only be 2-d
        with self.assertRaises(ValueError):
            distance.pdist(np.random.rand(3, 3, 3))

        # out type wrong
        with self.assertRaises(TypeError):
            distance.pdist(np.random.rand(3, 3), out=2)

        # out shape wrong
        with self.assertRaises(ValueError):
            distance.pdist(np.random.rand(3, 3),
                           out=tensor(np.random.rand(2)))

        # out dtype wrong
        with self.assertRaises(ValueError):
            distance.pdist(np.random.rand(3, 3),
                           out=tensor(np.random.randint(2, size=(3,))))

        # test extra param
        with self.assertRaises(TypeError):
            distance.pdist(np.random.rand(3, 3), unknown_kw='unknown_kw')
