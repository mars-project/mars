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

from mars.tensor.datasource import ones
from mars.tensor.base import broadcast_to


class Test(unittest.TestCase):
    def testBroadcastTo(self):
        arr = ones((10, 5), chunk_size=2)
        arr2 = broadcast_to(arr, (20, 10, 5))
        arr2.tiles()

        self.assertEqual(arr2.shape, (20, 10, 5))
        self.assertEqual(len(arr2.chunks), len(arr.chunks))
        self.assertEqual(arr2.chunks[0].shape, (20, 2, 2))

        arr = ones((10, 5, 1), chunk_size=2)
        arr3 = broadcast_to(arr, (5, 10, 5, 6))
        arr3.tiles()

        self.assertEqual(arr3.shape, (5, 10, 5, 6))
        self.assertEqual(len(arr3.chunks), len(arr.chunks))
        self.assertEqual(arr3.nsplits, ((5,), (2, 2, 2, 2, 2), (2, 2, 1), (6,)))
        self.assertEqual(arr3.chunks[0].shape, (5, 2, 2, 6))

        arr = ones((10, 1), chunk_size=2)
        arr4 = broadcast_to(arr, (20, 10, 5))
        arr4.tiles()

        self.assertEqual(arr4.shape, (20, 10, 5))
        self.assertEqual(len(arr4.chunks), len(arr.chunks))
        self.assertEqual(arr4.chunks[0].shape, (20, 2, 5))

        with self.assertRaises(ValueError):
            broadcast_to(arr, (10,))

        with self.assertRaises(ValueError):
            broadcast_to(arr, (5, 1))