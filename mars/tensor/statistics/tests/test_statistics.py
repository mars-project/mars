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

from mars.tensor.datasource import tensor, array
from mars.tensor.statistics import digitize, histogram_bin_edges


class Test(unittest.TestCase):
    def testDigitize(self):
        x = tensor(np.array([0.2, 6.4, 3.0, 1.6]), chunk_size=2)
        bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        inds = digitize(x, bins)

        self.assertEqual(inds.shape, (4,))
        self.assertIsNotNone(inds.dtype)

        inds = inds.tiles()

        self.assertEqual(len(inds.chunks), 2)

    def testHistogramBinEdges(self):
        a = array([0, 0, 0, 1, 2, 3, 3, 4, 5], chunk_size=3)

        with self.assertRaises(ValueError):
            histogram_bin_edges(a, bins='unknown')

        with self.assertRaises(TypeError):
            # bins is str, weights cannot be provided
            histogram_bin_edges(a, bins='scott', weights=a)

        with self.assertRaises(ValueError):
            histogram_bin_edges(a, bins=-1)

        with self.assertRaises(ValueError):
            # not asc
            histogram_bin_edges(a, bins=[3, 2, 1])

        with self.assertRaises(ValueError):
            # bins cannot be 2d
            histogram_bin_edges(a, bins=np.random.rand(2, 3))

        with self.assertRaises(ValueError):
            histogram_bin_edges(a, range=(5, 0))

        with self.assertRaises(ValueError):
            histogram_bin_edges(a, range=(np.nan, np.nan))

        bins = histogram_bin_edges(a, bins=3, range=(0, 5))
        # if range specified, no error will occur
        bins.tiles()
