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
from mars.tensor.statistics import digitize, histogram_bin_edges, quantile
from mars.tensor.statistics.quantile import INTERPOLATION_TYPES


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

    def testQuantile(self):
        raw = np.random.rand(100)
        q = np.random.rand(10)

        for dtype in [np.float32, np.int64, np.complex128]:
            raw2 = raw.astype(dtype)
            a = tensor(raw2, chunk_size=100)

            b = quantile(a, q)
            self.assertEqual(b.shape, (10,))
            self.assertEqual(b.dtype, np.quantile(raw2, q).dtype)

            b = b.tiles()
            self.assertEqual(len(b.chunks), 1)

        raw = np.random.rand(20, 10)
        q = np.random.rand(10)

        for dtype in [np.float32, np.int64, np.complex128]:
            for axis in (None, 0, 1):
                for interpolation in INTERPOLATION_TYPES:
                    for keepdims in [True, False]:
                        raw2 = raw.astype(dtype)
                        a = tensor(raw2, chunk_size=(4, 3))

                        b = quantile(a, q, axis=axis,
                                     interpolation=interpolation, keepdims=keepdims)
                        expected = np.quantile(raw2, q, axis=axis,
                                               interpolation=interpolation,
                                               keepdims=keepdims)
                        self.assertEqual(b.shape, expected.shape)
                        self.assertEqual(b.dtype, expected.dtype)

        a = tensor(raw, chunk_size=10)
        b = quantile(a, q)

        b = b.tiles()
        self.assertEqual(b.shape, (10,))

        # q has to be 1-d
        with self.assertRaises(ValueError):
            quantile(a, q.reshape(5, 2))

        # wrong out type
        with self.assertRaises(TypeError):
            quantile(a, q, out=2)

        # wrong q
        with self.assertRaises(ValueError):
            q2 = q.copy()
            q2[0] = 1.1
            quantile(a, q2)

        # wrong q, with size < 10
        with self.assertRaises(ValueError):
            q2 = np.random.rand(5)
            q2[0] = 1.1
            quantile(a, q2)

        # wrong interpolation
        with self.assertRaises(ValueError):
            quantile(a, q, interpolation='unknown')
