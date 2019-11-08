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

from mars.tensor.datasource import ones, tensor
from mars.tensor.merge import TensorConcatenate
from mars.tensor.reduction import all, TensorMean, TensorMeanChunk, TensorMeanCombine, \
    TensorArgmax, TensorArgmaxMap, TensorArgmaxCombine, TensorArgmin, TensorArgminMap, TensorArgminCombine


class Test(unittest.TestCase):
    def testBaseReduction(self):
        sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs).tiles()
        prod = lambda x, *args, **kwargs: x.prod(*args, **kwargs).tiles()
        max = lambda x, *args, **kwargs: x.max(*args, **kwargs).tiles()
        min = lambda x, *args, **kwargs: x.min(*args, **kwargs).tiles()
        all = lambda x, *args, **kwargs: x.all(*args, **kwargs).tiles()
        any = lambda x, *args, **kwargs: x.any(*args, **kwargs).tiles()

        for f in [sum, prod, max, min, all, any]:
            res = f(ones((8, 8), chunk_size=8))
            self.assertEqual(res.shape, ())

            res = f(ones((10, 8), chunk_size=3))
            self.assertIsNotNone(res.dtype)
            self.assertEqual(res.shape, ())

            res = f(ones((10, 8), chunk_size=3), axis=0)
            self.assertEqual(res.shape, (8,))

            res = f(ones((10, 8), chunk_size=3), axis=1)
            self.assertEqual(res.shape, (10,))

            with self.assertRaises(np.AxisError):
                f(ones((10, 8), chunk_size=3), axis=2)

            res = f(ones((10, 8), chunk_size=3), axis=-1)
            self.assertEqual(res.shape, (10,))

            with self.assertRaises(np.AxisError):
                f(ones((10, 8), chunk_size=3), axis=-3)

            res = f(ones((10, 8), chunk_size=3), keepdims=True)
            self.assertEqual(res.shape, (1, 1))

            res = f(ones((10, 8), chunk_size=3), axis=0, keepdims=True)
            self.assertEqual(res.shape, (1, 8))

            res = f(ones((10, 8), chunk_size=3), axis=1, keepdims=True)
            self.assertEqual(res.shape, (10, 1))

            res = f(ones((10, 8, 10), chunk_size=3), axis=1)
            self.assertEqual(res.shape, (10, 10))

            res = f(ones((10, 8, 10), chunk_size=3), axis=1, keepdims=True)
            self.assertEqual(res.shape, (10, 1, 10))

            res = f(ones((10, 8, 10), chunk_size=3), axis=(0, 2))
            self.assertEqual(res.shape, (8,))

            res = f(ones((10, 8, 10), chunk_size=3), axis=(0, 2), keepdims=True)
            self.assertEqual(res.shape, (1, 8, 1))

    def testMeanReduction(self):
        mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs).tiles()

        res = mean(ones((10, 8), chunk_size=3))
        self.assertEqual(res.shape, ())
        self.assertIsNotNone(res.dtype)
        self.assertIsInstance(res.chunks[0].op, TensorMean)
        self.assertIsInstance(res.chunks[0].inputs[0].op, TensorConcatenate)
        self.assertIsInstance(res.chunks[0].inputs[0].inputs[0].op, TensorMeanCombine)

        res = mean(ones((8, 8), chunk_size=8))
        self.assertEqual(res.shape, ())

        res = mean(ones((10, 8), chunk_size=3), axis=0)
        self.assertEqual(res.shape, (8,))

        res = mean(ones((10, 8), chunk_size=3), axis=1)
        self.assertEqual(res.shape, (10,))

        with self.assertRaises(np.AxisError):
            mean(ones((10, 8), chunk_size=3), axis=2)

        res = mean(ones((10, 8), chunk_size=3), axis=-1)
        self.assertEqual(res.shape, (10,))

        with self.assertRaises(np.AxisError):
            mean(ones((10, 8), chunk_size=3), axis=-3)

        res = mean(ones((10, 8), chunk_size=3), keepdims=True)
        self.assertEqual(res.shape, (1, 1))

        res = mean(ones((10, 8), chunk_size=3), axis=0, keepdims=True)
        self.assertEqual(res.shape, (1, 8))

        res = mean(ones((10, 8), chunk_size=3), axis=1, keepdims=True)
        self.assertEqual(res.shape, (10, 1))
        self.assertIsInstance(res.chunks[0].op, TensorMean)
        self.assertIsInstance(res.chunks[0].inputs[0].op, TensorConcatenate)
        self.assertIsInstance(res.chunks[0].inputs[0].inputs[0].op, TensorMeanChunk)

    def testArgReduction(self):
        argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs).tiles()
        argmin = lambda x, *args, **kwargs: x.argmin(*args, **kwargs).tiles()

        res1 = argmax(ones((10, 8, 10), chunk_size=3))
        res2 = argmin(ones((10, 8, 10), chunk_size=3))
        self.assertEqual(res1.shape, ())
        self.assertIsNotNone(res1.dtype)
        self.assertEqual(res2.shape, ())
        self.assertIsInstance(res1.chunks[0].op, TensorArgmax)
        self.assertIsInstance(res2.chunks[0].op, TensorArgmin)
        self.assertIsInstance(res1.chunks[0].inputs[0].op, TensorConcatenate)
        self.assertIsInstance(res2.chunks[0].inputs[0].op, TensorConcatenate)
        self.assertIsInstance(res1.chunks[0].inputs[0].inputs[0].op, TensorArgmaxCombine)
        self.assertIsInstance(res2.chunks[0].inputs[0].inputs[0].op, TensorArgminCombine)

        res1 = argmax(ones((10, 8), chunk_size=3), axis=1)
        res2 = argmin(ones((10, 8), chunk_size=3), axis=1)
        self.assertEqual(res1.shape, (10,))
        self.assertEqual(res2.shape, (10,))
        self.assertIsInstance(res1.chunks[0].op, TensorArgmax)
        self.assertIsInstance(res2.chunks[0].op, TensorArgmin)
        self.assertIsInstance(res1.chunks[0].inputs[0].op, TensorConcatenate)
        self.assertIsInstance(res2.chunks[0].inputs[0].op, TensorConcatenate)
        self.assertIsInstance(res1.chunks[0].inputs[0].inputs[0].op, TensorArgmaxMap)
        self.assertIsInstance(res2.chunks[0].inputs[0].inputs[0].op, TensorArgminMap)

        self.assertRaises(TypeError, lambda: argmax(ones((10, 8, 10), chunk_size=3), axis=(0, 1)))
        self.assertRaises(TypeError, lambda: argmin(ones((10, 8, 10), chunk_size=3), axis=(0, 1)))
        self.assertRaises(np.AxisError, lambda: argmin(ones((10, 8, 10), chunk_size=3), axis=3))
        self.assertRaises(np.AxisError, lambda: argmin(ones((10, 8, 10), chunk_size=3), axis=-4))

    def testCumReduction(self):
        cumsum = lambda x, *args, **kwargs: x.cumsum(*args, **kwargs).tiles()
        cumprod = lambda x, *args, **kwargs: x.cumprod(*args, **kwargs).tiles()

        res1 = cumsum(ones((10, 8), chunk_size=3), axis=0)
        res2 = cumprod(ones((10, 8), chunk_size=3), axis=0)
        self.assertEqual(res1.shape, (10, 8))
        self.assertIsNotNone(res1.dtype)
        self.assertEqual(res2.shape, (10, 8))
        self.assertIsNotNone(res2.dtype)

        res1 = cumsum(ones((10, 8, 8), chunk_size=3), axis=1)
        res2 = cumprod(ones((10, 8, 8), chunk_size=3), axis=1)
        self.assertEqual(res1.shape, (10, 8, 8))
        self.assertEqual(res2.shape, (10, 8, 8))

        res1 = cumsum(ones((10, 8, 8), chunk_size=3), axis=-2)
        res2 = cumprod(ones((10, 8, 8), chunk_size=3), axis=-2)
        self.assertEqual(res1.shape, (10, 8, 8))
        self.assertEqual(res2.shape, (10, 8, 8))

        with self.assertRaises(np.AxisError):
            cumsum(ones((10, 8), chunk_size=3), axis=2)
        with self.assertRaises(np.AxisError):
            cumsum(ones((10, 8), chunk_size=3), axis=-3)

    def testAllReduction(self):
        o = tensor([False])

        with self.assertRaises(ValueError):
            all([-1, 4, 5], out=o)

    def testVarReduction(self):
        var = lambda x, *args, **kwargs: x.var(*args, **kwargs).tiles()

        res1 = var(ones((10, 8), chunk_size=3), ddof=2)
        self.assertEqual(res1.shape, ())
        self.assertEqual(res1.op.ddof, 2)

        res1 = var(ones((10, 8, 8), chunk_size=3), axis=1)
        self.assertEqual(res1.shape, (10, 8))
