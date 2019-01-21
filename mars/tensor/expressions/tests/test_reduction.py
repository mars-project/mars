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

from mars.tensor.expressions.datasource import ones, tensor
from mars.operands import MeanCombine, MeanChunk, Mean, Concatenate, Argmax, Argmin,\
    ArgmaxChunk, ArgmaxCombine, ArgminChunk, ArgminCombine
from mars.tensor.expressions.reduction import all
from mars.tests.core import calc_shape


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
            self.assertEqual(calc_shape(res), res.shape)

            res = f(ones((10, 8), chunk_size=3))
            self.assertIsNotNone(res.dtype)
            self.assertEqual(res.shape, ())
            self.assertEqual(calc_shape(res), res.shape)

            res = f(ones((10, 8), chunk_size=3), axis=0)
            self.assertEqual(res.shape, (8,))
            self.assertEqual(calc_shape(res), res.shape)

            res = f(ones((10, 8), chunk_size=3), axis=1)
            self.assertEqual(res.shape, (10,))
            self.assertEqual(calc_shape(res), res.shape)

            res = f(ones((10, 8), chunk_size=3), keepdims=True)
            self.assertEqual(res.shape, (1, 1))
            self.assertEqual(calc_shape(res), res.shape)

            res = f(ones((10, 8), chunk_size=3), axis=0, keepdims=True)
            self.assertEqual(res.shape, (1, 8))
            self.assertEqual(calc_shape(res), res.shape)

            res = f(ones((10, 8), chunk_size=3), axis=1, keepdims=True)
            self.assertEqual(res.shape, (10, 1))
            self.assertEqual(calc_shape(res), res.shape)

            res = f(ones((10, 8, 10), chunk_size=3), axis=1)
            self.assertEqual(res.shape, (10, 10))
            self.assertEqual(calc_shape(res), res.shape)

            res = f(ones((10, 8, 10), chunk_size=3), axis=1, keepdims=True)
            self.assertEqual(res.shape, (10, 1, 10))
            self.assertEqual(calc_shape(res), res.shape)

            res = f(ones((10, 8, 10), chunk_size=3), axis=(0, 2))
            self.assertEqual(res.shape, (8,))
            self.assertEqual(calc_shape(res), res.shape)

            res = f(ones((10, 8, 10), chunk_size=3), axis=(0, 2), keepdims=True)
            self.assertEqual(res.shape, (1, 8, 1))
            self.assertEqual(calc_shape(res), res.shape)

    def testMeanReduction(self):
        mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs).tiles()

        res = mean(ones((10, 8), chunk_size=3))
        self.assertEqual(res.shape, ())
        self.assertIsNotNone(res.dtype)
        self.assertIsInstance(res.chunks[0].op, Mean)
        self.assertIsInstance(res.chunks[0].inputs[0].op, Concatenate)
        self.assertIsInstance(res.chunks[0].inputs[0].inputs[0].op, MeanCombine)

        res = mean(ones((8, 8), chunk_size=8))
        self.assertEqual(res.shape, ())
        self.assertEqual(calc_shape(res), res.shape)

        res = mean(ones((10, 8), chunk_size=3), axis=0)
        self.assertEqual(res.shape, (8,))
        self.assertEqual(calc_shape(res), res.shape)

        res = mean(ones((10, 8), chunk_size=3), axis=1)
        self.assertEqual(res.shape, (10,))
        self.assertEqual(calc_shape(res), res.shape)

        res = mean(ones((10, 8), chunk_size=3), keepdims=True)
        self.assertEqual(res.shape, (1, 1))
        self.assertEqual(calc_shape(res), res.shape)

        res = mean(ones((10, 8), chunk_size=3), axis=0, keepdims=True)
        self.assertEqual(res.shape, (1, 8))
        self.assertEqual(calc_shape(res), res.shape)

        res = mean(ones((10, 8), chunk_size=3), axis=1, keepdims=True)
        self.assertEqual(res.shape, (10, 1))
        self.assertEqual(calc_shape(res), res.shape)
        self.assertIsInstance(res.chunks[0].op, Mean)
        self.assertIsInstance(res.chunks[0].inputs[0].op, Concatenate)
        self.assertIsInstance(res.chunks[0].inputs[0].inputs[0].op, MeanChunk)
        self.assertEqual(calc_shape(res.chunks[0]), res.chunks[0].shape)

    def testArgReduction(self):
        argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs).tiles()
        argmin = lambda x, *args, **kwargs: x.argmin(*args, **kwargs).tiles()

        res1 = argmax(ones((10, 8, 10), chunk_size=3))
        res2 = argmin(ones((10, 8, 10), chunk_size=3))
        self.assertEqual(res1.shape, ())
        self.assertEqual(calc_shape(res1), res1.shape)
        self.assertIsNotNone(res1.dtype)
        self.assertEqual(res2.shape, ())
        self.assertEqual(calc_shape(res2), res2.shape)
        self.assertIsInstance(res1.chunks[0].op, Argmax)
        self.assertIsInstance(res2.chunks[0].op, Argmin)
        self.assertIsInstance(res1.chunks[0].inputs[0].op, Concatenate)
        self.assertIsInstance(res2.chunks[0].inputs[0].op, Concatenate)
        self.assertIsInstance(res1.chunks[0].inputs[0].inputs[0].op, ArgmaxCombine)
        self.assertIsInstance(res2.chunks[0].inputs[0].inputs[0].op, ArgminCombine)
        self.assertEqual(calc_shape(res1.chunks[0]), res1.chunks[0].shape)
        self.assertEqual(calc_shape(res2.chunks[0]), res2.chunks[0].shape)

        res1 = argmax(ones((10, 8), chunk_size=3), axis=1, keepdims=True)
        res2 = argmin(ones((10, 8), chunk_size=3), axis=1, keepdims=True)
        self.assertEqual(res1.shape, (10, 1))
        self.assertEqual(calc_shape(res1), res1.shape)
        self.assertEqual(res2.shape, (10, 1))
        self.assertEqual(calc_shape(res2), res2.shape)
        self.assertIsInstance(res1.chunks[0].op, Argmax)
        self.assertIsInstance(res2.chunks[0].op, Argmin)
        self.assertIsInstance(res1.chunks[0].inputs[0].op, Concatenate)
        self.assertIsInstance(res2.chunks[0].inputs[0].op, Concatenate)
        self.assertIsInstance(res1.chunks[0].inputs[0].inputs[0].op, ArgmaxChunk)
        self.assertIsInstance(res2.chunks[0].inputs[0].inputs[0].op, ArgminChunk)
        self.assertEqual(calc_shape(res1.chunks[0]), res1.chunks[0].shape)
        self.assertEqual(calc_shape(res2.chunks[0]), res2.chunks[0].shape)

        self.assertRaises(TypeError, lambda: argmax(ones((10, 8, 10), chunk_size=3), axis=(0, 1)))
        self.assertRaises(TypeError, lambda: argmin(ones((10, 8, 10), chunk_size=3), axis=(0, 1)))

    def testCumReduction(self):
        cumsum = lambda x, *args, **kwargs: x.cumsum(*args, **kwargs).tiles()
        cumprod = lambda x, *args, **kwargs: x.cumprod(*args, **kwargs).tiles()

        res1 = cumsum(ones((10, 8), chunk_size=3), axis=0)
        res2 = cumprod(ones((10, 8), chunk_size=3), axis=0)
        self.assertEqual(res1.shape, (10, 8))
        self.assertIsNotNone(res1.dtype)
        self.assertEqual(calc_shape(res1), res1.shape)
        self.assertEqual(calc_shape(res1.chunks[0]), res1.chunks[0].shape)
        self.assertEqual(res2.shape, (10, 8))
        self.assertIsNotNone(res2.dtype)
        self.assertEqual(calc_shape(res2), res2.shape)
        self.assertEqual(calc_shape(res2.chunks[0]), res2.chunks[0].shape)

        res1 = cumsum(ones((10, 8, 8), chunk_size=3), axis=1)
        res2 = cumprod(ones((10, 8, 8), chunk_size=3), axis=1)
        self.assertEqual(res1.shape, (10, 8, 8))
        self.assertEqual(calc_shape(res1), res1.shape)
        self.assertEqual(calc_shape(res1.chunks[0]), res1.chunks[0].shape)
        self.assertEqual(res2.shape, (10, 8, 8))
        self.assertEqual(calc_shape(res2), res2.shape)
        self.assertEqual(calc_shape(res2.chunks[0]), res2.chunks[0].shape)

    def testAllReduction(self):
        o = tensor([False])

        with self.assertRaises(ValueError):
            all([-1, 4, 5], out=o)

    def testVarReduction(self):
        var = lambda x, *args, **kwargs: x.var(*args, **kwargs).tiles()

        res1 = var(ones((10, 8), chunk_size=3), ddof=2)
        self.assertEqual(res1.shape, ())
        self.assertEqual(res1.op.ddof, 2)
        self.assertEqual(calc_shape(res1), res1.shape)
        self.assertEqual(calc_shape(res1.chunks[0]), res1.chunks[0].shape)

        res1 = var(ones((10, 8, 8), chunk_size=3), axis=1)
        self.assertEqual(res1.shape, (10, 8))
        self.assertEqual(calc_shape(res1), res1.shape)
        self.assertEqual(calc_shape(res1.chunks[0]), res1.chunks[0].shape)
