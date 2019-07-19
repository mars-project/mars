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

from mars.tensor.datasource import ones
from mars.tensor import fft


class Test(unittest.TestCase):
    def testStandardFFT(self):
        t = ones((10, 20, 30), chunk_size=(3, 20, 30))

        t1 = fft.fft(t)
        self.assertEqual(t1.shape, (10, 20, 30))
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

        t = ones((10, 20, 30), chunk_size=(3, 20, 30))

        t1 = fft.ifft(t)
        self.assertEqual(t1.shape, (10, 20, 30))
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

        t = ones((10, 20, 30), chunk_size=(3, 20, 30))

        t1 = fft.fft2(t, s=(23, 21))
        self.assertEqual(t1.shape, (10, 23, 21))
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

        t = ones((10, 20, 30), chunk_size=(3, 20, 30))

        t1 = fft.ifft2(t, s=(11, 9), axes=(1, 2))
        self.assertEqual(t1.shape, (10, 11, 9))
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

        t = ones((10, 20, 30), chunk_size=(3, 20, 30))

        t1 = fft.fftn(t, s=(11, 9), axes=(1, 2))
        self.assertEqual(t1.shape, (10, 11, 9))
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

        t = ones((10, 20, 30), chunk_size=(3, 20, 30))

        t1 = fft.ifftn(t, s=(11, 9), axes=(1, 2))
        self.assertEqual(t1.shape, (10, 11, 9))
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

    def testRealFFT(self):
        t = ones((10, 20, 30), chunk_size=(3, 20, 30))

        t1 = fft.rfft(t)
        self.assertEqual(t1.shape, np.fft.rfft(np.ones(t.shape)).shape)
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

        t = ones((10, 20, 30), chunk_size=(3, 20, 30))

        t1 = fft.irfft(t)
        self.assertEqual(t1.shape, np.fft.irfft(np.ones(t.shape)).shape)
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

        t = ones((10, 20, 30), chunk_size=(3, 20, 30))

        t1 = fft.rfft2(t, s=(23, 21))
        self.assertEqual(t1.shape, np.fft.rfft2(np.ones(t.shape), s=(23, 21)).shape)
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

        t = ones((10, 20, 30), chunk_size=(3, 20, 30))

        t1 = fft.irfft2(t, s=(11, 9), axes=(1, 2))
        self.assertEqual(t1.shape, np.fft.irfft2(np.ones(t.shape), s=(11, 9), axes=(1, 2)).shape)
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

        t = ones((10, 20, 30), chunk_size=(3, 20, 30))

        t1 = fft.rfftn(t, s=(11, 30), axes=(1, 2))
        self.assertEqual(t1.shape, np.fft.rfftn(np.ones(t.shape), s=(11, 30), axes=(1, 2)).shape)
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

        t = ones((10, 20, 30), chunk_size=(3, 20, 30))

        t1 = fft.irfftn(t, s=(11, 9), axes=(1, 2))
        self.assertEqual(t1.shape, np.fft.irfftn(np.ones(t.shape), s=(11, 9), axes=(1, 2)).shape)
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

    def testHermitianFFT(self):
        t = ones((10, 20, 30), chunk_size=(3, 20, 30))

        t1 = fft.hfft(t)
        self.assertEqual(t1.shape, np.fft.hfft(np.ones(t.shape)).shape)
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

        t = ones((10, 20, 30), chunk_size=(3, 20, 30))

        t1 = fft.hfft(t, n=100)
        self.assertEqual(t1.shape, np.fft.hfft(np.ones(t.shape), n=100).shape)
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

        t = ones((10, 20, 30), chunk_size=(3, 20, 30))

        t1 = fft.ihfft(t)
        self.assertEqual(t1.shape, np.fft.ihfft(np.ones(t.shape)).shape)
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

        t = ones((10, 20, 30), chunk_size=(3, 20, 30))

        t1 = fft.ihfft(t, n=100)
        self.assertEqual(t1.shape, np.fft.ihfft(np.ones(t.shape), n=100).shape)
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

        t1 = fft.ihfft(t, n=101)
        self.assertEqual(t1.shape, np.fft.ihfft(np.ones(t.shape), n=101).shape)
        t1.tiles()
        self.assertEqual(t1.shape, tuple(sum(ns) for ns in t1.nsplits))

    def testFFTShift(self):
        freqs = fft.fftfreq(9, d=1./9).reshape(3, 3)
        t = fft.ifftshift(fft.fftshift(freqs))

        self.assertIsNotNone(t.dtype)
        expect_dtype = np.fft.ifftshift(np.fft.fftshift(np.fft.fftfreq(9, d=1./9).reshape(3, 3))).dtype
        self.assertEqual(t.dtype, expect_dtype)

    def testFFTFreq(self):
        t = fft.fftfreq(10, .1, chunk_size=3)

        self.assertEqual(t.shape, np.fft.fftfreq(10, .1).shape)
        t.tiles()
        self.assertEqual(t.shape, tuple(sum(ns) for ns in t.nsplits))

        t = fft.rfftfreq(10, .1, chunk_size=3)

        self.assertEqual(t.shape, np.fft.rfftfreq(10, .1).shape)
        t.tiles()
        self.assertEqual(t.shape, tuple(sum(ns) for ns in t.nsplits))
