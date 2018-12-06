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


import numpy as np

from mars.compat import unittest
from mars.tensor.execution.core import Executor
from mars.tensor.expressions.datasource import tensor
from mars.tensor.expressions.fft import fft, ifft, fft2, ifft2, fftn, ifftn, rfft, irfft, rfft2, irfft2, \
    rfftn, irfftn, hfft, ihfft, fftfreq, rfftfreq, fftshift, ifftshift


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')

    def testFFTExecution(self):
        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(4, 4, 30))

        r = fft(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fft(raw)
        self.assertTrue(np.allclose(res, expected))

        r = fft(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fft(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = fft(t, n=11)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fft(raw, n=11)
        self.assertTrue(np.allclose(res, expected))

        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(4, 4, 4))

        r = fft(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fft(raw)
        self.assertTrue(np.allclose(res, expected))

        r = fft(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fft(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = fft(t, n=11)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fft(raw, n=11)
        self.assertTrue(np.allclose(res, expected))

    def testIFFTExecution(self):
        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(4, 4, 30))

        r = ifft(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifft(raw)
        self.assertTrue(np.allclose(res, expected))

        r = ifft(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifft(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = ifft(t, n=11)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifft(raw, n=11)
        self.assertTrue(np.allclose(res, expected))

        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(4, 4, 5))

        r = ifft(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifft(raw)
        self.assertTrue(np.allclose(res, expected))

        r = ifft(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifft(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = ifft(t, n=11)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifft(raw, n=11)
        self.assertTrue(np.allclose(res, expected))

    def testFFT2Execution(self):
        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(4, 20, 30))

        r = fft2(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fft2(raw)
        self.assertTrue(np.allclose(res, expected))

        r = fft2(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fft2(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = fft2(t, s=(11, 12))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fft2(raw, s=(11, 12))
        self.assertTrue(np.allclose(res, expected))

        r = fft2(t, s=(11, 12), axes=(-1, -2))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fft2(raw, s=(11, 12), axes=(-1, -2))
        self.assertTrue(np.allclose(res, expected))

        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(4, 5, 6))

        r = fft2(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fft2(raw)
        self.assertTrue(np.allclose(res, expected))

        r = fft2(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fft2(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = fft2(t, s=(11, 12))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fft2(raw, s=(11, 12))
        self.assertTrue(np.allclose(res, expected))

        r = fft2(t, s=(11, 12), axes=(-1, -2))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fft2(raw, s=(11, 12), axes=(-1, -2))
        self.assertTrue(np.allclose(res, expected))

    def testIFFT2Execution(self):
        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(4, 20, 30))

        r = ifft2(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifft2(raw)
        self.assertTrue(np.allclose(res, expected))

        r = ifft2(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifft2(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = ifft2(t, s=(11, 12))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifft2(raw, s=(11, 12))
        self.assertTrue(np.allclose(res, expected))

        r = ifft2(t, s=(11, 12), axes=(-1, -2))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifft2(raw, s=(11, 12), axes=(-1, -2))
        self.assertTrue(np.allclose(res, expected))

        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(4, 3, 5))

        r = ifft2(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifft2(raw)
        self.assertTrue(np.allclose(res, expected))

        r = ifft2(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifft2(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = ifft2(t, s=(11, 12))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifft2(raw, s=(11, 12))
        self.assertTrue(np.allclose(res, expected))

        r = ifft2(t, s=(11, 12), axes=(-1, -2))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifft2(raw, s=(11, 12), axes=(-1, -2))
        self.assertTrue(np.allclose(res, expected))

    def testFFTNExecution(self):
        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(10, 20, 30))

        r = fftn(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fftn(raw)
        self.assertTrue(np.allclose(res, expected))

        r = fftn(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fftn(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = fftn(t, s=(11, 12, 5))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fftn(raw, s=(11, 12, 5))
        self.assertTrue(np.allclose(res, expected))

        r = fftn(t, s=(11, 12, 5), axes=(-1, -2, -3))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fftn(raw, s=(11, 12, 5), axes=(-1, -2, -3))
        self.assertTrue(np.allclose(res, expected))

        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(3, 3, 4))

        r = fftn(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fftn(raw)
        self.assertTrue(np.allclose(res, expected))

        r = fftn(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fftn(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = fftn(t, s=(11, 12, 5))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fftn(raw, s=(11, 12, 5))
        self.assertTrue(np.allclose(res, expected))

        r = fftn(t, s=(11, 12, 5), axes=(-1, -2, -3))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fftn(raw, s=(11, 12, 5), axes=(-1, -2, -3))
        self.assertTrue(np.allclose(res, expected))

    def testIFFTNExecution(self):
        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(10, 20, 30))

        r = ifftn(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifftn(raw)
        self.assertTrue(np.allclose(res, expected))

        r = ifftn(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifftn(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = ifftn(t, s=(11, 12, 5))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifftn(raw, s=(11, 12, 5))
        self.assertTrue(np.allclose(res, expected))

        r = ifftn(t, s=(11, 12, 5), axes=(-1, -2, -3))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifftn(raw, s=(11, 12, 5), axes=(-1, -2, -3))
        self.assertTrue(np.allclose(res, expected))

        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(3, 4, 7))

        r = ifftn(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifftn(raw)
        self.assertTrue(np.allclose(res, expected))

        r = ifftn(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifftn(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = ifftn(t, s=(11, 12, 5))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifftn(raw, s=(11, 12, 5))
        self.assertTrue(np.allclose(res, expected))

        r = ifftn(t, s=(11, 12, 5), axes=(-1, -2, -3))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifftn(raw, s=(11, 12, 5), axes=(-1, -2, -3))
        self.assertTrue(np.allclose(res, expected))

    def testRFFTExecution(self):
        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(4, 4, 30))

        r = rfft(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.rfft(raw)
        self.assertTrue(np.allclose(res, expected))

        r = rfft(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.rfft(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = rfft(t, n=11)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.rfft(raw, n=11)
        self.assertTrue(np.allclose(res, expected))

    def testIRFFTExecution(self):
        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(4, 4, 30))

        r = irfft(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.irfft(raw)
        self.assertTrue(np.allclose(res, expected))

        r = irfft(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.irfft(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = irfft(t, n=11)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.irfft(raw, n=11)
        self.assertTrue(np.allclose(res, expected))

    def testRFFT2Execution(self):
        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(4, 20, 30))

        r = rfft2(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.rfft2(raw)
        self.assertTrue(np.allclose(res, expected))

        r = rfft2(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.rfft2(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = rfft2(t, s=(11, 12))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.rfft2(raw, s=(11, 12))
        self.assertTrue(np.allclose(res, expected))

        r = rfft2(t, s=(11, 12), axes=(-1, -2))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.rfft2(raw, s=(11, 12), axes=(-1, -2))
        self.assertTrue(np.allclose(res, expected))

    def testIRFFT2Execution(self):
        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(4, 20, 30))

        r = irfft2(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.irfft2(raw)
        self.assertTrue(np.allclose(res, expected))

        r = irfft2(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.irfft2(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = irfft2(t, s=(11, 12))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.irfft2(raw, s=(11, 12))
        self.assertTrue(np.allclose(res, expected))

        r = irfft2(t, s=(11, 12), axes=(-1, -2))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.irfft2(raw, s=(11, 12), axes=(-1, -2))
        self.assertTrue(np.allclose(res, expected))

    def testRFFTNExecution(self):
        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(10, 20, 30))

        r = rfftn(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.rfftn(raw)
        self.assertTrue(np.allclose(res, expected))

        r = rfftn(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.rfftn(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = rfftn(t, s=(11, 12, 5))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.rfftn(raw, s=(11, 12, 5))
        self.assertTrue(np.allclose(res, expected))

        r = rfftn(t, s=(11, 12, 11), axes=(-1, -2, -3))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.rfftn(raw, s=(11, 12, 11), axes=(-1, -2, -3))
        self.assertTrue(np.allclose(res, expected))

    def testIRFFTNExecution(self):
        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(10, 20, 30))

        r = irfftn(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.irfftn(raw)
        self.assertTrue(np.allclose(res, expected))

        r = irfftn(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.irfftn(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = irfftn(t, s=(11, 21, 5))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.irfftn(raw, s=(11, 21, 5))
        self.assertTrue(np.allclose(res, expected))

        r = irfftn(t, s=(11, 21, 30), axes=(-1, -2, -3))
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.irfftn(raw, s=(11, 21, 30), axes=(-1, -2, -3))
        self.assertTrue(np.allclose(res, expected))

    def testHFFTExecution(self):
        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(4, 4, 30))

        r = hfft(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.hfft(raw)
        self.assertTrue(np.allclose(res, expected))

        r = hfft(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.hfft(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = hfft(t, n=11)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.hfft(raw, n=11)
        self.assertTrue(np.allclose(res, expected))

    def testIHFFTExecution(self):
        raw = np.random.rand(10, 20, 30)
        t = tensor(raw, chunks=(4, 4, 30))

        r = ihfft(t)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ihfft(raw)
        self.assertTrue(np.allclose(res, expected))

        r = ihfft(t, norm='ortho')
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ihfft(raw, norm='ortho')
        self.assertTrue(np.allclose(res, expected))

        r = ihfft(t, n=11)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ihfft(raw, n=11)
        self.assertTrue(np.allclose(res, expected))

        r = ihfft(t, n=12)
        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ihfft(raw, n=12)
        self.assertTrue(np.allclose(res, expected))

    def testFFTFreqExecution(self):
        t = fftfreq(10, .1, chunks=3)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, np.fft.fftfreq(10, .1)))

        t = fftfreq(11, .01, chunks=3)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, np.fft.fftfreq(11, .01)))

    def testRFFTFreqExecution(self):
        t = rfftfreq(20, .1, chunks=3)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, np.fft.rfftfreq(20, .1)))

        t = rfftfreq(21, .01, chunks=3)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertTrue(np.allclose(res, np.fft.rfftfreq(21, .01)))

    def testFFTShiftExecution(self):
        t = fftfreq(10, .1, chunks=3)
        r = fftshift(t)

        res = self.executor.execute_tensor(r, concat=True)[0]
        self.assertTrue(np.allclose(res, np.fft.fftshift(np.fft.fftfreq(10, .1))))

        freqs = fftfreq(9, d=1./9, chunks=2).reshape(3, 3)
        r = fftshift(freqs, axes=(1,))

        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.fftshift(np.fft.fftfreq(9, d=1./9).reshape(3, 3), axes=(1,))
        self.assertTrue(np.allclose(res, expected))

    def testIFFTShiftExecution(self):
        t = fftfreq(9, d=1./9, chunks=2).reshape(3, 3)
        r = ifftshift(t)

        res = self.executor.execute_tensor(r, concat=True)[0]
        expected = np.fft.ifftshift(np.fft.fftfreq(9, d=1./9).reshape(3, 3))
        self.assertTrue(np.allclose(res, expected))
