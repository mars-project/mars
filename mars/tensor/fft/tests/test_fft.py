#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2021 Alibaba Group Holding Ltd.
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

from ....core import tile
from ...datasource import ones
from .. import (
    fft,
    ifft,
    fft2,
    ifft2,
    fftn,
    ifftn,
    rfft,
    irfft,
    rfft2,
    irfft2,
    rfftn,
    irfftn,
    hfft,
    ihfft,
    fftfreq,
    rfftfreq,
    fftshift,
    ifftshift,
)


def test_standard_fft():
    t = ones((10, 20, 30), chunk_size=(3, 20, 30))

    t1 = fft(t)
    assert t1.shape == (10, 20, 30)
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)

    t = ones((10, 20, 30), chunk_size=(3, 20, 30))

    t1 = ifft(t)
    assert t1.shape == (10, 20, 30)
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)

    t = ones((10, 20, 30), chunk_size=(3, 20, 30))

    t1 = fft2(t, s=(23, 21))
    assert t1.shape == (10, 23, 21)
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)

    t = ones((10, 20, 30), chunk_size=(3, 20, 30))

    t1 = ifft2(t, s=(11, 9), axes=(1, 2))
    assert t1.shape == (10, 11, 9)
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)

    t = ones((10, 20, 30), chunk_size=(3, 20, 30))

    t1 = fftn(t, s=(11, 9), axes=(1, 2))
    assert t1.shape == (10, 11, 9)
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)

    t = ones((10, 20, 30), chunk_size=(3, 20, 30))

    t1 = ifftn(t, s=(11, 9), axes=(1, 2))
    assert t1.shape == (10, 11, 9)
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)


def test_real_fft():
    t = ones((10, 20, 30), chunk_size=(3, 20, 30))

    t1 = rfft(t)
    assert t1.shape == np.fft.rfft(np.ones(t.shape)).shape
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)

    t = ones((10, 20, 30), chunk_size=(3, 20, 30))

    t1 = irfft(t)
    assert t1.shape == np.fft.irfft(np.ones(t.shape)).shape
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)

    t = ones((10, 20, 30), chunk_size=(3, 20, 30))

    t1 = rfft2(t, s=(23, 21))
    assert t1.shape == np.fft.rfft2(np.ones(t.shape), s=(23, 21)).shape
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)

    t = ones((10, 20, 30), chunk_size=(3, 20, 30))

    t1 = irfft2(t, s=(11, 9), axes=(1, 2))
    assert t1.shape == np.fft.irfft2(np.ones(t.shape), s=(11, 9), axes=(1, 2)).shape
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)

    t = ones((10, 20, 30), chunk_size=(3, 20, 30))

    t1 = rfftn(t, s=(11, 30), axes=(1, 2))
    assert t1.shape == np.fft.rfftn(np.ones(t.shape), s=(11, 30), axes=(1, 2)).shape
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)

    t = ones((10, 20, 30), chunk_size=(3, 20, 30))

    t1 = irfftn(t, s=(11, 9), axes=(1, 2))
    assert t1.shape == np.fft.irfftn(np.ones(t.shape), s=(11, 9), axes=(1, 2)).shape
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)


def test_hermitian_fft():
    t = ones((10, 20, 30), chunk_size=(3, 20, 30))

    t1 = hfft(t)
    assert t1.shape == np.fft.hfft(np.ones(t.shape)).shape
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)

    t = ones((10, 20, 30), chunk_size=(3, 20, 30))

    t1 = hfft(t, n=100)
    assert t1.shape == np.fft.hfft(np.ones(t.shape), n=100).shape
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)

    t = ones((10, 20, 30), chunk_size=(3, 20, 30))

    t1 = ihfft(t)
    assert t1.shape == np.fft.ihfft(np.ones(t.shape)).shape
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)

    t = ones((10, 20, 30), chunk_size=(3, 20, 30))

    t1 = ihfft(t, n=100)
    assert t1.shape == np.fft.ihfft(np.ones(t.shape), n=100).shape
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)

    t1 = ihfft(t, n=101)
    assert t1.shape == np.fft.ihfft(np.ones(t.shape), n=101).shape
    t1 = tile(t1)
    assert t1.shape == tuple(sum(ns) for ns in t1.nsplits)


def test_fft_shift():
    freqs = fftfreq(9, d=1.0 / 9).reshape(3, 3)
    t = ifftshift(fftshift(freqs))

    assert t.dtype is not None
    expect_dtype = np.fft.ifftshift(
        np.fft.fftshift(np.fft.fftfreq(9, d=1.0 / 9).reshape(3, 3))
    ).dtype
    assert t.dtype == expect_dtype


def test_fft_freq():
    t = fftfreq(10, 0.1, chunk_size=3)

    assert t.shape == np.fft.fftfreq(10, 0.1).shape
    t = tile(t)
    assert t.shape == tuple(sum(ns) for ns in t.nsplits)

    t = rfftfreq(10, 0.1, chunk_size=3)

    assert t.shape == np.fft.rfftfreq(10, 0.1).shape
    t = tile(t)
    assert t.shape == tuple(sum(ns) for ns in t.nsplits)
