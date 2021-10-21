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

from ....lib.mkl_interface import mkl_free_buffers
from ...datasource import tensor
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
    hfft,
    ihfft,
    fftfreq,
    rfftfreq,
    fftshift,
    ifftshift,
    irfftn,
)


def test_fft_execution(setup):
    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(8, 8, 30))

    r = fft(t)
    res = r.execute().fetch()
    expected = np.fft.fft(raw)
    np.testing.assert_allclose(res, expected)

    r = fft(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.fft(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = fft(t, n=11)
    res = r.execute().fetch()
    expected = np.fft.fft(raw, n=11)
    np.testing.assert_allclose(res, expected)

    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(8, 8, 8))

    r = fft(t)
    res = r.execute().fetch()
    expected = np.fft.fft(raw)
    np.testing.assert_allclose(res, expected)

    r = fft(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.fft(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = fft(t, n=11)
    res = r.execute().fetch()
    expected = np.fft.fft(raw, n=11)
    np.testing.assert_allclose(res, expected)


def test_ifft_execution(setup):
    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(8, 8, 30))

    r = ifft(t)
    res = r.execute().fetch()
    expected = np.fft.ifft(raw)
    np.testing.assert_allclose(res, expected)

    r = ifft(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.ifft(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = ifft(t, n=11)
    res = r.execute().fetch()
    expected = np.fft.ifft(raw, n=11)
    np.testing.assert_allclose(res, expected)

    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(8, 8, 10))

    r = ifft(t)
    res = r.execute().fetch()
    expected = np.fft.ifft(raw)
    np.testing.assert_allclose(res, expected)

    r = ifft(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.ifft(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = ifft(t, n=11)
    res = r.execute().fetch()
    expected = np.fft.ifft(raw, n=11)
    np.testing.assert_allclose(res, expected)


def test_fft2_execution(setup):
    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(8, 20, 30))

    r = fft2(t)
    res = r.execute().fetch()
    expected = np.fft.fft2(raw)
    np.testing.assert_allclose(res, expected)

    r = fft2(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.fft2(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = fft2(t, s=(11, 12))
    res = r.execute().fetch()
    expected = np.fft.fft2(raw, s=(11, 12))
    np.testing.assert_allclose(res, expected)

    r = fft2(t, s=(11, 12), axes=(-1, -2))
    res = r.execute().fetch()
    expected = np.fft.fft2(raw, s=(11, 12), axes=(-1, -2))
    np.testing.assert_allclose(res, expected)

    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(8, 10, 12))

    r = fft2(t)
    res = r.execute().fetch()
    expected = np.fft.fft2(raw)
    np.testing.assert_allclose(res, expected)

    r = fft2(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.fft2(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = fft2(t, s=(11, 12))
    res = r.execute().fetch()
    expected = np.fft.fft2(raw, s=(11, 12))
    np.testing.assert_allclose(res, expected)

    r = fft2(t, s=(11, 12), axes=(-1, -2))
    res = r.execute().fetch()
    expected = np.fft.fft2(raw, s=(11, 12), axes=(-1, -2))
    np.testing.assert_allclose(res, expected)


def test_ifft2_execution(setup):
    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(8, 20, 30))

    r = ifft2(t)
    res = r.execute().fetch()
    expected = np.fft.ifft2(raw)
    np.testing.assert_allclose(res, expected)

    r = ifft2(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.ifft2(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = ifft2(t, s=(11, 12))
    res = r.execute().fetch()
    expected = np.fft.ifft2(raw, s=(11, 12))
    np.testing.assert_allclose(res, expected)

    r = ifft2(t, s=(11, 12), axes=(-1, -2))
    res = r.execute().fetch()
    expected = np.fft.ifft2(raw, s=(11, 12), axes=(-1, -2))
    np.testing.assert_allclose(res, expected)

    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(8, 6, 10))

    r = ifft2(t)
    res = r.execute().fetch()
    expected = np.fft.ifft2(raw)
    np.testing.assert_allclose(res, expected)

    r = ifft2(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.ifft2(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = ifft2(t, s=(11, 12))
    res = r.execute().fetch()
    expected = np.fft.ifft2(raw, s=(11, 12))
    np.testing.assert_allclose(res, expected)

    r = ifft2(t, s=(11, 12), axes=(-1, -2))
    res = r.execute().fetch()
    expected = np.fft.ifft2(raw, s=(11, 12), axes=(-1, -2))
    np.testing.assert_allclose(res, expected)


def test_fftn_execution(setup):
    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(10, 20, 30))

    r = fftn(t)
    res = r.execute().fetch()
    expected = np.fft.fftn(raw)
    np.testing.assert_allclose(res, expected)

    r = fftn(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.fftn(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = fftn(t, s=(11, 12, 5))
    res = r.execute().fetch()
    expected = np.fft.fftn(raw, s=(11, 12, 5))
    np.testing.assert_allclose(res, expected)

    r = fftn(t, s=(11, 12, 5), axes=(-1, -2, -3))
    res = r.execute().fetch()
    expected = np.fft.fftn(raw, s=(11, 12, 5), axes=(-1, -2, -3))
    np.testing.assert_allclose(res, expected)

    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(6, 6, 8))

    r = fftn(t)
    res = r.execute().fetch()
    expected = np.fft.fftn(raw)
    np.testing.assert_allclose(res, expected)

    r = fftn(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.fftn(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = fftn(t, s=(11, 12, 5))
    res = r.execute().fetch()
    expected = np.fft.fftn(raw, s=(11, 12, 5))
    np.testing.assert_allclose(res, expected)

    r = fftn(t, s=(11, 12, 5), axes=(-1, -2, -3))
    res = r.execute().fetch()
    expected = np.fft.fftn(raw, s=(11, 12, 5), axes=(-1, -2, -3))
    np.testing.assert_allclose(res, expected)


def test_ifftn_execution(setup):
    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(10, 20, 30))

    r = ifftn(t)
    res = r.execute().fetch()
    expected = np.fft.ifftn(raw)
    np.testing.assert_allclose(res, expected)

    r = ifftn(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.ifftn(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = ifftn(t, s=(11, 12, 5))
    res = r.execute().fetch()
    expected = np.fft.ifftn(raw, s=(11, 12, 5))
    np.testing.assert_allclose(res, expected)

    r = ifftn(t, s=(11, 12, 5), axes=(-1, -2, -3))
    res = r.execute().fetch()
    expected = np.fft.ifftn(raw, s=(11, 12, 5), axes=(-1, -2, -3))
    np.testing.assert_allclose(res, expected)

    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(6, 8, 14))

    r = ifftn(t)
    res = r.execute().fetch()
    expected = np.fft.ifftn(raw)
    np.testing.assert_allclose(res, expected)

    r = ifftn(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.ifftn(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = ifftn(t, s=(11, 12, 5))
    res = r.execute().fetch()
    expected = np.fft.ifftn(raw, s=(11, 12, 5))
    np.testing.assert_allclose(res, expected)

    r = ifftn(t, s=(11, 12, 5), axes=(-1, -2, -3))
    res = r.execute().fetch()
    expected = np.fft.ifftn(raw, s=(11, 12, 5), axes=(-1, -2, -3))
    np.testing.assert_allclose(res, expected)


def test_rfft_execution(setup):
    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(8, 8, 30))

    r = rfft(t)
    res = r.execute().fetch()
    expected = np.fft.rfft(raw)
    np.testing.assert_allclose(res, expected)

    r = rfft(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.rfft(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = rfft(t, n=11)
    res = r.execute().fetch()
    expected = np.fft.rfft(raw, n=11)
    np.testing.assert_allclose(res, expected)


def test_irfft_execution(setup):
    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(8, 8, 30))

    r = irfft(t)
    res = r.execute().fetch()
    expected = np.fft.irfft(raw)
    np.testing.assert_allclose(res, expected)

    r = irfft(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.irfft(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = irfft(t, n=11)
    res = r.execute().fetch()
    expected = np.fft.irfft(raw, n=11)
    np.testing.assert_allclose(res, expected)


def test_rfft2_execution(setup):
    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(8, 20, 30))

    r = rfft2(t)
    res = r.execute().fetch()
    expected = np.fft.rfft2(raw)
    np.testing.assert_allclose(res, expected)

    r = rfft2(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.rfft2(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = rfft2(t, s=(11, 12))
    res = r.execute().fetch()
    expected = np.fft.rfft2(raw, s=(11, 12))
    np.testing.assert_allclose(res, expected)

    r = rfft2(t, s=(11, 12), axes=(-1, -2))
    res = r.execute().fetch()
    expected = np.fft.rfft2(raw, s=(11, 12), axes=(-1, -2))
    np.testing.assert_allclose(res, expected)


def test_irfft2_execution(setup):
    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(8, 20, 30))

    r = irfft2(t)
    res = r.execute().fetch()
    expected = np.fft.irfft2(raw)
    np.testing.assert_allclose(res, expected)

    r = irfft2(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.irfft2(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = irfft2(t, s=(11, 12))
    res = r.execute().fetch()
    expected = np.fft.irfft2(raw, s=(11, 12))
    np.testing.assert_allclose(res, expected)

    r = irfft2(t, s=(11, 12), axes=(-1, -2))
    res = r.execute().fetch()
    expected = np.fft.irfft2(raw, s=(11, 12), axes=(-1, -2))
    np.testing.assert_allclose(res, expected)


def test_rfftn_execution(setup):
    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(10, 20, 30))

    r = rfftn(t)
    res = r.execute().fetch()
    expected = np.fft.rfftn(raw)
    np.testing.assert_allclose(res, expected)

    r = rfftn(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.rfftn(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = rfftn(t, s=(11, 12, 5))
    res = r.execute().fetch()
    expected = np.fft.rfftn(raw, s=(11, 12, 5))
    np.testing.assert_allclose(res, expected)

    r = rfftn(t, s=(11, 12, 11), axes=(-1, -2, -3))
    res = r.execute().fetch()
    expected = np.fft.rfftn(raw, s=(11, 12, 11), axes=(-1, -2, -3))
    np.testing.assert_allclose(res, expected)


def test_irfftn_execution(setup):
    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(10, 20, 30))

    r = irfftn(t)
    res = r.execute().fetch()
    expected = np.fft.irfftn(raw)
    np.testing.assert_allclose(res, expected)

    r = irfftn(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.irfftn(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = irfftn(t, s=(11, 21, 5))
    res = r.execute().fetch()
    expected = np.fft.irfftn(raw, s=(11, 21, 5))
    np.testing.assert_allclose(res, expected)

    # a bug in mkl version will cause the section below to fail
    if mkl_free_buffers is None:
        r = irfftn(t, s=(11, 21, 30), axes=(-1, -2, -3))
        res = r.execute().fetch()
        expected = np.fft.irfftn(raw, s=(11, 21, 30), axes=(-1, -2, -3))
        np.testing.assert_allclose(res, expected)


def test_hfft_execution(setup):
    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(8, 8, 30))

    r = hfft(t)
    res = r.execute().fetch()
    expected = np.fft.hfft(raw)
    np.testing.assert_allclose(res, expected)

    r = hfft(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.hfft(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = hfft(t, n=11)
    res = r.execute().fetch()
    expected = np.fft.hfft(raw, n=11)
    np.testing.assert_allclose(res, expected)


def test_ihfft_execution(setup):
    raw = np.random.rand(10, 20, 30)
    t = tensor(raw, chunk_size=(8, 8, 30))

    r = ihfft(t)
    res = r.execute().fetch()
    expected = np.fft.ihfft(raw)
    np.testing.assert_allclose(res, expected)

    r = ihfft(t, norm="ortho")
    res = r.execute().fetch()
    expected = np.fft.ihfft(raw, norm="ortho")
    np.testing.assert_allclose(res, expected)

    r = ihfft(t, n=11)
    res = r.execute().fetch()
    expected = np.fft.ihfft(raw, n=11)
    np.testing.assert_allclose(res, expected)

    r = ihfft(t, n=12)
    res = r.execute().fetch()
    expected = np.fft.ihfft(raw, n=12)
    np.testing.assert_allclose(res, expected)


def test_fft_freq_execution(setup):
    t = fftfreq(10, 0.1, chunk_size=6)

    res = t.execute().fetch()
    np.testing.assert_allclose(res, np.fft.fftfreq(10, 0.1))

    t = fftfreq(11, 0.01, chunk_size=6)

    res = t.execute().fetch()
    np.testing.assert_allclose(res, np.fft.fftfreq(11, 0.01))


def test_rfft_freq_execution(setup):
    t = rfftfreq(20, 0.1, chunk_size=6)

    res = t.execute().fetch()
    np.testing.assert_allclose(res, np.fft.rfftfreq(20, 0.1))

    t = rfftfreq(21, 0.01, chunk_size=6)

    res = t.execute().fetch()
    np.testing.assert_allclose(res, np.fft.rfftfreq(21, 0.01))


def test_fft_shift_execution(setup):
    t = fftfreq(10, 0.1, chunk_size=6)
    r = fftshift(t)

    res = r.execute().fetch()
    np.testing.assert_allclose(res, np.fft.fftshift(np.fft.fftfreq(10, 0.1)))

    freqs = fftfreq(9, d=1.0 / 9, chunk_size=4).reshape(3, 3)
    r = fftshift(freqs, axes=(1,))

    res = r.execute().fetch()
    expected = np.fft.fftshift(np.fft.fftfreq(9, d=1.0 / 9).reshape(3, 3), axes=(1,))
    np.testing.assert_allclose(res, expected)


def test_ifft_shift_execution(setup):
    t = fftfreq(9, d=1.0 / 9, chunk_size=4).reshape(3, 3)
    r = ifftshift(t)

    res = r.execute().fetch()
    expected = np.fft.ifftshift(np.fft.fftfreq(9, d=1.0 / 9).reshape(3, 3))
    np.testing.assert_allclose(res, expected)
