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

from ....lib.version import parse as parse_version
import numpy as np
import scipy
import pytest
from scipy.special import (
    gammaln as scipy_gammaln,
    erf as scipy_erf,
    erfc as scipy_erfc,
    erfcx as scipy_erfcx,
    erfi as scipy_erfi,
    erfinv as scipy_erfinv,
    erfcinv as scipy_erfcinv,
    ellipk as scipy_ellipk,
    ellipkm1 as scipy_ellipkm1,
    ellipkinc as scipy_ellipkinc,
    ellipe as scipy_ellipe,
    ellipeinc as scipy_ellipeinc,
    elliprc as scipy_elliprc,
    elliprd as scipy_elliprd,
    elliprf as scipy_elliprf,
    elliprg as scipy_elliprg,
    elliprj as scipy_elliprj,
    betainc as scipy_betainc,
)

from ....core import tile
from ... import tensor
from ..err_fresnel import (
    erf,
    TensorErf,
    erfc,
    TensorErfc,
    erfcx,
    TensorErfcx,
    erfi,
    TensorErfi,
    erfinv,
    TensorErfinv,
    erfcinv,
    TensorErfcinv,
)
from ..gamma_funcs import (
    gammaln,
    TensorGammaln,
    betainc,
    TensorBetaInc,
)
from ..ellip_func_integrals import (
    ellipk,
    TensorEllipk,
    ellipkm1,
    TensorEllipkm1,
    ellipkinc,
    TensorEllipkinc,
    ellipe,
    TensorEllipe,
    ellipeinc,
    TensorEllipeinc,
    elliprc,
    TensorElliprc,
    elliprd,
    TensorElliprd,
    elliprf,
    TensorElliprf,
    elliprg,
    TensorElliprg,
    elliprj,
    TensorElliprj,
)


def test_gammaln():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = gammaln(t)
    expect = scipy_gammaln(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorGammaln)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_elf():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = erf(t)
    expect = scipy_erf(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorErf)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_erfc():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r_without_optional = erfc(t)
    expect = scipy_erfc(raw)

    assert r_without_optional.shape == raw.shape
    assert r_without_optional.dtype == expect.dtype

    t_without_optional, r_without_optional = tile(t, r_without_optional)

    assert r_without_optional.nsplits == t_without_optional.nsplits
    for c in r_without_optional.chunks:
        assert isinstance(c.op, TensorErfc)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape

    out = tensor(raw, chunk_size=3)
    r_with_optional = erfc(t, out)

    assert out.shape == raw.shape
    assert out.dtype == expect.dtype

    assert r_with_optional.shape == raw.shape
    assert r_with_optional.dtype == expect.dtype

    t_optional_out, out = tile(t, out)

    assert out.nsplits == t_optional_out.nsplits
    for c in out.chunks:
        assert isinstance(c.op, TensorErfc)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape

    t_optional_r, r_with_optional = tile(t, r_with_optional)

    assert r_with_optional.nsplits == t_optional_r.nsplits
    for c in r_with_optional.chunks:
        assert isinstance(c.op, TensorErfc)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_erfcx():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r_without_optional = erfcx(t)
    expect = scipy_erfcx(raw)

    assert r_without_optional.shape == raw.shape
    assert r_without_optional.dtype == expect.dtype

    t_without_optional, r_without_optional = tile(t, r_without_optional)

    assert r_without_optional.nsplits == t_without_optional.nsplits
    for c in r_without_optional.chunks:
        assert isinstance(c.op, TensorErfcx)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape

    out = tensor(raw, chunk_size=3)
    r_with_optional = erfcx(t, out)

    assert out.shape == raw.shape
    assert out.dtype == expect.dtype

    assert r_with_optional.shape == raw.shape
    assert r_with_optional.dtype == expect.dtype

    t_optional_out, out = tile(t, out)

    assert out.nsplits == t_optional_out.nsplits
    for c in out.chunks:
        assert isinstance(c.op, TensorErfcx)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape

    t_optional_r, r_with_optional = tile(t, r_with_optional)

    assert r_with_optional.nsplits == t_optional_r.nsplits
    for c in r_with_optional.chunks:
        assert isinstance(c.op, TensorErfcx)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_erfi():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r_without_optional = erfi(t)
    expect = scipy_erfi(raw)

    assert r_without_optional.shape == raw.shape
    assert r_without_optional.dtype == expect.dtype

    t_without_optional, r_without_optional = tile(t, r_without_optional)

    assert r_without_optional.nsplits == t_without_optional.nsplits
    for c in r_without_optional.chunks:
        assert isinstance(c.op, TensorErfi)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape

    out = tensor(raw, chunk_size=3)
    r_with_optional = erfi(t, out)

    assert out.shape == raw.shape
    assert out.dtype == expect.dtype

    assert r_with_optional.shape == raw.shape
    assert r_with_optional.dtype == expect.dtype

    t_optional_out, out = tile(t, out)

    assert out.nsplits == t_optional_out.nsplits
    for c in out.chunks:
        assert isinstance(c.op, TensorErfi)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape

    t_optional_r, r_with_optional = tile(t, r_with_optional)

    assert r_with_optional.nsplits == t_optional_r.nsplits
    for c in r_with_optional.chunks:
        assert isinstance(c.op, TensorErfi)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_erfinv():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = erfinv(t)
    expect = scipy_erfinv(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorErfinv)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_erfcinv():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = erfcinv(t)
    expect = scipy_erfcinv(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorErfcinv)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_beta_inc():
    raw1 = np.random.rand(4, 3, 2)
    raw2 = np.random.rand(4, 3, 2)
    raw3 = np.random.rand(4, 3, 2)
    a = tensor(raw1, chunk_size=3)
    b = tensor(raw2, chunk_size=3)
    c = tensor(raw3, chunk_size=3)

    r = betainc(a, b, c)
    expect = scipy_betainc(raw1, raw2, raw3)

    assert r.shape == raw1.shape
    assert r.dtype == expect.dtype

    tiled_a, r = tile(a, r)

    assert r.nsplits == tiled_a.nsplits
    for chunk in r.chunks:
        assert isinstance(chunk.op, TensorBetaInc)
        assert chunk.index == chunk.inputs[0].index
        assert chunk.shape == chunk.inputs[0].shape

    betainc(a, b, c, out=a)
    expect = scipy_betainc(raw1, raw2, raw3)

    assert a.shape == raw1.shape
    assert a.dtype == expect.dtype

    b, tiled_a = tile(b, a)

    assert tiled_a.nsplits == b.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorBetaInc)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_ellipk():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = ellipk(t)
    expect = scipy_ellipk(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorEllipk)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_ellipkm1():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = ellipkm1(t)
    expect = scipy_ellipkm1(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorEllipkm1)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_ellipkinc():
    raw1 = np.random.rand(4, 3, 2)
    raw2 = np.random.rand(4, 3, 2)
    a = tensor(raw1, chunk_size=3)
    b = tensor(raw2, chunk_size=3)

    r = ellipkinc(a, b)
    expect = scipy_ellipkinc(raw1, raw2)

    assert r.shape == raw1.shape
    assert r.dtype == expect.dtype

    tiled_a, r = tile(a, r)

    assert r.nsplits == tiled_a.nsplits
    for chunk in r.chunks:
        assert isinstance(chunk.op, TensorEllipkinc)
        assert chunk.index == chunk.inputs[0].index
        assert chunk.shape == chunk.inputs[0].shape


def test_ellipe():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = ellipe(t)
    expect = scipy_ellipe(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, TensorEllipe)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


def test_ellipeinc():
    raw1 = np.random.rand(4, 3, 2)
    raw2 = np.random.rand(4, 3, 2)
    a = tensor(raw1, chunk_size=3)
    b = tensor(raw2, chunk_size=3)

    r = ellipeinc(a, b)
    expect = scipy_ellipeinc(raw1, raw2)

    assert r.shape == raw1.shape
    assert r.dtype == expect.dtype

    tiled_a, r = tile(a, r)

    assert r.nsplits == tiled_a.nsplits
    for chunk in r.chunks:
        assert isinstance(chunk.op, TensorEllipeinc)
        assert chunk.index == chunk.inputs[0].index
        assert chunk.shape == chunk.inputs[0].shape


@pytest.mark.skipif(
    parse_version(scipy.__version__) < parse_version("1.8.0"), reason="function not implemented in scipy."
)
def test_elliprc():
    raw1 = np.random.rand(4, 3, 2)
    raw2 = np.random.rand(4, 3, 2)
    a = tensor(raw1, chunk_size=3)
    b = tensor(raw2, chunk_size=3)

    r = elliprc(a, b)
    expect = scipy_elliprc(raw1, raw2)

    assert r.shape == raw1.shape
    assert r.dtype == expect.dtype

    tiled_a, r = tile(a, r)

    assert r.nsplits == tiled_a.nsplits
    for chunk in r.chunks:
        assert isinstance(chunk.op, TensorElliprc)
        assert chunk.index == chunk.inputs[0].index
        assert chunk.shape == chunk.inputs[0].shape


@pytest.mark.skipif(
    parse_version(scipy.__version__) < parse_version("1.8.0"), reason="function not implemented in scipy."
)
def test_elliprd():
    raw1 = np.random.rand(4, 3, 2)
    raw2 = np.random.rand(4, 3, 2)
    raw3 = np.random.rand(4, 3, 2)
    a = tensor(raw1, chunk_size=3)
    b = tensor(raw2, chunk_size=3)
    c = tensor(raw3, chunk_size=3)

    r = elliprd(a, b, c)
    expect = scipy_elliprd(raw1, raw2, raw3)

    assert r.shape == raw1.shape
    assert r.dtype == expect.dtype

    tiled_a, r = tile(a, r)

    assert r.nsplits == tiled_a.nsplits
    for chunk in r.chunks:
        assert isinstance(chunk.op, TensorElliprd)
        assert chunk.index == chunk.inputs[0].index
        assert chunk.shape == chunk.inputs[0].shape


@pytest.mark.skipif(
    parse_version(scipy.__version__) < parse_version("1.8.0"), reason="function not implemented in scipy."
)
def test_elliprf():
    raw1 = np.random.rand(4, 3, 2)
    raw2 = np.random.rand(4, 3, 2)
    raw3 = np.random.rand(4, 3, 2)
    a = tensor(raw1, chunk_size=3)
    b = tensor(raw2, chunk_size=3)
    c = tensor(raw3, chunk_size=3)

    r = elliprf(a, b, c)
    expect = scipy_elliprf(raw1, raw2, raw3)

    assert r.shape == raw1.shape
    assert r.dtype == expect.dtype

    tiled_a, r = tile(a, r)

    assert r.nsplits == tiled_a.nsplits
    for chunk in r.chunks:
        assert isinstance(chunk.op, TensorElliprf)
        assert chunk.index == chunk.inputs[0].index
        assert chunk.shape == chunk.inputs[0].shape


@pytest.mark.skipif(
    parse_version(scipy.__version__) < parse_version("1.8.0"), reason="function not implemented in scipy."
)
def test_elliprg():
    raw1 = np.random.rand(4, 3, 2)
    raw2 = np.random.rand(4, 3, 2)
    raw3 = np.random.rand(4, 3, 2)
    a = tensor(raw1, chunk_size=3)
    b = tensor(raw2, chunk_size=3)
    c = tensor(raw3, chunk_size=3)

    r = elliprg(a, b, c)
    expect = scipy_elliprg(raw1, raw2, raw3)

    assert r.shape == raw1.shape
    assert r.dtype == expect.dtype

    tiled_a, r = tile(a, r)

    assert r.nsplits == tiled_a.nsplits
    for chunk in r.chunks:
        assert isinstance(chunk.op, TensorElliprg)
        assert chunk.index == chunk.inputs[0].index
        assert chunk.shape == chunk.inputs[0].shape


@pytest.mark.skipif(
    parse_version(scipy.__version__) < parse_version("1.8.0"), reason="function not implemented in scipy."
)
def test_elliprj():
    raw1 = np.random.rand(4, 3, 2)
    raw2 = np.random.rand(4, 3, 2)
    raw3 = np.random.rand(4, 3, 2)
    raw4 = np.random.rand(4, 3, 2)
    a = tensor(raw1, chunk_size=3)
    b = tensor(raw2, chunk_size=3)
    c = tensor(raw3, chunk_size=3)
    d = tensor(raw4, chunk_size=3)

    r = elliprj(a, b, c, d)
    expect = scipy_elliprj(raw1, raw2, raw3, raw4)

    assert r.shape == raw1.shape
    assert r.dtype == expect.dtype

    tiled_a, r = tile(a, r)

    assert r.nsplits == tiled_a.nsplits
    for chunk in r.chunks:
        assert isinstance(chunk.op, TensorElliprj)
        assert chunk.index == chunk.inputs[0].index
        assert chunk.shape == chunk.inputs[0].shape
