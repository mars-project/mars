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
    fresnel as scipy_fresnel,
    betainc as scipy_betainc,
)

from ....lib.version import parse as parse_version
from ....core import tile, ExecutableTuple
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
    fresnel,
    TensorFresnel,
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


def test_fresnel():
    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = fresnel(t)
    expect = scipy_fresnel(raw)

    assert isinstance(r, ExecutableTuple)
    assert len(r) == 2

    for i in range(len(r)):
        assert r[i].shape == expect[i].shape
        assert r[i].dtype == expect[i].dtype
        assert isinstance(r[i], TensorFresnel)

    non_tuple_out = tensor(raw, chunk_size=3)
    with pytest.raises(TypeError):
        r = fresnel(t, non_tuple_out)

    out = ExecutableTuple([t, t])
    r_out = fresnel(t, out=out)

    assert isinstance(out, ExecutableTuple)
    assert isinstance(r_out, ExecutableTuple)

    assert len(out) == 2
    assert len(r_out) == 2

    for i in range(len(r_out)):
        assert r_out[i].shape == expect[i].shape
        assert r_out[i].dtype == expect[i].dtype
        assert isinstance(r_out[i], TensorFresnel)

        assert out[i].shape == expect[i].shape
        assert out[i].dtype == expect[i].dtype
        assert isinstance(out[i], TensorFresnel)


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
    parse_version(scipy.__version__) < parse_version("1.8.0"),
    reason="function not implemented in scipy.",
)
def test_elliprc():
    from scipy.special import elliprc as scipy_elliprc
    from ..ellip_func_integrals import elliprc, TensorElliprc

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
    parse_version(scipy.__version__) < parse_version("1.8.0"),
    reason="function not implemented in scipy.",
)
def test_elliprd():
    from scipy.special import elliprd as scipy_elliprd
    from ..ellip_func_integrals import elliprd, TensorElliprd

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
    parse_version(scipy.__version__) < parse_version("1.8.0"),
    reason="function not implemented in scipy.",
)
def test_elliprf():
    from scipy.special import elliprf as scipy_elliprf
    from ..ellip_func_integrals import elliprf, TensorElliprf

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
    parse_version(scipy.__version__) < parse_version("1.8.0"),
    reason="function not implemented in scipy.",
)
def test_elliprg():
    from scipy.special import elliprg as scipy_elliprg
    from ..ellip_func_integrals import elliprg, TensorElliprg

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
    parse_version(scipy.__version__) < parse_version("1.8.0"),
    reason="function not implemented in scipy.",
)
def test_elliprj():
    from scipy.special import elliprj as scipy_elliprj
    from ..ellip_func_integrals import elliprj, TensorElliprj

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
