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
import scipy.special as spsecial

from ....lib.version import parse as parse_version
from ....core import tile, ExecutableTuple
from ... import tensor
from ... import special as mt_special
from ..err_fresnel import (
    TensorErf,
    TensorErfc,
    TensorErfcx,
    TensorErfi,
    TensorErfinv,
    TensorErfcinv,
    TensorWofz,
    TensorDawsn,
    TensorFresnel,
    TensorModFresnelP,
    TensorModFresnelM,
    TensorVoigtProfile,
)
from ..gamma_funcs import (
    TensorGammaln,
    TensorBetaInc,
)
from ..ellip_func_integrals import (
    TensorElliprc,
    TensorElliprd,
    TensorElliprf,
    TensorElliprg,
    TensorElliprj,
    TensorEllipk,
    TensorEllipkm1,
    TensorEllipkinc,
    TensorEllipe,
    TensorEllipeinc,
)
from ..airy import (
    TensorAiry,
    TensorAirye,
    TensorItairy,
)


@pytest.mark.parametrize(
    "func,tensor_cls",
    [
        ("gammaln", TensorGammaln),
        ("erf", TensorErf),
        ("erfinv", TensorErfinv),
        ("erfcinv", TensorErfcinv),
        ("wofz", TensorWofz),
        ("dawsn", TensorDawsn),
        ("ellipk", TensorEllipk),
        ("ellipkm1", TensorEllipkm1),
        ("ellipe", TensorEllipe),
        ("erfc", TensorErfc),
        ("erfcx", TensorErfcx),
        ("erfi", TensorErfi),
    ],
)
def test_unary_operand_no_out(func, tensor_cls):
    sp_func = getattr(spsecial, func)
    mt_func = getattr(mt_special, func)

    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = mt_func(t)
    expect = sp_func(raw)

    assert r.shape == raw.shape
    assert r.dtype == expect.dtype

    t, r = tile(t, r)

    assert r.nsplits == t.nsplits
    for c in r.chunks:
        assert isinstance(c.op, tensor_cls)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


@pytest.mark.parametrize(
    "func,tensor_cls",
    [
        ("erfc", TensorErfc),
        ("erfcx", TensorErfcx),
        ("erfi", TensorErfi),
    ],
)
def test_unary_operand_out(func, tensor_cls):
    sp_func = getattr(spsecial, func)
    mt_func = getattr(mt_special, func)

    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    out = tensor(raw, chunk_size=3)
    r_with_optional = mt_func(t, out)
    expect = sp_func(raw)

    assert out.shape == raw.shape
    assert out.dtype == expect.dtype

    assert r_with_optional.shape == raw.shape
    assert r_with_optional.dtype == expect.dtype

    t_optional_out, out = tile(t, out)

    assert out.nsplits == t_optional_out.nsplits
    for c in out.chunks:
        assert isinstance(c.op, tensor_cls)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape

    t_optional_r, r_with_optional = tile(t, r_with_optional)

    assert r_with_optional.nsplits == t_optional_r.nsplits
    for c in r_with_optional.chunks:
        assert isinstance(c.op, tensor_cls)
        assert c.index == c.inputs[0].index
        assert c.shape == c.inputs[0].shape


@pytest.mark.parametrize(
    "func,tensor_cls,n_outputs",
    [
        ("fresnel", TensorFresnel, 2),
        ("modfresnelp", TensorModFresnelP, 2),
        ("modfresnelm", TensorModFresnelM, 2),
        ("airy", TensorAiry, 4),
        ("airye", TensorAirye, 4),
        ("itairy", TensorItairy, 4),
    ],
)
def test_unary_tuple_operand(func, tensor_cls, n_outputs):
    sp_func = getattr(spsecial, func)
    mt_func = getattr(mt_special, func)

    raw = np.random.rand(10, 8, 5)
    t = tensor(raw, chunk_size=3)

    r = mt_func(t)
    expect = sp_func(raw)

    assert isinstance(r, ExecutableTuple)

    for r_i, expect_i in zip(r, expect):
        assert r_i.shape == expect_i.shape
        assert r_i.dtype == expect_i.dtype
        assert isinstance(r_i.op, tensor_cls)

    non_tuple_out = tensor(raw, chunk_size=3)
    with pytest.raises(TypeError):
        r = mt_func(t, non_tuple_out)

    mismatch_size_tuple = ExecutableTuple([t])
    with pytest.raises(TypeError):
        r = mt_func(t, mismatch_size_tuple)

    out = ExecutableTuple([t] * n_outputs)
    r_out = mt_func(t, out=out)

    assert isinstance(out, ExecutableTuple)
    assert isinstance(r_out, ExecutableTuple)

    for r_output, expected_output, out_output in zip(r, expect, out):
        assert r_output.shape == expected_output.shape
        assert r_output.dtype == expected_output.dtype
        assert isinstance(r_output.op, tensor_cls)

        assert out_output.shape == expected_output.shape
        assert out_output.dtype == expected_output.dtype
        assert isinstance(out_output.op, tensor_cls)


@pytest.mark.parametrize(
    "func,tensor_cls",
    [
        ("betainc", TensorBetaInc),
        ("voigt_profile", TensorVoigtProfile),
        pytest.param(
            "elliprd",
            TensorElliprd,
            marks=pytest.mark.skipif(
                parse_version(scipy.__version__) < parse_version("1.8.0"),
                reason="function not implemented in scipy.",
            ),
        ),
        pytest.param(
            "elliprf",
            TensorElliprf,
            marks=pytest.mark.skipif(
                parse_version(scipy.__version__) < parse_version("1.8.0"),
                reason="function not implemented in scipy.",
            ),
        ),
        pytest.param(
            "elliprg",
            TensorElliprg,
            marks=pytest.mark.skipif(
                parse_version(scipy.__version__) < parse_version("1.8.0"),
                reason="function not implemented in scipy.",
            ),
        ),
    ],
)
def test_triple_operand(func, tensor_cls):
    sp_func = getattr(spsecial, func)
    mt_func = getattr(mt_special, func)

    raw1 = np.random.rand(4, 3, 2)
    raw2 = np.random.rand(4, 3, 2)
    raw3 = np.random.rand(4, 3, 2)
    a = tensor(raw1, chunk_size=3)
    b = tensor(raw2, chunk_size=3)
    c = tensor(raw3, chunk_size=3)

    r = mt_func(a, b, c)
    expect = sp_func(raw1, raw2, raw3)

    assert r.shape == raw1.shape
    assert r.dtype == expect.dtype

    tiled_a, r = tile(a, r)

    assert r.nsplits == tiled_a.nsplits
    for chunk in r.chunks:
        assert isinstance(chunk.op, tensor_cls)
        assert chunk.index == chunk.inputs[0].index
        assert chunk.shape == chunk.inputs[0].shape


@pytest.mark.parametrize(
    "func,tensor_cls",
    [
        ("ellipkinc", TensorEllipkinc),
        ("ellipeinc", TensorEllipeinc),
        pytest.param(
            "elliprc",
            TensorElliprc,
            marks=pytest.mark.skipif(
                parse_version(scipy.__version__) < parse_version("1.8.0"),
                reason="function not implemented in scipy.",
            ),
        ),
    ],
)
def test_binary_operand(func, tensor_cls):
    sp_func = getattr(spsecial, func)
    mt_func = getattr(mt_special, func)

    raw1 = np.random.rand(4, 3, 2)
    raw2 = np.random.rand(4, 3, 2)
    a = tensor(raw1, chunk_size=3)
    b = tensor(raw2, chunk_size=3)

    r = mt_func(a, b)
    expect = sp_func(raw1, raw2)

    assert r.shape == raw1.shape
    assert r.dtype == expect.dtype

    tiled_a, r = tile(a, r)

    assert r.nsplits == tiled_a.nsplits
    for chunk in r.chunks:
        assert isinstance(chunk.op, tensor_cls)
        assert chunk.index == chunk.inputs[0].index
        assert chunk.shape == chunk.inputs[0].shape


@pytest.mark.parametrize(
    "func,tensor_cls",
    [
        pytest.param(
            "elliprj",
            TensorElliprj,
            marks=pytest.mark.skipif(
                parse_version(scipy.__version__) < parse_version("1.8.0"),
                reason="function not implemented in scipy.",
            ),
        ),
    ],
)
def test_quadruple_operand(func, tensor_cls):
    sp_func = getattr(spsecial, func)
    mt_func = getattr(mt_special, func)

    raw1 = np.random.rand(4, 3, 2)
    raw2 = np.random.rand(4, 3, 2)
    raw3 = np.random.rand(4, 3, 2)
    raw4 = np.random.rand(4, 3, 2)
    a = tensor(raw1, chunk_size=3)
    b = tensor(raw2, chunk_size=3)
    c = tensor(raw3, chunk_size=3)
    d = tensor(raw4, chunk_size=3)

    r = mt_func(a, b, c, d)
    expect = sp_func(raw1, raw2, raw3, raw4)

    assert r.shape == raw1.shape
    assert r.dtype == expect.dtype

    tiled_a, r = tile(a, r)

    assert r.nsplits == tiled_a.nsplits
    for chunk in r.chunks:
        assert isinstance(chunk.op, tensor_cls)
        assert chunk.index == chunk.inputs[0].index
        assert chunk.shape == chunk.inputs[0].shape
