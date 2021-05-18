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

import numpy as np
import pytest
try:
    import scipy
    from scipy.special import (
        gammaln as scipy_gammaln,
        erf as scipy_erf,
        betainc as scipy_betainc,
    )
    from mars.tensor.special.gamma_funcs import (
        gammaln, TensorGammaln,
        betainc, TensorBetaInc,
    )
    from mars.tensor.special.err_fresnel import erf, TensorErf
except ImportError:
    scipy = None

from mars.core import tile
from mars.tensor import tensor


@pytest.mark.skipif(scipy is None, reason='scipy not installed')
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


@pytest.mark.skipif(scipy is None, reason='scipy not installed')
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


@pytest.mark.skipif(scipy is None, reason='scipy not installed')
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
