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
import pytest
import scipy.sparse as sps
import scipy.special as spspecial

from ... import tensor
from ... import special as mt_special


@pytest.mark.parametrize('func', [
    'gamma',
    'gammaln',
    'loggamma',
    'gammasgn',
    'psi',
    'rgamma',
    'digamma',
    'erf',
    'entr',
])
def test_unary_execution(setup, func):
    sp_func = getattr(spspecial, func)
    mt_func = getattr(mt_special, func)

    raw = np.random.rand(10, 8, 6)
    a = tensor(raw, chunk_size=3)

    r = mt_func(a)

    result = r.execute().fetch()
    expected = sp_func(raw)

    np.testing.assert_array_equal(result, expected)

    # test sparse
    raw = sps.csr_matrix(np.array([0, 1.0, 1.01, np.nan]))
    a = tensor(raw, chunk_size=3)

    r = mt_func(a)

    result = r.execute().fetch()

    data = sp_func(raw.data)
    expected = sps.csr_matrix((data, raw.indices, raw.indptr), raw.shape)

    np.testing.assert_array_equal(result.toarray(), expected.toarray())


@pytest.mark.parametrize('func', [
    'gammainc',
    'gammaincinv',
    'gammaincc',
    'gammainccinv',
    'beta',
    'betaln',
    'polygamma',
    'poch',
    'rel_entr',
    'kl_div',
    'xlogy',
    'jv',
    'jve',
    'yn',
    'yv',
    'yve',
    'kn',
    'kv',
    'kve',
    'iv',
    'ive',
    'hankel1',
    'hankel1e',
    'hankel2',
    'hankel2e',
    'hyp0f1'
])
def test_binary_execution(setup, func):
    sp_func = getattr(spspecial, func)
    mt_func = getattr(mt_special, func)

    raw1 = np.random.rand(4, 3, 2)
    raw2 = np.random.rand(4, 3, 2)
    a = tensor(raw1, chunk_size=3)
    b = tensor(raw2, chunk_size=3)

    r = mt_func(a, b)

    result = r.execute().fetch()
    expected = sp_func(raw1, raw2)

    np.testing.assert_array_equal(result, expected)

    # test sparse
    raw1 = sps.csr_matrix(np.array([0, 1.0, 1.01, np.nan] * 3).reshape(4, 3))
    a = tensor(raw1, chunk_size=3)
    raw2 = np.random.rand(4, 3)
    b = tensor(raw2, chunk_size=3)

    r = mt_func(a, b)

    result = r.execute().fetch()

    expected = sp_func(raw1.toarray(), raw2)
    np.testing.assert_array_equal(result.toarray(), expected)


@pytest.mark.parametrize('func', [
    'betainc',
    'betaincinv',
    'hyp1f1',
    'hyperu',
])
def test_triple_execution(setup, func):
    sp_func = getattr(spspecial, func)
    mt_func = getattr(mt_special, func)

    raw1 = np.random.rand(4, 3, 2)
    raw2 = np.random.rand(4, 3, 2)
    raw3 = np.random.rand(4, 3, 2)
    a = tensor(raw1, chunk_size=3)
    b = tensor(raw2, chunk_size=3)
    c = tensor(raw3, chunk_size=3)

    r = mt_func(a, b, c)

    result = r.execute().fetch()
    expected = sp_func(raw1, raw2, raw3)

    np.testing.assert_array_equal(result, expected)

    # test sparse
    raw1 = sps.csr_matrix(np.array([0, 1.0, 1.01, np.nan] * 3).reshape(4, 3))
    a = tensor(raw1, chunk_size=3)
    raw2 = np.random.rand(4, 3)
    b = tensor(raw2, chunk_size=3)
    raw3 = np.random.rand(4, 3)
    c = tensor(raw3, chunk_size=3)

    r = mt_func(a, b, c)

    result = r.execute().fetch()

    expected = sp_func(raw1.toarray(), raw2, raw3)
    np.testing.assert_array_equal(result.toarray(), expected)


@pytest.mark.parametrize('func', [
    'hyp2f1',
])
def test_quadruple_execution(setup, func):
    sp_func = getattr(spspecial, func)
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

    result = r.execute().fetch()
    expected = sp_func(raw1, raw2, raw3, raw4)

    np.testing.assert_array_equal(result, expected)

    # test sparse
    raw1 = sps.csr_matrix(np.array([0, 1.0, 1.01, np.nan] * 3).reshape(4, 3))
    a = tensor(raw1, chunk_size=3)
    raw2 = np.random.rand(4, 3)
    b = tensor(raw2, chunk_size=3)
    raw3 = np.random.rand(4, 3)
    c = tensor(raw3, chunk_size=3)
    raw4 = np.random.rand(4, 3)
    d = tensor(raw4, chunk_size=3)

    r = mt_func(a, b, c, d)

    result = r.execute().fetch()

    expected = sp_func(raw1.toarray(), raw2, raw3, raw4)
    np.testing.assert_array_equal(result.toarray(), expected)
