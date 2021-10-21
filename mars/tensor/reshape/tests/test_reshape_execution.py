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

from ...datasource import ones, tensor


def test_reshape_execution(setup):
    x = ones((1, 2, 3), chunk_size=[4, 3, 5])
    y = x.reshape(3, 2)
    res = y.execute().fetch()
    assert y.shape == (3, 2)
    np.testing.assert_equal(res, np.ones((3, 2)))

    data = np.random.rand(6, 4)
    x2 = tensor(data, chunk_size=2)
    y2 = x2.reshape(3, 8, order="F")
    res = y2.execute().fetch()
    expected = data.reshape((3, 8), order="F")
    np.testing.assert_array_equal(res, expected)
    assert res.flags["F_CONTIGUOUS"] is True
    assert res.flags["C_CONTIGUOUS"] is False

    data2 = np.asfortranarray(np.random.rand(6, 4))
    x3 = tensor(data2, chunk_size=2)
    y3 = x3.reshape(3, 8)
    res = y3.execute().fetch()
    expected = data2.reshape((3, 8))
    np.testing.assert_array_equal(res, expected)
    assert res.flags["C_CONTIGUOUS"] is True
    assert res.flags["F_CONTIGUOUS"] is False

    data2 = np.asfortranarray(np.random.rand(6, 4))
    x3 = tensor(data2, chunk_size=2)
    y3 = x3.reshape(3, 8, order="F")
    res = y3.execute().fetch()
    expected = data2.reshape((3, 8), order="F")
    np.testing.assert_array_equal(res, expected)
    assert res.flags["F_CONTIGUOUS"] is True
    assert res.flags["C_CONTIGUOUS"] is False

    for chunk_size in [None, 3]:
        rs = np.random.RandomState(0)
        data = rs.rand(3, 4, 5)
        x = tensor(data, chunk_size=chunk_size)
        x = x[x[:, 0, 0] < 0.7]
        y = x.reshape(-1, 20)
        assert np.isnan(y.shape[0])
        res = y.execute().fetch()
        expected = data[data[:, 0, 0] < 0.7].reshape(-1, 20)
        np.testing.assert_array_equal(res, expected)


def test_shuffle_reshape_execution(setup):
    a = ones((31, 27), chunk_size=10)
    b = a.reshape(27, 31)
    b.op.extra_params["_reshape_with_shuffle"] = True

    res = b.execute().fetch()
    np.testing.assert_array_equal(res, np.ones((27, 31)))

    b2 = a.reshape(27, 31, order="F")
    b.op.extra_params["_reshape_with_shuffle"] = True
    res = b2.execute().fetch()
    assert res.flags["F_CONTIGUOUS"] is True
    assert res.flags["C_CONTIGUOUS"] is False

    data = np.random.rand(6, 4)
    x2 = tensor(data, chunk_size=2)
    y2 = x2.reshape(4, 6, order="F")
    y2.op.extra_params["_reshape_with_shuffle"] = True
    res = y2.execute().fetch()
    expected = data.reshape((4, 6), order="F")
    np.testing.assert_array_equal(res, expected)
    assert res.flags["F_CONTIGUOUS"] is True
    assert res.flags["C_CONTIGUOUS"] is False

    data2 = np.asfortranarray(np.random.rand(6, 4))
    x3 = tensor(data2, chunk_size=2)
    y3 = x3.reshape(4, 6)
    y3.op.extra_params["_reshape_with_shuffle"] = True
    res = y3.execute().fetch()
    expected = data2.reshape((4, 6))
    np.testing.assert_array_equal(res, expected)
    assert res.flags["C_CONTIGUOUS"] is True
    assert res.flags["F_CONTIGUOUS"] is False
