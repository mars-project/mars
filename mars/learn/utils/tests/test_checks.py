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
import pandas as pd
import pytest
import scipy.sparse as sps

from .... import tensor as mt
from .... import dataframe as md
from ....config import option_context
from ..checks import check_non_negative_then_return_value, assert_all_finite


def test_check_non_negative_then_return_value_execution(setup):
    raw = np.random.randint(10, size=(10, 5))
    c = mt.tensor(raw, chunk_size=(3, 2))

    r = check_non_negative_then_return_value(c, c, "sth")
    result = r.execute().fetch()
    np.testing.assert_array_equal(result, raw)

    raw = raw.copy()
    raw[1, 3] = -1
    c = mt.tensor(raw, chunk_size=(3, 2))

    r = check_non_negative_then_return_value(c, c, "sth")
    with pytest.raises(ValueError):
        _ = r.execute().fetch()

    raw = sps.random(10, 5, density=0.3, format="csr")
    c = mt.tensor(raw, chunk_size=(3, 2))

    r = check_non_negative_then_return_value(c, c, "sth")
    result = r.execute().fetch()
    np.testing.assert_array_equal(result.toarray(), raw.A)

    raw = raw.copy()
    raw[1, 3] = -1
    c = mt.tensor(raw, chunk_size=(3, 2))

    r = check_non_negative_then_return_value(c, c, "sth")
    with pytest.raises(ValueError):
        _ = r.execute().fetch()

    raw = pd.DataFrame(np.random.rand(10, 4))
    c = md.DataFrame(raw, chunk_size=(3, 2))

    r = check_non_negative_then_return_value(c, c, "sth")
    result = r.execute().fetch()

    pd.testing.assert_frame_equal(result, raw)

    raw = raw.copy()
    raw.iloc[1, 3] = -1
    c = md.DataFrame(raw, chunk_size=(3, 2))

    r = check_non_negative_then_return_value(c, c, "sth")
    with pytest.raises(ValueError):
        _ = r.execute().fetch()


def test_assert_all_finite(setup):
    raw = np.array([2.3, np.inf], dtype=np.float64)
    x = mt.tensor(raw)

    with pytest.raises(ValueError):
        r = assert_all_finite(x)
        r.execute()

    raw = np.array([2.3, np.nan], dtype=np.float64)
    x = mt.tensor(raw)

    with pytest.raises(ValueError):
        r = assert_all_finite(x, allow_nan=False)
        r.execute()

    max_float32 = np.finfo(np.float32).max
    raw = [max_float32] * 2
    assert not np.isfinite(np.sum(raw))
    x = mt.tensor(raw)

    r = assert_all_finite(x)
    result = r.execute().fetch()
    assert result is True

    raw = np.array([np.nan, "a"], dtype=object)
    x = mt.tensor(raw)

    with pytest.raises(ValueError):
        r = assert_all_finite(x)
        r.execute()

    raw = np.random.rand(10)
    x = mt.tensor(raw, chunk_size=2)

    r = assert_all_finite(x, check_only=False)
    result = r.execute().fetch()
    np.testing.assert_array_equal(result, raw)

    r = assert_all_finite(x)
    result = r.execute().fetch()
    assert result is True

    with option_context() as options:
        options.learn.assume_finite = True

        assert assert_all_finite(x) is None
        assert assert_all_finite(x, check_only=False) is x

    # test sparse
    s = sps.random(
        10, 3, density=0.1, format="csr", random_state=np.random.RandomState(0)
    )
    s[0, 2] = np.nan

    with pytest.raises(ValueError):
        r = assert_all_finite(s)
        r.execute()
