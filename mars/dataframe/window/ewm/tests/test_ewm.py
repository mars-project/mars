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

from ..... import dataframe as md
from .....core import tile


def test_ewm():
    df = pd.DataFrame(np.random.rand(4, 3), columns=list("abc"))
    df2 = md.DataFrame(df)

    with pytest.raises(NotImplementedError):
        _ = df2.ewm(2, adjust=False, ignore_na=False)

    with pytest.raises(ValueError):
        _ = df2.ewm()

    with pytest.raises(ValueError):
        _ = df2.ewm(com=2, alpha=0.3)

    assert pytest.approx(df2.ewm(com=1).alpha) == 0.5
    with pytest.raises(ValueError):
        _ = df2.ewm(com=-1)

    assert pytest.approx(df2.ewm(span=3).alpha) == 0.5
    with pytest.raises(ValueError):
        _ = df2.ewm(span=0)

    assert pytest.approx(df2.ewm(halflife=1).alpha) == 0.5
    with pytest.raises(ValueError):
        _ = df2.ewm(halflife=-1)

    with pytest.raises(ValueError):
        _ = df2.ewm(alpha=2)

    r = df2.ewm(3)
    expected = df.ewm(3)
    assert repr(r) == repr(expected)

    r = df2.ewm(alpha=1)
    assert type(r).__name__ == "Expanding"


def test_ewm_agg():
    df = pd.DataFrame(np.random.rand(4, 3), columns=list("abc"))
    df2 = md.DataFrame(df, chunk_size=3)

    with pytest.raises(NotImplementedError):
        _ = df2.ewm(span=3, axis=1).agg("mean")

    r = df2.ewm(span=3).agg("mean")
    expected = df.ewm(span=3).agg("mean")

    assert r.shape == df.shape
    assert r.index_value is df2.index_value
    pd.testing.assert_index_equal(r.columns_value.to_pandas(), expected.columns)
    pd.testing.assert_series_equal(r.dtypes, df2.dtypes)

    r = tile(r)
    for c in r.chunks:
        assert c.shape == c.inputs[0].shape
        assert c.index_value is c.inputs[0].index_value
        pd.testing.assert_index_equal(c.columns_value.to_pandas(), expected.columns)
        pd.testing.assert_series_equal(c.dtypes, expected.dtypes)

    aggs = ["mean", "var", "std"]
    for a in aggs:
        r = getattr(df2.ewm(span=3), a)()
        assert r.op.func == a
