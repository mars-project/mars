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

from ....core import tile
from ....tensor import Tensor
from ...core import Series, DataFrame
from ...datasource.dataframe import from_pandas as df_from_pandas
from ...datasource.series import from_pandas as series_from_pandas


def test_series_quantile():
    raw = pd.Series(np.random.rand(10))
    s = series_from_pandas(raw, chunk_size=3)

    r = s.quantile()
    assert isinstance(r, Tensor)
    tile(r)

    s = series_from_pandas(raw, chunk_size=3)

    r = s.quantile([0.3, 0.7])
    assert isinstance(r, Series)
    assert r.shape == (2,)
    pd.testing.assert_index_equal(r.index_value.to_pandas(), pd.Index([0.3, 0.7]))
    tile(r)


def test_dataframe_quantile():
    raw = pd.DataFrame(
        {
            "a": np.random.rand(10),
            "b": np.random.randint(1000, size=10),
            "c": [np.random.bytes(5) for _ in range(10)],
        }
    )
    s = df_from_pandas(raw, chunk_size=7)

    # q = 0.3, axis = 0
    r = s.quantile(0.3)
    e = raw.quantile(0.3)
    assert isinstance(r, Series)
    assert r.shape == (2,)
    assert r.dtype == e.dtype
    pd.testing.assert_index_equal(r.index_value.to_pandas(), e.index)

    tile(r)

    # q = 0.3, axis = 1
    r = s.quantile(0.3, axis=1)
    e = raw.quantile(0.3, axis=1)
    assert isinstance(r, Series)
    assert r.shape == e.shape
    assert r.dtype == e.dtype
    pd.testing.assert_index_equal(r.index_value.to_pandas(), e.index)

    tile(r)

    # q = [0.3, 0.7], axis = 0
    r = s.quantile([0.3, 0.7])
    e = raw.quantile([0.3, 0.7])
    assert isinstance(r, DataFrame)
    assert r.shape == e.shape
    pd.testing.assert_series_equal(r.dtypes, e.dtypes)
    pd.testing.assert_index_equal(r.index_value.to_pandas(), e.index)
    pd.testing.assert_index_equal(r.columns_value.to_pandas(), e.columns)

    tile(r)

    # q = [0.3, 0.7], axis = 1
    r = s.quantile([0.3, 0.7], axis=1)
    e = raw.quantile([0.3, 0.7], axis=1)
    assert isinstance(r, DataFrame)
    assert r.shape == e.shape
    pd.testing.assert_series_equal(r.dtypes, e.dtypes)
    pd.testing.assert_index_equal(r.index_value.to_pandas(), e.index)
    pd.testing.assert_index_equal(r.columns_value.to_pandas(), e.columns)

    tile(r)
