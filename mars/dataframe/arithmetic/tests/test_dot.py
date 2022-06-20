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

from ... import DataFrame, Series


def test_dot_execution(setup):
    df1_raw = pd.DataFrame(np.random.rand(4, 7))
    df2_raw = pd.DataFrame(np.random.rand(7, 5), columns=list("efghi"))
    s1_raw = pd.Series(np.random.rand(7))
    s2_raw = pd.Series(np.random.rand(7))

    df1 = DataFrame(df1_raw, chunk_size=(3, 2))
    df2 = DataFrame(df2_raw, chunk_size=(3, 4))

    # df.dot(df)
    r = df1.dot(df2)
    result = r.execute().fetch()
    expected = df1_raw.dot(df2_raw)
    pd.testing.assert_frame_equal(result, expected)

    # test @
    r = df1 @ df2
    result = r.execute().fetch()
    expected = df1_raw @ df2_raw
    pd.testing.assert_frame_equal(result, expected)

    # test reversed @
    r = df1_raw @ df2
    result = r.execute().fetch()
    expected = df1_raw @ df2_raw
    pd.testing.assert_frame_equal(result, expected)

    series1 = Series(s1_raw, chunk_size=5)

    # df.dot(series)
    r = df1.dot(series1)
    result = r.execute().fetch()
    expected = df1_raw.dot(s1_raw)
    pd.testing.assert_series_equal(result, expected)

    # df.dot(2d_array)
    r = df1.dot(df2_raw.to_numpy())
    result = r.execute().fetch()
    expected = df1_raw.dot(df2_raw.to_numpy())
    pd.testing.assert_frame_equal(result, expected)

    # df.dot(1d_array)
    r = df1.dot(s1_raw.to_numpy())
    result = r.execute().fetch()
    expected = df1_raw.dot(s1_raw.to_numpy())
    pd.testing.assert_series_equal(result, expected)

    series2 = Series(s2_raw, chunk_size=4)

    # series.dot(series)
    r = series1.dot(series2)
    result = r.execute().fetch()
    expected = s1_raw.dot(s2_raw)
    assert pytest.approx(result) == expected

    # series.dot(df)
    r = series1.dot(df2)
    result = r.execute().fetch()
    expected = s1_raw.dot(df2_raw)
    pd.testing.assert_series_equal(result, expected)

    # series.dot(2d_array)
    r = series1.dot(df2_raw.to_numpy())
    result = r.execute().fetch()
    expected = s1_raw.dot(df2_raw.to_numpy())
    np.testing.assert_almost_equal(result, expected)

    # series.dot(1d_array)
    r = series1.dot(s2_raw.to_numpy())
    result = r.execute().fetch()
    expected = s1_raw.dot(s2_raw.to_numpy())
    assert pytest.approx(result) == expected
