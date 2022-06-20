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

import operator
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ....core import enter_mode
from ...initializer import DataFrame, Series


def test_comp(setup):
    raw_df1 = pd.DataFrame(np.random.rand(4, 3))
    raw_df2 = pd.DataFrame(np.random.rand(4, 3))
    df1 = DataFrame(raw_df1)
    df2 = DataFrame(raw_df2)

    with enter_mode(build=True):
        assert not df1.data == df2.data
        assert df1.data == df1.data

    for op in [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.gt,
        operator.le,
        operator.ge,
    ]:
        eq_df = op(df1, df2)
        pd.testing.assert_index_equal(
            eq_df.index_value.to_pandas(), df1.index_value.to_pandas()
        )
        eq_df = op(raw_df1, df2)
        pd.testing.assert_index_equal(
            eq_df.index_value.to_pandas(), df1.index_value.to_pandas()
        )

        # index not identical
        df3 = DataFrame(pd.DataFrame(np.random.rand(4, 3), index=[1, 2, 3, 4]))
        with pytest.raises(ValueError):
            op(df1, df3)

        # columns not identical
        df4 = DataFrame(pd.DataFrame(np.random.rand(4, 3), columns=["a", "b", "c"]))
        with pytest.raises(ValueError):
            op(df1, df4)

    # test datetime
    df = DataFrame(pd.DataFrame(pd.date_range("20130101", periods=6)))
    for op in [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.gt,
        operator.le,
        operator.ge,
    ]:
        r_df = op(df, datetime(2013, 1, 2))
        pd.testing.assert_index_equal(
            r_df.index_value.to_pandas(), df.index_value.to_pandas()
        )

    # test period type
    raw = pd.period_range("2000-01-01", periods=10, freq="D")
    raw_series = pd.Series(raw)
    series = Series(raw, chunk_size=5)
    r = series >= series[1]
    pd.testing.assert_series_equal(r.to_pandas(), raw_series >= raw_series[1])
