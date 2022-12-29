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

from collections import OrderedDict

import numpy as np
import pandas as pd

from ..... import dataframe as md


def test_dataframe_ewm_agg(setup):
    np.random.seed(0)

    raw = pd.DataFrame(
        {
            "a": np.random.randint(100, size=(10,)),
            "b": np.random.rand(10),
            "c": np.random.randint(100, size=(10,)),
            "d": ["c" * i for i in np.random.randint(4, size=10)],
        }
    )
    raw.b[0:3] = np.nan
    raw.b[5:7] = np.nan
    raw.b[9] = np.nan

    df = md.DataFrame(raw, chunk_size=(10, 3))

    r = df.ewm(alpha=0.5).agg("mean")
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.ewm(alpha=0.5).agg("mean"))

    r = df.ewm(alpha=0.5).agg(["mean"])
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.ewm(alpha=0.5).agg(["mean"]))

    df = md.DataFrame(raw, chunk_size=(3, 3))

    aggs = ["mean", "var", "std"]

    for fun_name in aggs:
        r = df.ewm(alpha=0.3).agg(fun_name)
        pd.testing.assert_frame_equal(
            r.execute().fetch(), raw.ewm(alpha=0.3).agg(fun_name)
        )

        r = df.ewm(alpha=0.3, ignore_na=True).agg(fun_name)
        pd.testing.assert_frame_equal(
            r.execute().fetch(), raw.ewm(alpha=0.3, ignore_na=True).agg(fun_name)
        )

    r = df.ewm(alpha=0.3).agg("mean")
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.ewm(alpha=0.3).agg("mean"))

    r = df.ewm(alpha=0.3).agg(["mean"])
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.ewm(alpha=0.3).agg(["mean"]))

    r = df.ewm(alpha=0.3).agg(aggs)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.ewm(alpha=0.3).agg(aggs))

    agg_dict = {"c": "mean"}
    r = df.ewm(alpha=0.3).agg(agg_dict)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.ewm(alpha=0.3).agg(agg_dict))

    agg_dict = OrderedDict([("a", ["mean", "var"]), ("b", "var")])
    r = df.ewm(alpha=0.3).agg(agg_dict)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.ewm(alpha=0.3).agg(agg_dict))

    r = df.ewm(alpha=0.3, min_periods=0).agg(aggs)
    pd.testing.assert_frame_equal(
        r.execute().fetch(), raw.ewm(alpha=0.3, min_periods=0).agg(aggs)
    )

    r = df.ewm(alpha=0.3, min_periods=2).agg(aggs)
    pd.testing.assert_frame_equal(
        r.execute().fetch(), raw.ewm(alpha=0.3, min_periods=2).agg(aggs)
    )

    agg_dict = OrderedDict([("a", ["mean", "var"]), ("b", "var"), ("c", "mean")])
    r = df.ewm(alpha=0.3, min_periods=2).agg(agg_dict)
    pd.testing.assert_frame_equal(
        r.execute().fetch(), raw.ewm(alpha=0.3, min_periods=2).agg(agg_dict)
    )


def test_series_expanding_agg(setup):
    raw = pd.Series(np.random.rand(10), name="a")
    raw[:3] = np.nan
    raw[5:10:2] = np.nan

    series = md.Series(raw, chunk_size=10)

    r = series.ewm(alpha=0.3).agg(["mean"])
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.ewm(alpha=0.3).agg(["mean"]))

    r = series.ewm(alpha=0.3).agg("mean")
    pd.testing.assert_series_equal(r.execute().fetch(), raw.ewm(alpha=0.3).agg("mean"))

    series = md.Series(raw, chunk_size=3)

    aggs = ["mean", "var", "std"]

    for fun_name in aggs:
        r = series.ewm(alpha=0.3).agg(fun_name)
        pd.testing.assert_series_equal(
            r.execute().fetch(), raw.ewm(alpha=0.3).agg(fun_name)
        )

        r = series.ewm(alpha=0.3, ignore_na=True).agg(fun_name)
        pd.testing.assert_series_equal(
            r.execute().fetch(), raw.ewm(alpha=0.3, ignore_na=True).agg(fun_name)
        )

    r = series.ewm(alpha=0.3).agg(["mean"])
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.ewm(alpha=0.3).agg(["mean"]))

    r = series.ewm(alpha=0.3).agg(aggs)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.ewm(alpha=0.3).agg(aggs))

    r = series.ewm(alpha=0.3, min_periods=0).agg(aggs)
    pd.testing.assert_frame_equal(
        r.execute().fetch(), raw.ewm(alpha=0.3, min_periods=0).agg(aggs)
    )

    r = series.ewm(alpha=0.3, min_periods=2).agg(aggs)
    pd.testing.assert_frame_equal(
        r.execute().fetch(), raw.ewm(alpha=0.3, min_periods=2).agg(aggs)
    )
