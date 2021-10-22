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


def test_dataframe_expanding_agg(setup):
    raw = pd.DataFrame(
        {
            "a": np.random.randint(100, size=(10,)),
            "b": np.random.rand(10),
            "c": np.random.randint(100, size=(10,)),
            "d": ["c" * i for i in np.random.randint(4, size=10)],
        }
    )
    raw.b[:3] = np.nan
    raw.b[5:7] = np.nan

    df = md.DataFrame(raw, chunk_size=(10, 3))

    r = df.expanding().agg(["sum"])
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.expanding().agg(["sum"]))

    df = md.DataFrame(raw, chunk_size=(3, 2))

    aggs = ["sum", "count", "min", "max", "mean", "var", "std"]

    for fun_name in aggs:
        r = df.expanding().agg(fun_name)
        pd.testing.assert_frame_equal(
            r.execute().fetch(), raw.expanding().agg(fun_name)
        )

    r = df.expanding().agg(["sum"])
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.expanding().agg(["sum"]))

    r = df.expanding().agg(aggs)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.expanding().agg(aggs))

    agg_dict = {"c": "sum"}
    r = df.expanding().agg(agg_dict)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.expanding().agg(agg_dict))

    agg_dict = OrderedDict([("a", ["sum", "var"]), ("b", "var")])
    r = df.expanding().agg(agg_dict)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.expanding().agg(agg_dict))

    r = df.expanding(0).agg(aggs)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.expanding(0).agg(aggs))

    r = df.expanding(2).agg(aggs)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.expanding(2).agg(aggs))

    agg_dict = OrderedDict([("a", ["min", "max"]), ("b", "max"), ("c", "sum")])
    r = df.expanding(2).agg(agg_dict)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.expanding(2).agg(agg_dict))


def test_series_expanding_agg(setup):
    raw = pd.Series(np.random.rand(10), name="a")
    raw[:3] = np.nan
    raw[5:7] = np.nan

    series = md.Series(raw, chunk_size=10)

    r = series.expanding().agg(["sum"])
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.expanding().agg(["sum"]))

    r = series.expanding().agg("sum")
    pd.testing.assert_series_equal(r.execute().fetch(), raw.expanding().agg("sum"))

    series = md.Series(raw, chunk_size=3)

    aggs = ["sum", "count", "min", "max", "mean", "var", "std"]

    for fun_name in aggs:
        r = series.expanding().agg(fun_name)
        pd.testing.assert_series_equal(
            r.execute().fetch(), raw.expanding().agg(fun_name)
        )

    r = series.expanding().agg(["sum"])
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.expanding().agg(["sum"]))

    r = series.expanding().agg(aggs)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.expanding().agg(aggs))

    r = series.expanding(2).agg(aggs)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.expanding(2).agg(aggs))

    r = series.expanding(0).agg(aggs)
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.expanding(0).agg(aggs))
