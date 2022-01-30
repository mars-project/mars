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
import pytest

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

from .... import dataframe as md
from ....core.operand import OperandStage
from ....tests.core import assert_groupby_equal, require_cudf
from ....utils import arrow_array_to_objects, pd_release_version
from ..aggregation import DataFrameGroupByAgg

pytestmark = pytest.mark.pd_compat

_agg_size_as_frame = pd_release_version[:2] > (1, 0)


class MockReduction1(md.CustomReduction):
    def agg(self, v1):
        return v1.sum()


class MockReduction2(md.CustomReduction):
    def pre(self, value):
        return value + 1, value * 2

    def agg(self, v1, v2):
        return v1.sum(), v2.min()

    def post(self, v1, v2):
        return v1 + v2


def test_groupby(setup):
    rs = np.random.RandomState(0)
    data_size = 100
    data_dict = {
        "a": rs.randint(0, 10, size=(data_size,)),
        "b": rs.randint(0, 10, size=(data_size,)),
        "c": rs.choice(list("abcd"), size=(data_size,)),
    }

    # test groupby with DataFrames and RangeIndex
    df1 = pd.DataFrame(data_dict)
    mdf = md.DataFrame(df1, chunk_size=13)
    grouped = mdf.groupby("b")
    assert_groupby_equal(grouped.execute().fetch(), df1.groupby("b"))

    # test groupby with string index with duplications
    df2 = pd.DataFrame(data_dict, index=["i" + str(i % 3) for i in range(data_size)])
    mdf = md.DataFrame(df2, chunk_size=13)
    grouped = mdf.groupby("b")
    assert_groupby_equal(grouped.execute().fetch(), df2.groupby("b"))

    # test groupby with DataFrames by series
    grouped = mdf.groupby(mdf["b"])
    assert_groupby_equal(grouped.execute().fetch(), df2.groupby(df2["b"]))

    # test groupby with DataFrames by multiple series
    grouped = mdf.groupby(by=[mdf["b"], mdf["c"]])
    assert_groupby_equal(
        grouped.execute().fetch(), df2.groupby(by=[df2["b"], df2["c"]])
    )

    # test groupby with DataFrames with MultiIndex
    df3 = pd.DataFrame(
        data_dict,
        index=pd.MultiIndex.from_tuples(
            [(i % 3, "i" + str(i)) for i in range(data_size)]
        ),
    )
    mdf = md.DataFrame(df3, chunk_size=13)
    grouped = mdf.groupby(level=0)
    assert_groupby_equal(grouped.execute().fetch(), df3.groupby(level=0))

    # test groupby with DataFrames by integer columns
    df4 = pd.DataFrame(list(data_dict.values())).T
    mdf = md.DataFrame(df4, chunk_size=13)
    grouped = mdf.groupby(0)
    assert_groupby_equal(grouped.execute().fetch(), df4.groupby(0))

    series1 = pd.Series(data_dict["a"])
    ms1 = md.Series(series1, chunk_size=13)
    grouped = ms1.groupby(lambda x: x % 3)
    assert_groupby_equal(grouped.execute().fetch(), series1.groupby(lambda x: x % 3))

    # test groupby series
    grouped = ms1.groupby(ms1)
    assert_groupby_equal(grouped.execute().fetch(), series1.groupby(series1))

    series2 = pd.Series(data_dict["a"], index=["i" + str(i) for i in range(data_size)])
    ms2 = md.Series(series2, chunk_size=13)
    grouped = ms2.groupby(lambda x: int(x[1:]) % 3)
    assert_groupby_equal(
        grouped.execute().fetch(), series2.groupby(lambda x: int(x[1:]) % 3)
    )


def test_groupby_getitem(setup):
    rs = np.random.RandomState(0)
    data_size = 100
    raw = pd.DataFrame(
        {
            "a": rs.randint(0, 10, size=(data_size,)),
            "b": rs.randint(0, 10, size=(data_size,)),
            "c": rs.choice(list("abcd"), size=(data_size,)),
        },
        index=pd.MultiIndex.from_tuples(
            [(i % 3, "i" + str(i)) for i in range(data_size)]
        ),
    )
    mdf = md.DataFrame(raw, chunk_size=13)

    r = mdf.groupby(level=0)[["a", "b"]]
    assert_groupby_equal(
        r.execute().fetch(), raw.groupby(level=0)[["a", "b"]], with_selection=True
    )

    for method in ("tree", "shuffle"):
        r = mdf.groupby(level=0)[["a", "b"]].sum(method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            raw.groupby(level=0)[["a", "b"]].sum().sort_index(),
        )

    r = mdf.groupby(level=0)[["a", "b"]].apply(lambda x: x + 1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        raw.groupby(level=0)[["a", "b"]].apply(lambda x: x + 1).sort_index(),
    )

    r = mdf.groupby("b")[["a", "b"]]
    assert_groupby_equal(
        r.execute().fetch(), raw.groupby("b")[["a", "b"]], with_selection=True
    )

    r = mdf.groupby("b")[["a", "c"]]
    assert_groupby_equal(
        r.execute().fetch(), raw.groupby("b")[["a", "c"]], with_selection=True
    )

    for method in ("tree", "shuffle"):
        r = mdf.groupby("b")[["a", "b"]].sum(method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            raw.groupby("b")[["a", "b"]].sum().sort_index(),
        )

        r = mdf.groupby("b")[["a", "b"]].agg(["sum", "count"], method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            raw.groupby("b")[["a", "b"]].agg(["sum", "count"]).sort_index(),
        )

        r = mdf.groupby("b")[["a", "c"]].agg(["sum", "count"], method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            raw.groupby("b")[["a", "c"]].agg(["sum", "count"]).sort_index(),
        )

    r = mdf.groupby("b")[["a", "b"]].apply(lambda x: x + 1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        raw.groupby("b")[["a", "b"]].apply(lambda x: x + 1).sort_index(),
    )

    r = mdf.groupby("b")[["a", "b"]].transform(lambda x: x + 1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        raw.groupby("b")[["a", "b"]].transform(lambda x: x + 1).sort_index(),
    )

    r = mdf.groupby("b")[["a", "b"]].cumsum()
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        raw.groupby("b")[["a", "b"]].cumsum().sort_index(),
    )

    r = mdf.groupby("b").a
    assert_groupby_equal(r.execute().fetch(), raw.groupby("b").a, with_selection=True)

    for method in ("shuffle", "tree"):
        r = mdf.groupby("b").a.sum(method=method)
        pd.testing.assert_series_equal(
            r.execute().fetch().sort_index(), raw.groupby("b").a.sum().sort_index()
        )

        r = mdf.groupby("b").a.agg(["sum", "mean", "var"], method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            raw.groupby("b").a.agg(["sum", "mean", "var"]).sort_index(),
        )

        r = mdf.groupby("b", as_index=False).a.sum(method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_values("b", ignore_index=True),
            raw.groupby("b", as_index=False)
            .a.sum()
            .sort_values("b", ignore_index=True),
        )

        r = mdf.groupby("b", as_index=False).b.count(method=method)
        result = r.execute().fetch().sort_values("b", ignore_index=True)
        try:
            expected = (
                raw.groupby("b", as_index=False)
                .b.count()
                .sort_values("b", ignore_index=True)
            )
        except ValueError:
            expected = raw.groupby("b").b.count().to_frame()
            expected.index.names = [None] * expected.index.nlevels
            expected = expected.sort_values("b", ignore_index=True)
        pd.testing.assert_frame_equal(result, expected)

        r = mdf.groupby("b", as_index=False).b.agg({"cnt": "count"}, method=method)
        result = r.execute().fetch().sort_values("b", ignore_index=True)
        try:
            expected = (
                raw.groupby("b", as_index=False)
                .b.agg({"cnt": "count"})
                .sort_values("b", ignore_index=True)
            )
        except ValueError:
            expected = raw.groupby("b").b.agg({"cnt": "count"}).to_frame()
            expected.index.names = [None] * expected.index.nlevels
            expected = expected.sort_values("b", ignore_index=True)
        pd.testing.assert_frame_equal(result, expected)

    r = mdf.groupby("b").a.apply(lambda x: x + 1)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(),
        raw.groupby("b").a.apply(lambda x: x + 1).sort_index(),
    )

    r = mdf.groupby("b").a.transform(lambda x: x + 1)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(),
        raw.groupby("b").a.transform(lambda x: x + 1).sort_index(),
    )

    r = mdf.groupby("b").a.cumsum()
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(), raw.groupby("b").a.cumsum().sort_index()
    )

    # special test for selection key == 0
    raw = pd.DataFrame(rs.rand(data_size, 10))
    raw[0] = 0
    mdf = md.DataFrame(raw, chunk_size=13)
    r = mdf.groupby(0, as_index=False)[0].agg({"cnt": "count"}, method="tree")
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        raw.groupby(0, as_index=False)[0].agg({"cnt": "count"}),
    )

    # test groupby getitem then agg(#GH 2640)
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        {
            "c1": rs.randint(0, 10, size=(100,)).astype(np.int64),
            "c2": rs.choice(["a", "b", "c"], (100,)),
            "c3": rs.rand(100),
            "c4": rs.rand(100),
        }
    )
    mdf = md.DataFrame(raw, chunk_size=20)

    r = mdf.groupby(["c2"])[["c1", "c3"]].agg({"c1": "max", "c3": "min"}, method="tree")
    pd.testing.assert_frame_equal(
        r.execute().fetch(),
        raw.groupby(["c2"])[["c1", "c3"]].agg({"c1": "max", "c3": "min"}),
    )

    mdf = md.DataFrame(raw.copy(), chunk_size=30)
    r = mdf.groupby(["c2"])[["c1", "c4"]].agg(
        {"c1": "max", "c4": "mean"}, method="shuffle"
    )
    pd.testing.assert_frame_equal(
        r.execute().fetch(),
        raw.groupby(["c2"])[["c1", "c4"]].agg({"c1": "max", "c4": "mean"}),
    )

    # test anonymous function lists
    agg_funs = [lambda x: (x + 1).sum()]
    r = mdf.groupby(["c2"])["c1"].agg(agg_funs)
    pd.testing.assert_frame_equal(
        r.execute().fetch(), raw.groupby(["c2"])["c1"].agg(agg_funs)
    )

    # test group by multiple cols
    r = mdf.groupby(["c1", "c2"], as_index=False)["c3"].sum()
    expected = raw.groupby(["c1", "c2"], as_index=False)["c3"].sum()
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_values(["c1", "c2"]).reset_index(drop=True),
        expected.sort_values(["c1", "c2"]).reset_index(drop=True),
    )


def test_dataframe_groupby_agg(setup):
    agg_funs = [
        "std",
        "mean",
        "var",
        "max",
        "count",
        "size",
        "all",
        "any",
        "skew",
        "kurt",
        "sem",
    ]

    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        {
            "c1": np.arange(100).astype(np.int64),
            "c2": rs.choice(["a", "b", "c"], (100,)),
            "c3": rs.rand(100),
        }
    )
    mdf = md.DataFrame(raw, chunk_size=13)

    for method in ["tree", "shuffle"]:
        r = mdf.groupby("c2").agg("size", method=method)
        pd.testing.assert_series_equal(
            r.execute().fetch().sort_index(), raw.groupby("c2").agg("size").sort_index()
        )

        for agg_fun in agg_funs:
            if agg_fun == "size":
                continue
            r = mdf.groupby("c2").agg(agg_fun, method=method)
            pd.testing.assert_frame_equal(
                r.execute().fetch().sort_index(),
                raw.groupby("c2").agg(agg_fun).sort_index(),
            )

        r = mdf.groupby("c2").agg(agg_funs, method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            raw.groupby("c2").agg(agg_funs).sort_index(),
        )

        agg = OrderedDict([("c1", ["min", "mean"]), ("c3", "std")])
        r = mdf.groupby("c2").agg(agg, method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(), raw.groupby("c2").agg(agg).sort_index()
        )

        agg = OrderedDict([("c1", "min"), ("c3", "sum")])
        r = mdf.groupby("c2").agg(agg, method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(), raw.groupby("c2").agg(agg).sort_index()
        )

        r = mdf.groupby("c2").agg({"c1": "min", "c3": "min"}, method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            raw.groupby("c2").agg({"c1": "min", "c3": "min"}).sort_index(),
        )

        r = mdf.groupby("c2").agg({"c1": "min"}, method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            raw.groupby("c2").agg({"c1": "min"}).sort_index(),
        )

        # test groupby series
        r = mdf.groupby(mdf["c2"]).sum(method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(), raw.groupby(raw["c2"]).sum().sort_index()
        )

    r = mdf.groupby("c2").size(method="tree")
    pd.testing.assert_series_equal(r.execute().fetch(), raw.groupby("c2").size())

    # test inserted kurt method
    r = mdf.groupby("c2").kurtosis(method="tree")
    pd.testing.assert_frame_equal(r.execute().fetch(), raw.groupby("c2").kurtosis())

    for agg_fun in agg_funs:
        if agg_fun == "size" or callable(agg_fun):
            continue
        r = getattr(mdf.groupby("c2"), agg_fun)(method="tree")
        pd.testing.assert_frame_equal(
            r.execute().fetch(), getattr(raw.groupby("c2"), agg_fun)()
        )

    # test as_index=False
    for method in ["tree", "shuffle"]:
        r = mdf.groupby("c2", as_index=False).agg("size", method=method)
        if _agg_size_as_frame:
            result = r.execute().fetch().sort_values("c2", ignore_index=True)
            expected = (
                raw.groupby("c2", as_index=False)
                .agg("size")
                .sort_values("c2", ignore_index=True)
            )
            pd.testing.assert_frame_equal(result, expected)
        else:
            result = r.execute().fetch().sort_index()
            expected = raw.groupby("c2", as_index=False).agg("size").sort_index()
            pd.testing.assert_series_equal(result, expected)

        r = mdf.groupby("c2", as_index=False).agg("mean", method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_values("c2", ignore_index=True),
            raw.groupby("c2", as_index=False)
            .agg("mean")
            .sort_values("c2", ignore_index=True),
        )
        assert r.op.groupby_params["as_index"] is False

        r = mdf.groupby(["c1", "c2"], as_index=False).agg("mean", method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_values(["c1", "c2"], ignore_index=True),
            raw.groupby(["c1", "c2"], as_index=False)
            .agg("mean")
            .sort_values(["c1", "c2"], ignore_index=True),
        )
        assert r.op.groupby_params["as_index"] is False

    # test as_index=False takes no effect
    r = mdf.groupby(["c1", "c2"], as_index=False).agg(["mean", "count"])
    pd.testing.assert_frame_equal(
        r.execute().fetch(),
        raw.groupby(["c1", "c2"], as_index=False).agg(["mean", "count"]),
    )
    assert r.op.groupby_params["as_index"] is True

    r = mdf.groupby("c2").agg(["cumsum", "cumcount"])
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        raw.groupby("c2").agg(["cumsum", "cumcount"]).sort_index(),
    )

    r = mdf[["c1", "c3"]].groupby(mdf["c2"]).agg(MockReduction2())
    pd.testing.assert_frame_equal(
        r.execute().fetch(), raw[["c1", "c3"]].groupby(raw["c2"]).agg(MockReduction2())
    )

    r = mdf.groupby("c2").agg(
        sum_c1=md.NamedAgg("c1", "sum"),
        min_c1=md.NamedAgg("c1", "min"),
        mean_c3=md.NamedAgg("c3", "mean"),
        method="tree",
    )
    pd.testing.assert_frame_equal(
        r.execute().fetch(),
        raw.groupby("c2").agg(
            sum_c1=md.NamedAgg("c1", "sum"),
            min_c1=md.NamedAgg("c1", "min"),
            mean_c3=md.NamedAgg("c3", "mean"),
        ),
    )


def test_series_groupby_agg(setup):
    rs = np.random.RandomState(0)
    series1 = pd.Series(rs.rand(10))
    ms1 = md.Series(series1, chunk_size=3)

    agg_funs = [
        "std",
        "mean",
        "var",
        "max",
        "count",
        "size",
        "all",
        "any",
        "skew",
        "kurt",
        "sem",
    ]

    for method in ["tree", "shuffle"]:
        for agg_fun in agg_funs:
            r = ms1.groupby(lambda x: x % 2).agg(agg_fun, method=method)
            pd.testing.assert_series_equal(
                r.execute().fetch(), series1.groupby(lambda x: x % 2).agg(agg_fun)
            )

        r = ms1.groupby(lambda x: x % 2).agg(agg_funs, method=method)
        pd.testing.assert_frame_equal(
            r.execute().fetch(), series1.groupby(lambda x: x % 2).agg(agg_funs)
        )

        # test groupby series
        r = ms1.groupby(ms1).sum(method=method)
        pd.testing.assert_series_equal(
            r.execute().fetch().sort_index(),
            series1.groupby(series1).sum().sort_index(),
        )

        r = ms1.groupby(ms1).sum(method=method)
        pd.testing.assert_series_equal(
            r.execute().fetch().sort_index(),
            series1.groupby(series1).sum().sort_index(),
        )

    # test inserted kurt method
    r = ms1.groupby(ms1).kurtosis(method="tree")
    pd.testing.assert_series_equal(
        r.execute().fetch(), series1.groupby(series1).kurtosis()
    )

    for agg_fun in agg_funs:
        r = getattr(ms1.groupby(lambda x: x % 2), agg_fun)(method="tree")
        pd.testing.assert_series_equal(
            r.execute().fetch(), getattr(series1.groupby(lambda x: x % 2), agg_fun)()
        )

    r = ms1.groupby(lambda x: x % 2).agg(["cumsum", "cumcount"], method="tree")
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        series1.groupby(lambda x: x % 2).agg(["cumsum", "cumcount"]).sort_index(),
    )

    r = ms1.groupby(lambda x: x % 2).agg(MockReduction2(name="custom_r"), method="tree")
    pd.testing.assert_series_equal(
        r.execute().fetch(),
        series1.groupby(lambda x: x % 2).agg(MockReduction2(name="custom_r")),
    )

    r = ms1.groupby(lambda x: x % 2).agg(col_var="var", col_skew="skew", method="tree")
    pd.testing.assert_frame_equal(
        r.execute().fetch(),
        series1.groupby(lambda x: x % 2).agg(col_var="var", col_skew="skew"),
    )


def test_groupby_agg_auto_method(setup):
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        {
            "c1": rs.randint(20, size=100),
            "c2": rs.choice(["a", "b", "c"], (100,)),
            "c3": rs.rand(100),
        }
    )
    mdf = md.DataFrame(raw, chunk_size=20)

    def _disallow_reduce(ctx, op):
        assert op.stage != OperandStage.reduce
        op.execute(ctx, op)

    r = mdf.groupby("c2").agg("sum")
    operand_executors = {DataFrameGroupByAgg: _disallow_reduce}
    result = r.execute(
        extra_config={"operand_executors": operand_executors, "check_all": False}
    ).fetch()
    pd.testing.assert_frame_equal(result.sort_index(), raw.groupby("c2").agg("sum"))

    def _disallow_combine_and_agg(ctx, op):
        assert op.stage not in (OperandStage.combine, OperandStage.agg)
        op.execute(ctx, op)

    r = mdf.groupby("c1").agg("sum")
    operand_executors = {DataFrameGroupByAgg: _disallow_combine_and_agg}
    result = r.execute(
        extra_config={"operand_executors": operand_executors, "check_all": False}
    ).fetch()
    pd.testing.assert_frame_equal(result.sort_index(), raw.groupby("c1").agg("sum"))


def test_groupby_agg_str_cat(setup):
    agg_fun = lambda x: x.str.cat(sep="_", na_rep="NA")

    rs = np.random.RandomState(0)
    raw_df = pd.DataFrame(
        {
            "a": rs.choice(["A", "B", "C"], size=(100,)),
            "b": rs.choice([None, "alfa", "bravo", "charlie"], size=(100,)),
        }
    )

    mdf = md.DataFrame(raw_df, chunk_size=13)

    r = mdf.groupby("a").agg(agg_fun, method="tree")
    pd.testing.assert_frame_equal(r.execute().fetch(), raw_df.groupby("a").agg(agg_fun))

    raw_series = pd.Series(rs.choice([None, "alfa", "bravo", "charlie"], size=(100,)))

    ms = md.Series(raw_series, chunk_size=13)

    r = ms.groupby(lambda x: x % 2).agg(agg_fun, method="tree")
    pd.testing.assert_series_equal(
        r.execute().fetch(), raw_series.groupby(lambda x: x % 2).agg(agg_fun)
    )


@require_cudf
def test_gpu_groupby_agg(setup_gpu):
    rs = np.random.RandomState(0)
    df1 = pd.DataFrame(
        {"a": rs.choice([2, 3, 4], size=(100,)), "b": rs.choice([2, 3, 4], size=(100,))}
    )
    mdf = md.DataFrame(df1, chunk_size=13).to_gpu()

    r = mdf.groupby("a").sum()
    pd.testing.assert_frame_equal(
        r.execute().fetch().to_pandas(), df1.groupby("a").sum()
    )

    r = mdf.groupby("a").kurt()
    pd.testing.assert_frame_equal(
        r.execute().fetch().to_pandas(), df1.groupby("a").kurt()
    )

    r = mdf.groupby("a").agg(["sum", "var"])
    pd.testing.assert_frame_equal(
        r.execute().fetch().to_pandas(), df1.groupby("a").agg(["sum", "var"])
    )

    rs = np.random.RandomState(0)
    idx = pd.Index(np.where(rs.rand(10) > 0.5, "A", "B"))
    series1 = pd.Series(rs.rand(10), index=idx)
    ms = md.Series(series1, index=idx, chunk_size=3).to_gpu().to_gpu()

    r = ms.groupby(level=0).sum()
    pd.testing.assert_series_equal(
        r.execute().fetch().to_pandas(), series1.groupby(level=0).sum()
    )

    r = ms.groupby(level=0).kurt()
    pd.testing.assert_series_equal(
        r.execute().fetch().to_pandas(), series1.groupby(level=0).kurt()
    )

    r = ms.groupby(level=0).agg(["sum", "var"])
    pd.testing.assert_frame_equal(
        r.execute().fetch().to_pandas(), series1.groupby(level=0).agg(["sum", "var"])
    )


def test_groupby_apply(setup):
    df1 = pd.DataFrame(
        {
            "a": [3, 4, 5, 3, 5, 4, 1, 2, 3],
            "b": [1, 3, 4, 5, 6, 5, 4, 4, 4],
            "c": list("aabaaddce"),
        }
    )

    def apply_df(df, ret_series=False):
        df = df.sort_index()
        df.a += df.b
        if len(df.index) > 0:
            if not ret_series:
                df = df.iloc[:-1, :]
            else:
                df = df.iloc[-1, :]
        return df

    def apply_series(s, truncate=True):
        s = s.sort_index()
        if truncate and len(s.index) > 0:
            s = s.iloc[:-1]
        return s

    mdf = md.DataFrame(df1, chunk_size=3)

    applied = mdf.groupby("b").apply(lambda df: None)
    pd.testing.assert_frame_equal(
        applied.execute().fetch(), df1.groupby("b").apply(lambda df: None)
    )

    applied = mdf.groupby("b").apply(apply_df)
    pd.testing.assert_frame_equal(
        applied.execute().fetch().sort_index(),
        df1.groupby("b").apply(apply_df).sort_index(),
    )

    applied = mdf.groupby("b").apply(apply_df, ret_series=True)
    pd.testing.assert_frame_equal(
        applied.execute().fetch().sort_index(),
        df1.groupby("b").apply(apply_df, ret_series=True).sort_index(),
    )

    applied = mdf.groupby("b").apply(lambda df: df.a, output_type="series")
    pd.testing.assert_series_equal(
        applied.execute().fetch().sort_index(),
        df1.groupby("b").apply(lambda df: df.a).sort_index(),
    )

    applied = mdf.groupby("b").apply(lambda df: df.a.sum())
    pd.testing.assert_series_equal(
        applied.execute().fetch().sort_index(),
        df1.groupby("b").apply(lambda df: df.a.sum()).sort_index(),
    )

    series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
    ms1 = md.Series(series1, chunk_size=3)

    applied = ms1.groupby(lambda x: x % 3).apply(lambda df: None)
    pd.testing.assert_series_equal(
        applied.execute().fetch(),
        series1.groupby(lambda x: x % 3).apply(lambda df: None),
    )

    applied = ms1.groupby(lambda x: x % 3).apply(apply_series)
    pd.testing.assert_series_equal(
        applied.execute().fetch().sort_index(),
        series1.groupby(lambda x: x % 3).apply(apply_series).sort_index(),
    )

    sindex2 = pd.MultiIndex.from_arrays([list(range(9)), list("ABCDEFGHI")])
    series2 = pd.Series(list("CDECEDABC"), index=sindex2)
    ms2 = md.Series(series2, chunk_size=3)

    applied = ms2.groupby(lambda x: x[0] % 3).apply(apply_series)
    pd.testing.assert_series_equal(
        applied.execute().fetch().sort_index(),
        series2.groupby(lambda x: x[0] % 3).apply(apply_series).sort_index(),
    )


def test_groupby_transform(setup):
    df1 = pd.DataFrame(
        {
            "a": [3, 4, 5, 3, 5, 4, 1, 2, 3],
            "b": [1, 3, 4, 5, 6, 5, 4, 4, 4],
            "c": list("aabaaddce"),
            "d": [3, 4, 5, 3, 5, 4, 1, 2, 3],
            "e": [1, 3, 4, 5, 6, 5, 4, 4, 4],
            "f": list("aabaaddce"),
        }
    )

    def transform_series(s, truncate=True):
        s = s.sort_index()
        if truncate and len(s.index) > 1:
            s = s.iloc[:-1].reset_index(drop=True)
        return s

    mdf = md.DataFrame(df1, chunk_size=3)

    r = mdf.groupby("b").transform(transform_series, truncate=False)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        df1.groupby("b").transform(transform_series, truncate=False).sort_index(),
    )

    if pd.__version__ != "1.1.0":
        r = mdf.groupby("b").transform(["cummax", "cumsum"], _call_agg=True)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            df1.groupby("b").agg(["cummax", "cumsum"]).sort_index(),
        )

        agg_list = ["cummax", "cumsum"]
        r = mdf.groupby("b").transform(agg_list, _call_agg=True)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            df1.groupby("b").agg(agg_list).sort_index(),
        )

        agg_dict = OrderedDict([("d", "cummax"), ("b", "cumsum")])
        r = mdf.groupby("b").transform(agg_dict, _call_agg=True)
        pd.testing.assert_frame_equal(
            r.execute().fetch().sort_index(),
            df1.groupby("b").agg(agg_dict).sort_index(),
        )

    agg_list = ["sum", lambda s: s.sum()]
    r = mdf.groupby("b").transform(agg_list, _call_agg=True)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b").agg(agg_list).sort_index()
    )

    series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
    ms1 = md.Series(series1, chunk_size=3)

    r = ms1.groupby(lambda x: x % 3).transform(lambda x: x + 1)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(),
        series1.groupby(lambda x: x % 3).transform(lambda x: x + 1).sort_index(),
    )

    r = ms1.groupby(lambda x: x % 3).transform("cummax", _call_agg=True)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(),
        series1.groupby(lambda x: x % 3).agg("cummax").sort_index(),
    )

    agg_list = ["cummax", "cumcount"]
    r = ms1.groupby(lambda x: x % 3).transform(agg_list, _call_agg=True)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(),
        series1.groupby(lambda x: x % 3).agg(agg_list).sort_index(),
    )


def test_groupby_cum(setup):
    df1 = pd.DataFrame(
        {
            "a": [3, 5, 2, 7, 1, 2, 4, 6, 2, 4],
            "b": [8, 3, 4, 1, 8, 2, 2, 2, 2, 3],
            "c": [1, 8, 8, 5, 3, 5, 0, 0, 5, 4],
        }
    )
    mdf = md.DataFrame(df1, chunk_size=3)

    for fun in ["cummin", "cummax", "cumprod", "cumsum"]:
        r1 = getattr(mdf.groupby("b"), fun)()
        pd.testing.assert_frame_equal(
            r1.execute().fetch().sort_index(),
            getattr(df1.groupby("b"), fun)().sort_index(),
        )

        r2 = getattr(mdf.groupby("b"), fun)(axis=1)
        pd.testing.assert_frame_equal(
            r2.execute().fetch().sort_index(),
            getattr(df1.groupby("b"), fun)(axis=1).sort_index(),
        )

    r3 = mdf.groupby("b").cumcount()
    pd.testing.assert_series_equal(
        r3.execute().fetch().sort_index(), df1.groupby("b").cumcount().sort_index()
    )

    series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
    ms1 = md.Series(series1, chunk_size=3)

    for fun in ["cummin", "cummax", "cumprod", "cumsum", "cumcount"]:
        r1 = getattr(ms1.groupby(lambda x: x % 2), fun)()
        pd.testing.assert_series_equal(
            r1.execute().fetch().sort_index(),
            getattr(series1.groupby(lambda x: x % 2), fun)().sort_index(),
        )


def test_groupby_fill(setup):
    df1 = pd.DataFrame(
        [
            [1, 1, 10],
            [1, 1, np.nan],
            [1, 1, np.nan],
            [1, 2, np.nan],
            [1, 2, 20],
            [1, 2, np.nan],
            [1, 3, np.nan],
            [1, 3, np.nan],
        ],
        columns=["one", "two", "three"],
    )
    mdf = md.DataFrame(df1, chunk_size=3)
    r1 = getattr(mdf.groupby(["one", "two"]), "ffill")()
    pd.testing.assert_frame_equal(
        r1.execute().fetch().sort_index(),
        getattr(df1.groupby(["one", "two"]), "ffill")().sort_index(),
    )

    r2 = getattr(mdf.groupby("two"), "bfill")()
    pd.testing.assert_frame_equal(
        r2.execute().fetch().sort_index(),
        getattr(df1.groupby("two"), "bfill")().sort_index(),
    )

    r3 = getattr(mdf.groupby("one"), "fillna")(5)
    pd.testing.assert_frame_equal(
        r3.execute().fetch().sort_index(),
        getattr(df1.groupby("one"), "fillna")(5).sort_index(),
    )

    r4 = getattr(mdf.groupby("two"), "backfill")()
    pd.testing.assert_frame_equal(
        r4.execute().fetch().sort_index(),
        getattr(df1.groupby("two"), "backfill")().sort_index(),
    )

    s1 = pd.Series([4, 3, 9, np.nan, np.nan, 7, 10, 8, 1, 6])
    ms1 = md.Series(s1, chunk_size=3)

    r1 = getattr(ms1.groupby(lambda x: x % 2), "ffill")()
    pd.testing.assert_series_equal(
        r1.execute().fetch().sort_index(),
        getattr(s1.groupby(lambda x: x % 2), "ffill")().sort_index(),
    )

    r2 = getattr(ms1.groupby(lambda x: x % 2), "bfill")()
    pd.testing.assert_series_equal(
        r2.execute().fetch().sort_index(),
        getattr(s1.groupby(lambda x: x % 2), "bfill")().sort_index(),
    )

    r4 = getattr(ms1.groupby(lambda x: x % 2), "backfill")()
    pd.testing.assert_series_equal(
        r4.execute().fetch().sort_index(),
        getattr(s1.groupby(lambda x: x % 2), "backfill")().sort_index(),
    )


def test_groupby_head(setup):
    df1 = pd.DataFrame(
        {
            "a": [3, 5, 2, 7, 1, 2, 4, 6, 2, 4],
            "b": [8, 3, 4, 1, 8, 2, 2, 2, 2, 3],
            "c": [1, 8, 8, 5, 3, 5, 0, 0, 5, 4],
            "d": [9, 7, 6, 3, 6, 3, 2, 1, 5, 8],
        }
    )
    # test single chunk
    mdf = md.DataFrame(df1)

    r = mdf.groupby("b").head(1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b").head(1)
    )
    r = mdf.groupby("b").head(-1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b").head(-1)
    )
    r = mdf.groupby("b")["a", "c"].head(1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b")["a", "c"].head(1)
    )

    # test multiple chunks
    mdf = md.DataFrame(df1, chunk_size=3)

    r = mdf.groupby("b").head(1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b").head(1)
    )

    r = mdf.groupby("b").head(-1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b").head(-1)
    )

    # test head with selection
    r = mdf.groupby("b")["a", "d"].head(1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b")["a", "d"].head(1)
    )
    r = mdf.groupby("b")["c", "a", "d"].head(1)
    pd.testing.assert_frame_equal(
        r.execute().fetch().sort_index(), df1.groupby("b")["c", "a", "d"].head(1)
    )
    r = mdf.groupby("b")["c"].head(1)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(), df1.groupby("b")["c"].head(1)
    )

    # test single chunk
    series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3])
    ms = md.Series(series1)

    r = ms.groupby(lambda x: x % 2).head(1)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(), series1.groupby(lambda x: x % 2).head(1)
    )
    r = ms.groupby(lambda x: x % 2).head(-1)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(), series1.groupby(lambda x: x % 2).head(-1)
    )

    # test multiple chunk
    ms = md.Series(series1, chunk_size=3)

    r = ms.groupby(lambda x: x % 2).head(1)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(), series1.groupby(lambda x: x % 2).head(1)
    )

    # test with special index
    series1 = pd.Series([3, 4, 5, 3, 5, 4, 1, 2, 3], index=[4, 1, 2, 3, 5, 8, 6, 7, 9])
    ms = md.Series(series1, chunk_size=3)

    r = ms.groupby(lambda x: x % 2).head(1)
    pd.testing.assert_series_equal(
        r.execute().fetch().sort_index(),
        series1.groupby(lambda x: x % 2).head(1).sort_index(),
    )


def test_groupby_sample(setup):
    rs = np.random.RandomState(0)
    sample_count = 10
    src_data_list = []
    for b in range(5):
        data_count = int(rs.randint(20, 100))
        src_data_list.append(
            pd.DataFrame(
                {
                    "a": rs.randint(0, 100, size=data_count),
                    "b": np.array([b] * data_count),
                    "c": rs.randint(0, 100, size=data_count),
                    "d": rs.randint(0, 100, size=data_count),
                }
            )
        )
    df1 = pd.concat(src_data_list)
    shuffle_idx = np.arange(len(df1))
    rs.shuffle(shuffle_idx)
    df1 = df1.iloc[shuffle_idx].reset_index(drop=True)

    # test single chunk
    mdf = md.DataFrame(df1)

    r1 = mdf.groupby("b").sample(sample_count, random_state=rs)
    result1 = r1.execute().fetch()
    r2 = mdf.groupby("b").sample(sample_count, random_state=rs)
    result2 = r2.execute().fetch()
    pd.testing.assert_frame_equal(result1, result2)
    assert not (result1.groupby("b").count() - sample_count).any()[0]

    r1 = mdf.groupby("b").sample(
        sample_count, weights=df1["c"] / df1["c"].sum(), random_state=rs
    )
    result1 = r1.execute().fetch()
    r2 = mdf.groupby("b").sample(
        sample_count, weights=df1["c"] / df1["c"].sum(), random_state=rs
    )
    result2 = r2.execute().fetch()
    pd.testing.assert_frame_equal(result1, result2)
    assert not (result1.groupby("b").count() - sample_count).any()[0]

    r1 = mdf.groupby("b")[["b", "c"]].sample(sample_count, random_state=rs)
    result1 = r1.execute().fetch()
    r2 = mdf.groupby("b")[["b", "c"]].sample(sample_count, random_state=rs)
    result2 = r2.execute().fetch()
    pd.testing.assert_frame_equal(result1, result2)
    assert len(result1.columns) == 2
    assert not (result1.groupby("b").count() - sample_count).any()[0]

    r1 = mdf.groupby("b").c.sample(sample_count, random_state=rs)
    result1 = r1.execute().fetch()
    r2 = mdf.groupby("b").c.sample(sample_count, random_state=rs)
    result2 = r2.execute().fetch()
    pd.testing.assert_series_equal(result1, result2)

    r1 = mdf.groupby("b").c.sample(len(df1), random_state=rs)
    result1 = r1.execute().fetch()
    assert len(result1) == len(df1)

    with pytest.raises(ValueError):
        r1 = mdf.groupby("b").c.sample(len(df1), random_state=rs, errors="raises")
        r1.execute().fetch()

    # test multiple chunks
    mdf = md.DataFrame(df1, chunk_size=47)

    r1 = mdf.groupby("b").sample(sample_count, random_state=rs)
    result1 = r1.execute().fetch()
    r2 = mdf.groupby("b").sample(sample_count, random_state=rs)
    result2 = r2.execute().fetch()
    pd.testing.assert_frame_equal(result1, result2)
    assert not (result1.groupby("b").count() - sample_count).any()[0]

    r1 = mdf.groupby("b").sample(
        sample_count, weights=df1["c"] / df1["c"].sum(), random_state=rs
    )
    result1 = r1.execute().fetch()
    r2 = mdf.groupby("b").sample(
        sample_count, weights=df1["c"] / df1["c"].sum(), random_state=rs
    )
    result2 = r2.execute().fetch()
    pd.testing.assert_frame_equal(result1, result2)
    assert not (result1.groupby("b").count() - sample_count).any()[0]

    r1 = mdf.groupby("b")[["b", "c"]].sample(sample_count, random_state=rs)
    result1 = r1.execute().fetch()
    r2 = mdf.groupby("b")[["b", "c"]].sample(sample_count, random_state=rs)
    result2 = r2.execute().fetch()
    pd.testing.assert_frame_equal(result1, result2)
    assert len(result1.columns) == 2
    assert not (result1.groupby("b").count() - sample_count).any()[0]

    r1 = mdf.groupby("b").c.sample(sample_count, random_state=rs)
    result1 = r1.execute().fetch()
    r2 = mdf.groupby("b").c.sample(sample_count, random_state=rs)
    result2 = r2.execute().fetch()
    pd.testing.assert_series_equal(result1, result2)

    r1 = mdf.groupby("b").c.sample(len(df1), random_state=rs)
    result1 = r1.execute().fetch()
    assert len(result1) == len(df1)

    with pytest.raises(ValueError):
        r1 = mdf.groupby("b").c.sample(len(df1), random_state=rs, errors="raises")
        r1.execute().fetch()


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_groupby_agg_with_arrow_dtype(setup):
    df1 = pd.DataFrame({"a": [1, 2, 1], "b": ["a", "b", "a"]})
    mdf = md.DataFrame(df1)
    mdf["b"] = mdf["b"].astype("Arrow[string]")

    r = mdf.groupby("a").count()
    result = r.execute().fetch()
    expected = df1.groupby("a").count()
    pd.testing.assert_frame_equal(result, expected)

    r = mdf.groupby("b").count()
    result = r.execute().fetch()
    result.index = result.index.astype(object)
    expected = df1.groupby("b").count()
    pd.testing.assert_frame_equal(result, expected)

    series1 = df1["b"]
    mseries = md.Series(series1).astype("Arrow[string]")

    r = mseries.groupby(mseries).count()
    result = r.execute().fetch()
    result.index = result.index.astype(object)
    expected = series1.groupby(series1).count()
    pd.testing.assert_series_equal(result, expected)

    series2 = series1.copy()
    series2.index = pd.MultiIndex.from_tuples([(0, 1), (2, 3), (4, 5)])
    mseries = md.Series(series2).astype("Arrow[string]")

    r = mseries.groupby(mseries).count()
    result = r.execute().fetch()
    result.index = result.index.astype(object)
    expected = series2.groupby(series2).count()
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_groupby_apply_with_arrow_dtype(setup):
    df1 = pd.DataFrame({"a": [1, 2, 1], "b": ["a", "b", "a"]})
    mdf = md.DataFrame(df1)
    mdf["b"] = mdf["b"].astype("Arrow[string]")

    applied = mdf.groupby("b").apply(lambda df: df.a.sum())
    result = applied.execute().fetch()
    result.index = result.index.astype(object)
    expected = df1.groupby("b").apply(lambda df: df.a.sum())
    pd.testing.assert_series_equal(result, expected)

    series1 = df1["b"]
    mseries = md.Series(series1).astype("Arrow[string]")

    applied = mseries.groupby(mseries).apply(lambda s: s)
    result = applied.execute().fetch()
    result.index = result.index.astype(np.int64)
    expected = series1.groupby(series1).apply(lambda s: s)
    pd.testing.assert_series_equal(arrow_array_to_objects(result), expected)


def test_groupby_nunique(setup):
    rs = np.random.RandomState(0)
    data_size = 100
    data_dict = {
        "a": rs.randint(0, 10, size=(data_size,)),
        "b": rs.choice(list("abcd"), size=(data_size,)),
        "c": rs.choice(list("abcd"), size=(data_size,)),
    }
    df1 = pd.DataFrame(data_dict)

    # one chunk
    mdf = md.DataFrame(df1)
    pd.testing.assert_frame_equal(
        mdf.groupby("c").nunique().execute().fetch().sort_index(),
        df1.groupby("c").nunique().sort_index(),
    )

    # multiple chunks
    mdf = md.DataFrame(df1, chunk_size=13)
    pd.testing.assert_frame_equal(
        mdf.groupby("b").nunique().execute().fetch().sort_index(),
        df1.groupby("b").nunique().sort_index(),
    )

    # getitem and nunique
    mdf = md.DataFrame(df1, chunk_size=13)
    pd.testing.assert_series_equal(
        mdf.groupby("b")["a"].nunique().execute().fetch().sort_index(),
        df1.groupby("b")["a"].nunique().sort_index(),
    )

    # test with as_index=False
    mdf = md.DataFrame(df1, chunk_size=13)
    if _agg_size_as_frame:
        pd.testing.assert_frame_equal(
            mdf.groupby("b", as_index=False)["a"]
            .nunique()
            .execute()
            .fetch()
            .sort_values(by="b", ignore_index=True),
            df1.groupby("b", as_index=False)["a"]
            .nunique()
            .sort_values(by="b", ignore_index=True),
        )
