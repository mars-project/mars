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

from typing import NamedTuple

import numpy as np
import pytest
import pandas as pd

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

from .... import dataframe as md
from ....config import option_context
from ....deploy.oscar.session import get_default_session
from ....tests.core import require_cudf, require_cupy
from ....utils import lazy_import, pd_release_version
from ... import CustomReduction, NamedAgg
from ...base import to_gpu

pytestmark = pytest.mark.pd_compat

cp = lazy_import("cupy", rename="cp", globals=globals())
_agg_size_as_series = pd_release_version >= (1, 3)
_support_kw_agg = pd_release_version >= (1, 1)


@pytest.fixture
def check_ref_counts():
    yield
    import gc

    gc.collect()
    sess = get_default_session()
    assert len(sess._get_ref_counts()) == 0


class FunctionOptions(NamedTuple):
    has_min_count: bool = False


reduction_functions = [
    ("sum", FunctionOptions(has_min_count=True)),
    ("prod", FunctionOptions(has_min_count=True)),
    ("min", FunctionOptions()),
    ("max", FunctionOptions()),
    ("mean", FunctionOptions()),
    ("var", FunctionOptions()),
    ("std", FunctionOptions()),
    ("sem", FunctionOptions()),
    ("skew", FunctionOptions()),
    ("kurt", FunctionOptions()),
]


@pytest.mark.parametrize("func_name,func_opts", reduction_functions)
def test_series_reduction(
    setup, check_ref_counts, func_name, func_opts: FunctionOptions
):
    def compute(data, **kwargs):
        return getattr(data, func_name)(**kwargs)

    rs = np.random.RandomState(0)
    data = pd.Series(
        rs.randint(0, 8, (10,)), index=[str(i) for i in range(10)], name="a"
    )
    r = compute(md.Series(data))
    assert pytest.approx(compute(data)) == r.execute().fetch()

    r = compute(md.Series(data, chunk_size=6))
    assert pytest.approx(compute(data)) == r.execute().fetch()

    r = compute(md.Series(data, chunk_size=3))
    assert pytest.approx(compute(data)) == r.execute().fetch()

    r = compute(md.Series(data, chunk_size=4), axis="index")
    assert pytest.approx(compute(data, axis="index")) == r.execute().fetch()

    r = compute(md.Series(data, chunk_size=4), axis="index")
    assert pytest.approx(compute(data, axis="index")) == r.execute().fetch()

    data = pd.Series(rs.rand(20), name="a")
    data[0] = 0.1  # make sure not all elements are NAN
    data[data > 0.5] = np.nan
    r = compute(md.Series(data, chunk_size=3))
    assert pytest.approx(compute(data)) == r.execute().fetch()

    r = compute(md.Series(data, chunk_size=3), skipna=False)
    assert np.isnan(r.execute().fetch())

    if func_opts.has_min_count:
        r = compute(md.Series(data, chunk_size=3), skipna=False, min_count=2)
        assert np.isnan(r.execute().fetch())

        r = compute(md.Series(data, chunk_size=3), min_count=1)
        assert pytest.approx(compute(data, min_count=1)) == r.execute().fetch()

        reduction_df5 = compute(md.Series(data, chunk_size=3), min_count=21)
        assert np.isnan(reduction_df5.execute().fetch())

    # test reduction on empty series
    data = pd.Series([], dtype=float, name="a")
    r = compute(md.Series(data))
    np.testing.assert_equal(r.execute().fetch(), compute(data))


@pytest.mark.parametrize("func_name,func_opts", reduction_functions)
def test_series_level_reduction(setup, func_name, func_opts: FunctionOptions):
    def compute(data, **kwargs):
        return getattr(data, func_name)(**kwargs)

    rs = np.random.RandomState(0)
    idx = pd.MultiIndex.from_arrays(
        [[str(i) for i in range(100)], rs.choice(["A", "B"], size=(100,))],
        names=["a", "b"],
    )
    data = pd.Series(rs.randint(0, 8, size=(100,)), index=idx)

    r = compute(md.Series(data, chunk_size=13), level=1, method="tree")
    pd.testing.assert_series_equal(
        compute(data, level=1).sort_index(), r.execute().fetch().sort_index()
    )

    # test null
    data = pd.Series(rs.rand(100), name="a", index=idx)
    idx_df = idx.to_frame()
    data[data > 0.5] = np.nan
    data[int(idx_df[idx_df.b == "A"].iloc[0, 0])] = 0.1
    data[int(idx_df[idx_df.b == "B"].iloc[0, 0])] = 0.1

    r = compute(md.Series(data, chunk_size=13), level=1, method="tree")
    pd.testing.assert_series_equal(
        compute(data, level=1).sort_index(), r.execute().fetch().sort_index()
    )

    r = compute(md.Series(data, chunk_size=13), level=1, skipna=False, method="tree")
    pd.testing.assert_series_equal(
        compute(data, level=1, skipna=False).sort_index(),
        r.execute().fetch().sort_index(),
    )

    if func_opts.has_min_count:
        r = compute(md.Series(data, chunk_size=13), min_count=1, level=1, method="tree")
        pd.testing.assert_series_equal(
            compute(data, min_count=1, level=1).sort_index(),
            r.execute().fetch().sort_index(),
        )


@pytest.mark.parametrize("func_name,func_opts", reduction_functions)
def test_dataframe_reduction(
    setup, check_ref_counts, func_name, func_opts: FunctionOptions
):
    def compute(data, **kwargs):
        return getattr(data, func_name)(**kwargs)

    rs = np.random.RandomState(0)
    data = pd.DataFrame(rs.rand(20, 10))
    r = compute(md.DataFrame(data))
    pd.testing.assert_series_equal(compute(data), r.execute().fetch())

    r = compute(md.DataFrame(data, chunk_size=3))
    pd.testing.assert_series_equal(compute(data), r.execute().fetch())

    r = compute(md.DataFrame(data, chunk_size=6), axis="index", numeric_only=True)
    pd.testing.assert_series_equal(
        compute(data, axis="index", numeric_only=True), r.execute().fetch()
    )

    r = compute(md.DataFrame(data, chunk_size=3), axis=1)
    pd.testing.assert_series_equal(compute(data, axis=1), r.execute().fetch())

    # test null
    np_data = rs.rand(20, 10)
    np_data[np_data > 0.6] = np.nan
    data = pd.DataFrame(np_data)

    r = compute(md.DataFrame(data, chunk_size=3))
    pd.testing.assert_series_equal(compute(data), r.execute().fetch())

    r = compute(md.DataFrame(data, chunk_size=3), skipna=False)
    pd.testing.assert_series_equal(compute(data, skipna=False), r.execute().fetch())

    r = compute(md.DataFrame(data, chunk_size=3), skipna=False)
    pd.testing.assert_series_equal(compute(data, skipna=False), r.execute().fetch())

    if func_opts.has_min_count:
        r = compute(md.DataFrame(data, chunk_size=3), min_count=15)
        pd.testing.assert_series_equal(compute(data, min_count=15), r.execute().fetch())

        r = compute(md.DataFrame(data, chunk_size=3), min_count=3)
        pd.testing.assert_series_equal(compute(data, min_count=3), r.execute().fetch())

        r = compute(md.DataFrame(data, chunk_size=3), axis=1, min_count=3)
        pd.testing.assert_series_equal(
            compute(data, axis=1, min_count=3), r.execute().fetch()
        )

        r = compute(md.DataFrame(data, chunk_size=3), axis=1, min_count=8)
        pd.testing.assert_series_equal(
            compute(data, axis=1, min_count=8), r.execute().fetch()
        )

    # test numeric_only
    data = pd.DataFrame(
        rs.rand(10, 10),
        index=rs.randint(-100, 100, size=(10,)),
        columns=[rs.bytes(10) for _ in range(10)],
    )
    r = compute(md.DataFrame(data, chunk_size=2))
    pd.testing.assert_series_equal(compute(data), r.execute().fetch())

    r = compute(md.DataFrame(data, chunk_size=6), axis="index", numeric_only=True)
    pd.testing.assert_series_equal(
        compute(data, axis="index", numeric_only=True), r.execute().fetch()
    )

    r = compute(md.DataFrame(data, chunk_size=3), axis="columns")
    pd.testing.assert_series_equal(compute(data, axis="columns"), r.execute().fetch())

    data_dict = dict((str(i), rs.rand(10)) for i in range(10))
    data_dict["string"] = pd.Series([str(i) for i in range(10)]).radd("O")
    data_dict["bool"] = rs.choice([True, False], (10,))
    data = pd.DataFrame(data_dict)
    r = compute(md.DataFrame(data, chunk_size=3), axis="index", numeric_only=True)
    pd.testing.assert_series_equal(
        compute(data, axis="index", numeric_only=True), r.execute().fetch()
    )

    data1 = pd.DataFrame(rs.rand(10, 10), columns=[str(i) for i in range(10)])
    data2 = pd.DataFrame(rs.rand(10, 10), columns=[str(i) for i in range(10)])
    df = md.DataFrame(data1, chunk_size=5) + md.DataFrame(data2, chunk_size=6)
    r = compute(df)
    pd.testing.assert_series_equal(
        compute(data1 + data2).sort_index(), r.execute().fetch().sort_index()
    )


@pytest.mark.parametrize("func_name,func_opts", reduction_functions)
def test_dataframe_level_reduction(
    setup, check_ref_counts, func_name, func_opts: FunctionOptions
):
    def compute(data, **kwargs):
        return getattr(data, func_name)(**kwargs)

    rs = np.random.RandomState(0)
    idx = pd.MultiIndex.from_arrays(
        [[str(i) for i in range(100)], rs.choice(["A", "B"], size=(100,))],
        names=["a", "b"],
    )
    data = pd.DataFrame(rs.rand(100, 10), index=idx)

    r = compute(md.DataFrame(data, chunk_size=13), level=1, method="tree")
    pd.testing.assert_frame_equal(
        compute(data, level=1).sort_index(), r.execute().fetch().sort_index()
    )

    r = compute(
        md.DataFrame(data, chunk_size=13), level=1, numeric_only=True, method="tree"
    )
    pd.testing.assert_frame_equal(
        compute(data, numeric_only=True, level=1).sort_index(),
        r.execute().fetch().sort_index(),
    )

    # test null
    data = pd.DataFrame(rs.rand(100, 10), index=idx)
    data[data > 0.6] = np.nan

    r = compute(md.DataFrame(data, chunk_size=13), level=1, method="tree")
    pd.testing.assert_frame_equal(
        compute(data, level=1).sort_index(), r.execute().fetch().sort_index()
    )

    r = compute(md.DataFrame(data, chunk_size=13), level=1, skipna=False, method="tree")
    pd.testing.assert_frame_equal(
        compute(data, level=1, skipna=False).sort_index(),
        r.execute().fetch().sort_index(),
    )

    if func_opts.has_min_count:
        r = compute(
            md.DataFrame(data, chunk_size=13), level=1, min_count=10, method="tree"
        )
        pd.testing.assert_frame_equal(
            compute(data, level=1, min_count=10).sort_index(),
            r.execute().fetch().sort_index(),
        )

    # behavior of 'skew', 'kurt' differs for cases with and without level
    skip_funcs = ("skew", "kurt")
    if pd_release_version <= (1, 2, 0):
        # fails under pandas 1.2. see pandas-dev/pandas#38774 for more details
        skip_funcs += ("sem",)

    if func_name not in skip_funcs:
        data_dict = dict((str(i), rs.rand(100)) for i in range(10))
        data_dict["string"] = ["O" + str(i) for i in range(100)]
        data_dict["bool"] = rs.choice([True, False], (100,))
        data = pd.DataFrame(data_dict, index=idx)

        r = compute(
            md.DataFrame(data, chunk_size=13), level=1, numeric_only=True, method="tree"
        )
        pd.testing.assert_frame_equal(
            compute(data, level=1, numeric_only=True).sort_index(),
            r.execute().fetch().sort_index(),
        )


@require_cudf
@require_cupy
def test_gpu_execution(setup_gpu, check_ref_counts):
    df_raw = pd.DataFrame(np.random.rand(30, 3), columns=list("abc"))
    df = to_gpu(md.DataFrame(df_raw, chunk_size=6))

    r = df.sum()
    res = r.execute().fetch()
    pd.testing.assert_series_equal(res.to_pandas(), df_raw.sum())

    r = df.kurt()
    res = r.execute().fetch()
    pd.testing.assert_series_equal(res.to_pandas(), df_raw.kurt())

    r = df.agg(["sum", "var"])
    res = r.execute().fetch()
    pd.testing.assert_frame_equal(res.to_pandas(), df_raw.agg(["sum", "var"]))

    s_raw = pd.Series(np.random.rand(30))
    s = to_gpu(md.Series(s_raw, chunk_size=6))

    r = s.sum()
    res = r.execute().fetch()
    assert pytest.approx(res) == s_raw.sum()

    r = s.kurt()
    res = r.execute().fetch()
    assert pytest.approx(res) == s_raw.kurt()

    r = s.agg(["sum", "var"])
    res = r.execute().fetch()
    pd.testing.assert_series_equal(res.to_pandas(), s_raw.agg(["sum", "var"]))

    s_raw = pd.Series(
        np.random.randint(0, 3, size=(30,)) * np.random.randint(0, 5, size=(30,))
    )
    s = to_gpu(md.Series(s_raw, chunk_size=6))

    r = s.unique()
    res = r.execute().fetch()
    np.testing.assert_array_equal(cp.asnumpy(res).sort(), s_raw.unique().sort())


bool_reduction_functions = ["all", "any"]


@pytest.mark.parametrize("func_name", bool_reduction_functions)
def test_series_bool_reduction(setup, check_ref_counts, func_name):
    def compute(data, **kwargs):
        return getattr(data, func_name)(**kwargs)

    rs = np.random.RandomState(0)
    data = pd.Series(rs.rand(10) > 0.5, index=[str(i) for i in range(10)], name="a")
    r = compute(md.Series(data))
    assert compute(data) == r.execute().fetch()

    r = compute(md.Series(data, chunk_size=6))
    assert pytest.approx(compute(data)) == r.execute().fetch()

    r = compute(md.Series(data, chunk_size=3))
    assert pytest.approx(compute(data)) == r.execute().fetch()

    r = compute(md.Series(data, chunk_size=4), axis="index")
    assert pytest.approx(compute(data, axis="index")) == r.execute().fetch()

    # test null
    data = pd.Series(rs.rand(20), name="a")
    data[0] = 0.1  # make sure not all elements are NAN
    data[data > 0.5] = np.nan
    r = compute(md.Series(data, chunk_size=3))
    assert compute(data) == r.execute().fetch()

    r = compute(md.Series(data, chunk_size=3), skipna=False)
    assert r.execute().fetch() is True


@pytest.mark.parametrize("func_name", bool_reduction_functions)
def test_series_bool_level_reduction(setup, check_ref_counts, func_name):
    def compute(data, **kwargs):
        return getattr(data, func_name)(**kwargs)

    rs = np.random.RandomState(0)
    idx = pd.MultiIndex.from_arrays(
        [[str(i) for i in range(100)], rs.choice(["A", "B"], size=(100,))],
        names=["a", "b"],
    )
    data = pd.Series(rs.randint(0, 8, size=(100,)), index=idx)

    r = compute(md.Series(data, chunk_size=13), level=1, method="tree")
    pd.testing.assert_series_equal(
        compute(data, level=1).sort_index(), r.execute().fetch().sort_index()
    )

    # test null
    data = pd.Series(rs.rand(100), name="a", index=idx)
    idx_df = idx.to_frame()
    data[data > 0.5] = np.nan
    data[int(idx_df[idx_df.b == "A"].iloc[0, 0])] = 0.1
    data[int(idx_df[idx_df.b == "B"].iloc[0, 0])] = 0.1

    r = compute(md.Series(data, chunk_size=13), level=1, method="tree")
    pd.testing.assert_series_equal(
        compute(data, level=1).sort_index(), r.execute().fetch().sort_index()
    )

    r = compute(md.Series(data, chunk_size=13), level=1, skipna=False, method="tree")
    pd.testing.assert_series_equal(
        compute(data, level=1, skipna=False).sort_index(),
        r.execute().fetch().sort_index(),
    )


@pytest.mark.parametrize("func_name", bool_reduction_functions)
def test_dataframe_bool_reduction(setup, check_ref_counts, func_name):
    def compute(data, **kwargs):
        return getattr(data, func_name)(**kwargs)

    rs = np.random.RandomState(0)
    data = pd.DataFrame(rs.rand(20, 10))
    data.iloc[:, :5] = data.iloc[:, :5] > 0.5
    r = compute(md.DataFrame(data))
    pd.testing.assert_series_equal(compute(data), r.execute().fetch())

    r = compute(md.DataFrame(data, chunk_size=3))
    pd.testing.assert_series_equal(compute(data), r.execute().fetch())

    r = compute(
        md.DataFrame(data, chunk_size=6), axis="index", bool_only=True, method="tree"
    )
    pd.testing.assert_series_equal(
        compute(data, axis="index", bool_only=True),
        r.execute(extra_config={"check_all": False}).fetch(),
    )

    r = compute(md.DataFrame(data, chunk_size=3), axis=1)
    pd.testing.assert_series_equal(compute(data, axis=1), r.execute().fetch())

    # test null
    np_data = rs.rand(20, 10)
    np_data[np_data > 0.6] = np.nan
    data = pd.DataFrame(np_data)

    r = compute(md.DataFrame(data, chunk_size=3))
    pd.testing.assert_series_equal(compute(data), r.execute().fetch())

    r = compute(md.DataFrame(data, chunk_size=3), skipna=False)
    pd.testing.assert_series_equal(compute(data, skipna=False), r.execute().fetch())

    r = compute(md.DataFrame(data, chunk_size=3), skipna=False)
    pd.testing.assert_series_equal(compute(data, skipna=False), r.execute().fetch())

    # test bool_only
    data = pd.DataFrame(
        rs.rand(10, 10),
        index=rs.randint(-100, 100, size=(10,)),
        columns=[rs.bytes(10) for _ in range(10)],
    )
    data.iloc[:, :5] = data.iloc[:, :5] > 0.5
    data.iloc[:5, 5:] = data.iloc[:5, 5:] > 0.5
    r = compute(md.DataFrame(data, chunk_size=2))
    pd.testing.assert_series_equal(compute(data), r.execute().fetch())

    r = compute(md.DataFrame(data, chunk_size=6), axis="index", bool_only=True)
    pd.testing.assert_series_equal(
        compute(data, axis="index", bool_only=True), r.execute().fetch()
    )

    r = compute(md.DataFrame(data, chunk_size=3), axis="columns")
    pd.testing.assert_series_equal(compute(data, axis="columns"), r.execute().fetch())

    data_dict = dict((str(i), rs.rand(10)) for i in range(10))
    data_dict["string"] = [str(i) for i in range(10)]
    data_dict["bool"] = rs.choice([True, False], (10,))
    data = pd.DataFrame(data_dict)
    r = compute(md.DataFrame(data, chunk_size=3), axis="index", bool_only=True)
    pd.testing.assert_series_equal(
        compute(data, axis="index", bool_only=True), r.execute().fetch()
    )


@pytest.mark.parametrize("func_name", bool_reduction_functions)
def test_dataframe_bool_level_reduction(setup, check_ref_counts, func_name):
    def compute(data, **kwargs):
        return getattr(data, func_name)(**kwargs)

    rs = np.random.RandomState(0)
    idx = pd.MultiIndex.from_arrays(
        [[str(i) for i in range(100)], rs.choice(["A", "B"], size=(100,))],
        names=["a", "b"],
    )
    data = pd.DataFrame(rs.rand(100, 10), index=idx)
    data.iloc[:, :5] = data.iloc[:, :5] > 0.5

    r = compute(md.DataFrame(data, chunk_size=13), level=1, method="tree")
    pd.testing.assert_frame_equal(
        compute(data, level=1).sort_index(), r.execute().fetch().sort_index()
    )

    # test null
    data = pd.DataFrame(rs.rand(100, 10), index=idx)
    data[data > 0.6] = np.nan

    r = compute(md.DataFrame(data, chunk_size=13), level=1, method="tree")
    pd.testing.assert_frame_equal(
        compute(data, level=1).sort_index(), r.execute().fetch().sort_index()
    )

    r = compute(md.DataFrame(data, chunk_size=13), level=1, skipna=False, method="tree")
    pd.testing.assert_frame_equal(
        compute(data, level=1, skipna=False).sort_index(),
        r.execute().fetch().sort_index(),
    )

    # test bool_only
    # bool_only not supported when level specified


def test_series_count(setup, check_ref_counts):
    array = np.random.rand(10)
    array[[2, 7, 9]] = np.nan
    data = pd.Series(array)
    series = md.Series(data)

    result = series.count().execute().fetch()
    expected = data.count()
    assert result == expected

    series2 = md.Series(data, chunk_size=1)

    result = series2.count().execute().fetch()
    expected = data.count()
    assert result == expected

    series2 = md.Series(data, chunk_size=3)

    result = series2.count().execute().fetch()
    expected = data.count()
    assert result == expected


def test_dataframe_count(setup, check_ref_counts):
    data = pd.DataFrame(
        {
            "Person": ["John", "Myla", "Lewis", "John", "Myla"],
            "Age": [24.0, np.nan, 21.0, 33, 26],
            "Single": [False, True, True, True, False],
        }
    )
    df = md.DataFrame(data)

    result = df.count().execute().fetch()
    expected = data.count()
    pd.testing.assert_series_equal(result, expected)

    result = df.count(axis="columns").execute().fetch()
    expected = data.count(axis="columns")
    pd.testing.assert_series_equal(result, expected)

    df2 = md.DataFrame(data, chunk_size=2)

    result = df2.count().execute().fetch()
    expected = data.count()
    pd.testing.assert_series_equal(result, expected)

    result = df2.count(axis="columns").execute().fetch()
    expected = data.count(axis="columns")
    pd.testing.assert_series_equal(result, expected)

    df3 = md.DataFrame(data, chunk_size=3)

    result = df3.count(numeric_only=True).execute().fetch()
    expected = data.count(numeric_only=True)
    pd.testing.assert_series_equal(result, expected)

    result = df3.count(axis="columns", numeric_only=True).execute().fetch()
    expected = data.count(axis="columns", numeric_only=True)
    pd.testing.assert_series_equal(result, expected)


def test_nunique(setup, check_ref_counts):
    data1 = pd.Series(np.random.randint(0, 5, size=(20,)))

    series = md.Series(data1)
    result = series.nunique().execute().fetch()
    expected = data1.nunique()
    assert result == expected

    series = md.Series(data1, chunk_size=6)
    result = series.nunique().execute().fetch()
    expected = data1.nunique()
    assert result == expected

    # test dropna
    data2 = data1.copy()
    data2[[2, 9, 18]] = np.nan

    series = md.Series(data2)
    result = series.nunique().execute().fetch()
    expected = data2.nunique()
    assert result == expected

    series = md.Series(data2, chunk_size=3)
    result = series.nunique(dropna=False).execute().fetch()
    expected = data2.nunique(dropna=False)
    assert result == expected

    # test dataframe
    data1 = pd.DataFrame(
        np.random.randint(0, 6, size=(20, 20)),
        columns=["c" + str(i) for i in range(20)],
    )
    df = md.DataFrame(data1)
    result = df.nunique().execute().fetch()
    expected = data1.nunique()
    pd.testing.assert_series_equal(result, expected)

    df = md.DataFrame(data1, chunk_size=6)
    result = df.nunique().execute().fetch()
    expected = data1.nunique()
    pd.testing.assert_series_equal(result, expected)

    df = md.DataFrame(data1)
    result = df.nunique(axis=1).execute().fetch()
    expected = data1.nunique(axis=1)
    pd.testing.assert_series_equal(result, expected)

    df = md.DataFrame(data1, chunk_size=3)
    result = df.nunique(axis=1).execute().fetch()
    expected = data1.nunique(axis=1)
    pd.testing.assert_series_equal(result, expected)

    # test dropna
    data2 = data1.copy()
    data2.iloc[[2, 9, 18], [2, 9, 18]] = np.nan

    df = md.DataFrame(data2)
    result = df.nunique().execute().fetch()
    expected = data2.nunique()
    pd.testing.assert_series_equal(result, expected)

    df = md.DataFrame(data2, chunk_size=3)
    result = df.nunique(dropna=False).execute().fetch()
    expected = data2.nunique(dropna=False)
    pd.testing.assert_series_equal(result, expected)

    df = md.DataFrame(data1, chunk_size=3)
    result = df.nunique(axis=1).execute().fetch()
    expected = data1.nunique(axis=1)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_use_arrow_dtype_n_unique(setup, check_ref_counts):
    with option_context({"dataframe.use_arrow_dtype": True, "combine_size": 2}):
        rs = np.random.RandomState(0)
        data1 = pd.DataFrame(
            {"a": rs.random(10), "b": [f"s{i}" for i in rs.randint(100, size=10)]}
        )
        data1["c"] = data1["b"].copy()
        data1["d"] = data1["b"].copy()
        data1["e"] = data1["b"].copy()

        df = md.DataFrame(data1, chunk_size=(3, 2))
        r = df.nunique(axis=0)
        result = r.execute().fetch()
        expected = data1.nunique(axis=0)
        pd.testing.assert_series_equal(result, expected)

        r = df.nunique(axis=1)
        result = r.execute().fetch()
        expected = data1.nunique(axis=1)
        pd.testing.assert_series_equal(result, expected)


def test_unique(setup, check_ref_counts):
    data1 = pd.Series(np.random.randint(0, 5, size=(20,)))

    series = md.Series(data1)
    result = series.unique().execute().fetch()
    expected = data1.unique()
    np.testing.assert_array_equal(result, expected)

    series = md.Series(data1, chunk_size=6)
    result = series.unique().execute().fetch()
    expected = data1.unique()
    np.testing.assert_array_equal(result, expected)

    data2 = pd.Series(
        [pd.Timestamp("20200101")] * 5
        + [pd.Timestamp("20200202")]
        + [pd.Timestamp("20020101")] * 9
    )
    series = md.Series(data2)
    result = series.unique().execute().fetch()
    expected = data2.unique()
    np.testing.assert_array_equal(result, expected)

    series = md.Series(data2, chunk_size=6)
    result = series.unique().execute().fetch()
    expected = data2.unique()
    np.testing.assert_array_equal(result, expected)


def test_index_reduction(setup, check_ref_counts):
    rs = np.random.RandomState(0)
    data = pd.Index(rs.randint(0, 5, (100,)))
    data2 = pd.Index(rs.randint(1, 6, (100,)))

    for method in ["min", "max", "all", "any"]:
        idx = md.Index(data)
        result = getattr(idx, method)().execute().fetch()
        assert result == getattr(data, method)()

        idx = md.Index(data, chunk_size=10)
        result = getattr(idx, method)().execute().fetch()
        assert result == getattr(data, method)()

        idx = md.Index(data2)
        result = getattr(idx, method)().execute().fetch()
        assert result == getattr(data2, method)()

        idx = md.Index(data2, chunk_size=10)
        result = getattr(idx, method)().execute().fetch()
        assert result == getattr(data2, method)()


cum_reduction_functions = ["cummax", "cummin", "cumprod", "cumsum"]


@pytest.mark.parametrize("func_name", cum_reduction_functions)
def test_series_cum_reduction(setup, check_ref_counts, func_name):
    def compute(data, **kwargs):
        return getattr(data, func_name)(**kwargs)

    data = pd.Series(np.random.rand(20), index=[str(i) for i in range(20)], name="a")
    r = compute(md.Series(data))
    pd.testing.assert_series_equal(compute(data), r.execute().fetch())

    r = compute(md.Series(data, chunk_size=6))
    pd.testing.assert_series_equal(compute(data), r.execute().fetch())

    r = compute(md.Series(data, chunk_size=3))
    pd.testing.assert_series_equal(compute(data), r.execute().fetch())

    r = compute(md.Series(data, chunk_size=4), axis="index")
    pd.testing.assert_series_equal(compute(data, axis="index"), r.execute().fetch())

    data = pd.Series(np.random.rand(20), name="a")
    data[0] = 0.1  # make sure not all elements are NAN
    data[data > 0.5] = np.nan
    r = compute(md.Series(data, chunk_size=3))
    pd.testing.assert_series_equal(compute(data), r.execute().fetch())

    r = compute(md.Series(data, chunk_size=3), skipna=False)
    pd.testing.assert_series_equal(compute(data, skipna=False), r.execute().fetch())


@pytest.mark.parametrize("func_name", cum_reduction_functions)
def test_dataframe_cum_reduction(setup, check_ref_counts, func_name):
    def compute(data, **kwargs):
        return getattr(data, func_name)(**kwargs)

    data = pd.DataFrame(np.random.rand(20, 10))
    r = compute(md.DataFrame(data))
    pd.testing.assert_frame_equal(compute(data), r.execute().fetch())

    r = compute(md.DataFrame(data, chunk_size=3))
    pd.testing.assert_frame_equal(compute(data), r.execute().fetch())

    r = compute(md.DataFrame(data, chunk_size=3), axis=1)
    pd.testing.assert_frame_equal(compute(data, axis=1), r.execute().fetch())

    # test null
    np_data = np.random.rand(20, 10)
    np_data[np_data > 0.6] = np.nan
    data = pd.DataFrame(np_data)

    r = compute(md.DataFrame(data, chunk_size=3))
    pd.testing.assert_frame_equal(compute(data), r.execute().fetch())

    r = compute(md.DataFrame(data, chunk_size=3), skipna=False)
    pd.testing.assert_frame_equal(compute(data, skipna=False), r.execute().fetch())

    r = compute(md.DataFrame(data, chunk_size=3), skipna=False)
    pd.testing.assert_frame_equal(compute(data, skipna=False), r.execute().fetch())

    # test numeric_only
    data = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.random.randint(-100, 100, size=(10,)),
        columns=[np.random.bytes(10) for _ in range(10)],
    )
    r = compute(md.DataFrame(data, chunk_size=2))
    pd.testing.assert_frame_equal(compute(data), r.execute().fetch())

    r = compute(md.DataFrame(data, chunk_size=3), axis="columns")
    pd.testing.assert_frame_equal(compute(data, axis="columns"), r.execute().fetch())


def test_dataframe_aggregate(setup, check_ref_counts):
    all_aggs = [
        "sum",
        "prod",
        "min",
        "max",
        "count",
        "size",
        "mean",
        "var",
        "std",
        "sem",
        "skew",
        "kurt",
    ]
    data = pd.DataFrame(np.random.rand(20, 20))

    df = md.DataFrame(data)
    result = df.agg(all_aggs)
    pd.testing.assert_frame_equal(result.execute().fetch(), data.agg(all_aggs))

    result = df.agg("size")
    if _agg_size_as_series:
        pd.testing.assert_series_equal(result.execute().fetch(), data.agg("size"))
    else:
        assert result.execute().fetch() == data.agg("size")

    for func in (a for a in all_aggs if a != "size"):
        result = df.agg(func)
        pd.testing.assert_series_equal(result.execute().fetch(), data.agg(func))

        result = df.agg(func, axis=1)
        pd.testing.assert_series_equal(result.execute().fetch(), data.agg(func, axis=1))

    df = md.DataFrame(data, chunk_size=3)

    # will redirect to transform
    result = df.agg(["cumsum", "cummax"])
    pd.testing.assert_frame_equal(
        result.execute().fetch(), data.agg(["cumsum", "cummax"])
    )

    result = df.agg("size")
    if _agg_size_as_series:
        pd.testing.assert_series_equal(result.execute().fetch(), data.agg("size"))
    else:
        assert result.execute().fetch() == data.agg("size")

    for func in (a for a in all_aggs if a != "size"):
        result = df.agg(func)
        pd.testing.assert_series_equal(result.execute().fetch(), data.agg(func))

        result = df.agg(func, axis=1)
        pd.testing.assert_series_equal(result.execute().fetch(), data.agg(func, axis=1))

    result = df.agg(["sum"])
    pd.testing.assert_frame_equal(result.execute().fetch(), data.agg(["sum"]))

    result = df.agg(all_aggs)
    pd.testing.assert_frame_equal(result.execute().fetch(), data.agg(all_aggs))

    result = df.agg(all_aggs, axis=1)
    pd.testing.assert_frame_equal(result.execute().fetch(), data.agg(all_aggs, axis=1))

    result = df.agg({0: ["sum", "min", "var"], 9: ["mean", "var", "std"]})
    pd.testing.assert_frame_equal(
        result.execute().fetch(),
        data.agg({0: ["sum", "min", "var"], 9: ["mean", "var", "std"]}),
    )

    if _support_kw_agg:
        agg_kw = dict(
            sum_0=NamedAgg(0, "sum"),
            min_0=NamedAgg(0, "min"),
            mean_9=NamedAgg(9, "mean"),
        )
        result = df.agg(**agg_kw)
        pd.testing.assert_frame_equal(result.execute().fetch(), data.agg(**agg_kw))


def test_series_aggregate(setup, check_ref_counts):
    all_aggs = [
        "sum",
        "prod",
        "min",
        "max",
        "count",
        "size",
        "mean",
        "var",
        "std",
        "sem",
        "skew",
        "kurt",
    ]
    data = pd.Series(np.random.rand(20), index=[str(i) for i in range(20)], name="a")
    series = md.Series(data)

    result = series.agg(all_aggs)
    pd.testing.assert_series_equal(result.execute().fetch(), data.agg(all_aggs))

    for func in all_aggs:
        result = series.agg(func)
        assert pytest.approx(result.execute().fetch()) == data.agg(func)

    series = md.Series(data, chunk_size=3)

    for func in all_aggs:
        result = series.agg(func)
        assert pytest.approx(result.execute().fetch()) == data.agg(func)

    result = series.agg(all_aggs)
    pd.testing.assert_series_equal(result.execute().fetch(), data.agg(all_aggs))

    result = series.agg({"col_sum": "sum", "col_count": "count"})
    pd.testing.assert_series_equal(
        result.execute().fetch(), data.agg({"col_sum": "sum", "col_count": "count"})
    )

    if _support_kw_agg:
        result = series.agg(col_var="var", col_skew="skew")
        pd.testing.assert_series_equal(
            result.execute().fetch(), data.agg(col_var="var", col_skew="skew")
        )


def test_aggregate_str_cat(setup, check_ref_counts):
    agg_fun = lambda x: x.str.cat(sep="_", na_rep="NA")

    rs = np.random.RandomState(0)
    raw_df = pd.DataFrame(
        {
            "a": rs.choice(["A", "B", "C"], size=(100,)),
            "b": rs.choice([None, "alfa", "bravo", "charlie"], size=(100,)),
        }
    )

    mdf = md.DataFrame(raw_df, chunk_size=13)

    r = mdf.agg(agg_fun)
    pd.testing.assert_series_equal(r.execute().fetch(), raw_df.agg(agg_fun))

    raw_series = pd.Series(rs.choice([None, "alfa", "bravo", "charlie"], size=(100,)))

    ms = md.Series(raw_series, chunk_size=13)

    r = ms.agg(agg_fun)
    assert r.execute().fetch() == raw_series.agg(agg_fun)


class MockReduction1(CustomReduction):
    def agg(self, v1):
        return v1.sum()


class MockReduction2(CustomReduction):
    def pre(self, value):
        return value + 1, value**2

    def agg(self, v1, v2):
        return v1.sum(), v2.prod()

    def post(self, v1, v2):
        return v1 + v2


def test_custom_dataframe_aggregate(setup, check_ref_counts):
    rs = np.random.RandomState(0)
    data = pd.DataFrame(rs.rand(30, 20))

    df = md.DataFrame(data)
    result = df.agg(MockReduction1())
    pd.testing.assert_series_equal(result.execute().fetch(), data.agg(MockReduction1()))

    result = df.agg(MockReduction2())
    pd.testing.assert_series_equal(result.execute().fetch(), data.agg(MockReduction2()))

    df = md.DataFrame(data, chunk_size=5)
    result = df.agg(MockReduction2())
    pd.testing.assert_series_equal(result.execute().fetch(), data.agg(MockReduction2()))

    result = df.agg(MockReduction2())
    pd.testing.assert_series_equal(result.execute().fetch(), data.agg(MockReduction2()))


def test_custom_series_aggregate(setup, check_ref_counts):
    rs = np.random.RandomState(0)
    data = pd.Series(rs.rand(20))

    s = md.Series(data)
    result = s.agg(MockReduction1())
    assert result.execute().fetch() == data.agg(MockReduction1())

    result = s.agg(MockReduction2())
    assert result.execute().fetch() == data.agg(MockReduction2())

    s = md.Series(data, chunk_size=5)
    result = s.agg(MockReduction2())
    assert pytest.approx(result.execute().fetch()) == data.agg(MockReduction2())

    result = s.agg(MockReduction2())
    assert pytest.approx(result.execute().fetch()) == data.agg(MockReduction2())
