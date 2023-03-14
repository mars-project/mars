# Copyright 2022 XProbe Inc.
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

from .... import dataframe as md
from ....dataframe.core import DataFrame, DATAFRAME_OR_SERIES_TYPE
from ....dataframe.fetch.core import DataFrameFetch


def test_dataframe_apply_execution(setup):
    df = pd.DataFrame({"col": [1, 2, 3, 4]})
    mdf = md.DataFrame(df)

    apply_func = lambda x: 20 if x[0] else 10
    with pytest.raises(TypeError):
        mdf.apply(apply_func)

    res = mdf.apply(apply_func, output_type="df_or_series", axis=1).execute()
    assert res.data_type == "series"
    assert "dtype" in res.data_params
    assert not ("dtypes" in res.data_params)
    assert res.data_params["shape"] == (4,)
    pd.testing.assert_series_equal(res.fetch(), df.apply(apply_func, axis=1))

    res = mdf.apply(apply_func, output_type="df_or_series", axis=0).execute()
    assert res.data_type == "series"
    assert "dtype" in res.data_params
    assert not ("dtypes" in res.data_params)
    assert res.data_params["shape"] == (1,)
    pd.testing.assert_series_equal(res.fetch(), df.apply(apply_func, axis=0))

    apply_func = lambda x: x + 1
    res = mdf.apply(apply_func, output_type="df_or_series", axis=1).execute()
    assert res.data_type == "dataframe"
    assert "dtypes" in res.data_params
    assert not ("dtype" in res.data_params)
    assert res.data_params["shape"] == (4, 1)
    pd.testing.assert_frame_equal(res.fetch(), df.apply(apply_func, axis=1))

    res = mdf.apply(apply_func, output_type="df_or_series", axis=0).execute()
    assert res.data_type == "dataframe"
    assert "dtypes" in res.data_params
    assert not ("dtype" in res.data_params)
    assert res.data_params["shape"] == (4, 1)
    pd.testing.assert_frame_equal(res.fetch(), df.apply(apply_func, axis=0))

    apply_func = lambda x: sum(x)
    res = mdf.apply(apply_func, output_type="df_or_series", axis=1).execute()
    assert res.data_type == "series"
    assert "dtype" in res.data_params
    assert not ("dtypes" in res.data_params)
    assert res.data_params["shape"] == (4,)
    pd.testing.assert_series_equal(res.fetch(), df.apply(apply_func, axis=1))

    res = mdf.apply(apply_func, output_type="df_or_series", axis=0).execute()
    assert res.data_type == "series"
    assert "dtype" in res.data_params
    assert not ("dtypes" in res.data_params)
    assert res.data_params["shape"] == (1,)
    pd.testing.assert_series_equal(res.fetch(), df.apply(apply_func, axis=0))

    df = pd.DataFrame({"c1": [1, 2, 3, 4], "c2": [5, 6, 7, 8]})
    mdf = md.DataFrame(df)
    apply_func = lambda x: sum(x) / len(x)

    res = mdf.apply(apply_func, output_type="df_or_series", axis=1).execute()
    assert res.data_type == "series"
    assert "dtype" in res.data_params
    assert res.data_params["dtype"] == "float64"
    assert not ("dtypes" in res.data_params)
    assert res.data_params["shape"] == (4,)
    pd.testing.assert_series_equal(res.fetch(), df.apply(apply_func, axis=1))

    res = mdf.apply(apply_func, output_type="df_or_series", axis=0).execute()
    assert res.data_type == "series"
    assert "dtype" in res.data_params
    assert res.data_params["dtype"] == "float64"
    assert not ("dtypes" in res.data_params)
    assert res.data_params["shape"] == (2,)
    pd.testing.assert_series_equal(res.fetch(), df.apply(apply_func, axis=0))

    apply_func = lambda x: pd.Series([1, 2])
    res = mdf.apply(apply_func, output_type="df_or_series", axis=0).execute()
    assert res.data_type == "dataframe"
    assert "dtypes" in res.data_params
    assert res.data_params["shape"] == (2, 2)
    pd.testing.assert_frame_equal(res.fetch(), df.apply(apply_func, axis=0))

    res = mdf.apply(apply_func, output_type="df_or_series", axis=1).execute()
    assert res.data_type == "dataframe"
    assert "dtypes" in res.data_params
    assert res.data_params["shape"] == (4, 2)
    pd.testing.assert_frame_equal(res.fetch(), df.apply(apply_func, axis=1))

    apply_func = lambda x: [1, 2]
    res = mdf.apply(apply_func, output_type="df_or_series", axis=0).execute()
    assert res.data_type == "dataframe"
    assert res.data_params["shape"] == (2, 2)
    pd.testing.assert_frame_equal(res.fetch(), df.apply(apply_func, axis=0))

    res = mdf.apply(apply_func, output_type="df_or_series", axis=1).execute()
    assert res.data_type == "series"
    assert res.data_params["shape"] == (4,)
    assert res.data_params["dtype"] == "object"
    pd.testing.assert_series_equal(res.fetch(), df.apply(apply_func, axis=1))

    apply_func = lambda x: pd.Series([1, 2, 3.0], index=["c1", "c2", "c3"])
    res = mdf.apply(apply_func, output_type="df_or_series", axis=0).execute()
    assert res.data_type == "dataframe"
    assert res.data_params["shape"] == (3, 2)
    pd.testing.assert_frame_equal(res.fetch(), df.apply(apply_func, axis=0))

    res = mdf.apply(apply_func, output_type="df_or_series", axis=1).execute()
    assert res.data_type == "dataframe"
    assert res.data_params["shape"] == (4, 3)
    pd.testing.assert_frame_equal(res.fetch(), df.apply(apply_func, axis=1))

    apply_func = lambda x: [1, 2, 3]
    res = mdf.apply(
        apply_func, output_type="df_or_series", axis=1, result_type="expand"
    ).execute()
    expected = df.apply(apply_func, axis=1, result_type="expand")
    pd.testing.assert_frame_equal(res.fetch(), expected)

    res = mdf.apply(
        apply_func, output_type="df_or_series", axis=1, result_type="reduce"
    ).execute()
    expected = df.apply(apply_func, axis=1, result_type="reduce")
    pd.testing.assert_series_equal(res.fetch(), expected)

    apply_func = lambda x: [1, 2]
    res = mdf.apply(
        apply_func, output_type="df_or_series", axis=1, result_type="broadcast"
    ).execute()
    expected = df.apply(apply_func, axis=1, result_type="broadcast")
    pd.testing.assert_frame_equal(res.fetch(), expected)


def test_apply_with_skip_infer(setup):
    df = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": list("abcd")})
    mdf = md.DataFrame(df, chunk_size=2)

    def apply_func(series):
        if series[1] not in "abcd":
            # make it fail when inferring
            raise TypeError
        else:
            return 1

    with pytest.raises(TypeError):
        mdf.apply(apply_func, axis=1)

    res = mdf.apply(apply_func, axis=1, skip_infer=True).execute()
    assert isinstance(res, DATAFRAME_OR_SERIES_TYPE)
    pd.testing.assert_series_equal(res.fetch(), pd.Series([1] * 4))

    s = pd.Series([1, 2, 3, 4])
    ms = md.Series(s, chunk_size=2)

    apply_func = lambda x: pd.Series([1, 2])
    res = ms.apply(apply_func, skip_infer=True).execute()
    assert isinstance(res, DATAFRAME_OR_SERIES_TYPE)
    pd.testing.assert_frame_equal(res.fetch(), pd.DataFrame([[1, 2]] * 4))


def test_series_apply_execution(setup):
    s = pd.Series([1, 2, 3, 4])
    ms = md.Series(s)

    apply_func = lambda x: x + 1
    res = ms.apply(apply_func, output_type="df_or_series").execute()
    assert res.data_type == "series"
    assert res.data_params["shape"] == (4,)
    assert res.data_params["dtype"] == "int64"
    pd.testing.assert_series_equal(res.fetch(), s.apply(apply_func))

    apply_func = lambda x: [1, 2]
    res = ms.apply(apply_func, output_type="df_or_series").execute()
    assert res.data_type == "series"
    assert res.data_params["shape"] == (4,)
    assert res.data_params["dtype"] == "object"
    pd.testing.assert_series_equal(res.fetch(), s.apply(apply_func))

    apply_func = lambda x: pd.Series([1, 2, 3])
    res = ms.apply(apply_func, output_type="df_or_series").execute()
    assert res.data_type == "dataframe"
    assert "dtypes" in res.data_params
    assert res.data_params["shape"] == (4, 3)
    pd.testing.assert_frame_equal(res.fetch(), s.apply(apply_func))

    def subtract_custom_value(x, custom_value):
        return x - custom_value

    apply_func = subtract_custom_value
    res = ms.apply(
        apply_func, args=(5,), convert_dtype=False, output_type="df_or_series"
    ).execute()
    assert res.data_params["dtype"] == "object"
    pd.testing.assert_series_equal(
        res.fetch(), s.apply(apply_func, args=(5,), convert_dtype=False)
    )

    res = ms.apply(
        apply_func, args=(5,), convert_dtype=True, output_type="df_or_series"
    ).execute()
    assert res.dtype == "int64"
    assert res.shape == (4,)
    with pytest.raises(AttributeError):
        _ = res.dtypes
    pd.testing.assert_series_equal(
        res.fetch(), s.apply(apply_func, args=(5,), convert_dtype=True)
    )


def test_apply_execution_with_multi_chunks(setup):
    df = pd.DataFrame({"c1": [1, 2, 3, 4], "c2": [5, 6, 7, 8]})
    mdf = md.DataFrame(df, chunk_size=5)
    apply_func = np.sqrt

    res = mdf.apply(apply_func, output_type="df_or_series", axis=0).execute()
    assert res.data_type == "dataframe"
    assert "dtypes" in res.data_params
    assert res.data_params["dtypes"]["c1"] == np.dtype("float")
    assert not ("dtype" in res.data_params)
    assert res.data_params["shape"] == (4, 2)
    pd.testing.assert_frame_equal(res.fetch(), df.apply(apply_func, axis=0))

    res = mdf.apply(apply_func, output_type="df_or_series", axis=1).execute()
    assert res.data_type == "dataframe"
    assert "dtypes" in res.data_params
    assert res.data_params["dtypes"]["c2"] == np.dtype("float")
    assert not ("dtype" in res.data_params)
    assert res.data_params["shape"] == (4, 2)
    pd.testing.assert_frame_equal(res.fetch(), df.apply(apply_func, axis=1))

    s = pd.Series([1, 2, 3, 4])
    ms = md.Series(s, chunk_size=4)

    res = ms.apply(apply_func, output_type="df_or_series").execute()
    assert res.data_type == "series"
    assert "dtype" in res.data_params
    assert res.data_params["dtype"] == "float64"
    pd.testing.assert_series_equal(res.fetch(), s.apply(apply_func))


def test_apply_ensure_data(setup):
    df = pd.DataFrame({"c1": [1, 2, 3, 4], "c2": [5, 6, 7, 8]})
    mdf = md.DataFrame(df, chunk_size=3)
    apply_func = np.sqrt

    r = mdf.apply(apply_func, output_type="df_or_series")
    res = r.ensure_data()
    assert isinstance(res, DataFrame)
    assert isinstance(res.op, DataFrameFetch)
    pd.testing.assert_frame_equal(res.execute().fetch(), df.apply(apply_func))
    pd.testing.assert_frame_equal((res + 1).execute().fetch(), df.apply(apply_func) + 1)
    pd.testing.assert_frame_equal((res * 3).execute().fetch(), df.apply(apply_func) * 3)

    r = res.groupby("c1").max()
    expected = df.apply(apply_func).groupby("c1").max()
    pd.testing.assert_frame_equal(r.execute().fetch(), expected)

    apply_func = np.mean
    res = mdf.apply(apply_func, output_type="df_or_series", axis=1).ensure_data()
    expected = df.apply(apply_func, axis=1)
    pd.testing.assert_series_equal(res.execute().fetch(), expected)

    res = res.to_frame(name="foo").groupby("foo")[["foo"]].max().execute()
    expected = expected.to_frame(name="foo").groupby("foo")[["foo"]].max()
    pd.testing.assert_frame_equal(res.fetch(), expected)
