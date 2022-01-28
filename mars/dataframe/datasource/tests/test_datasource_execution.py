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

import os
import tempfile
import time
from collections import OrderedDict
from datetime import datetime
from string import printable

import numpy as np
import pandas as pd
import pytest

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None
try:
    import fastparquet
except ImportError:  # pragma: no cover
    fastparquet = None
try:
    import sqlalchemy
except ImportError:  # pragma: no cover
    sqlalchemy = None


from .... import tensor as mt
from .... import dataframe as md
from ....config import option_context
from ....tests.core import require_cudf, require_ray
from ....utils import arrow_array_to_objects, lazy_import
from ..dataframe import from_pandas as from_pandas_df
from ..series import from_pandas as from_pandas_series
from ..index import from_pandas as from_pandas_index, from_tileable
from ..from_tensor import dataframe_from_tensor, dataframe_from_1d_tileables
from ..from_records import from_records


ray = lazy_import("ray")


def test_from_pandas_dataframe_execution(setup):
    # test empty DataFrame
    pdf = pd.DataFrame()
    df = from_pandas_df(pdf)

    result = df.execute().fetch()
    pd.testing.assert_frame_equal(pdf, result)

    pdf = pd.DataFrame(columns=list("ab"))
    df = from_pandas_df(pdf)

    result = df.execute().fetch()
    pd.testing.assert_frame_equal(pdf, result)

    pdf = pd.DataFrame(
        np.random.rand(20, 30), index=[np.arange(20), np.arange(20, 0, -1)]
    )
    df = from_pandas_df(pdf, chunk_size=(13, 21))

    result = df.execute().fetch()
    pd.testing.assert_frame_equal(pdf, result)


def test_from_pandas_series_execution(setup):
    # test empty Series
    ps = pd.Series(name="a")
    series = from_pandas_series(ps, chunk_size=13)

    result = series.execute().fetch()
    pd.testing.assert_series_equal(ps, result)

    series = from_pandas_series(ps)

    result = series.execute().fetch()
    pd.testing.assert_series_equal(ps, result)

    ps = pd.Series(
        np.random.rand(20), index=[np.arange(20), np.arange(20, 0, -1)], name="a"
    )
    series = from_pandas_series(ps, chunk_size=13)

    result = series.execute().fetch()
    pd.testing.assert_series_equal(ps, result)


def test_from_pandas_index_execution(setup):
    pd_index = pd.timedelta_range("1 days", periods=10)
    index = from_pandas_index(pd_index, chunk_size=7)

    result = index.execute().fetch()
    pd.testing.assert_index_equal(pd_index, result)


def test_index_execution(setup):
    rs = np.random.RandomState(0)
    pdf = pd.DataFrame(
        rs.rand(20, 10),
        index=np.arange(20, 0, -1),
        columns=["a" + str(i) for i in range(10)],
    )
    df = from_pandas_df(pdf, chunk_size=13)

    # test df.index
    result = df.index.execute().fetch()
    pd.testing.assert_index_equal(result, pdf.index)

    result = df.columns.execute().fetch()
    pd.testing.assert_index_equal(result, pdf.columns)

    # df has unknown chunk shape on axis 0
    df = df[df.a1 < 0.5]

    # test df.index
    result = df.index.execute().fetch()
    pd.testing.assert_index_equal(result, pdf[pdf.a1 < 0.5].index)

    s = pd.Series(pdf["a1"], index=pd.RangeIndex(20))
    series = from_pandas_series(s, chunk_size=13)

    # test series.index which has value
    result = series.index.execute().fetch()
    pd.testing.assert_index_equal(result, s.index)

    s = pdf["a2"]
    series = from_pandas_series(s, chunk_size=13)

    # test series.index
    result = series.index.execute().fetch()
    pd.testing.assert_index_equal(result, s.index)

    # test tensor
    raw = rs.random(20)
    t = mt.tensor(raw, chunk_size=13)

    result = from_tileable(t).execute().fetch()
    pd.testing.assert_index_equal(result, pd.Index(raw))


def test_initializer_execution(setup):
    arr = np.random.rand(20, 30)

    pdf = pd.DataFrame(arr, index=[np.arange(20), np.arange(20, 0, -1)])
    df = md.DataFrame(pdf, chunk_size=(15, 10))
    result = df.execute().fetch()
    pd.testing.assert_frame_equal(pdf, result)

    df = md.DataFrame(arr, index=md.date_range("2020-1-1", periods=20))
    result = df.execute().fetch()
    pd.testing.assert_frame_equal(
        result, pd.DataFrame(arr, index=pd.date_range("2020-1-1", periods=20))
    )

    df = md.DataFrame(
        {"prices": [100, 101, np.nan, 100, 89, 88]},
        index=md.date_range("1/1/2010", periods=6, freq="D"),
    )
    result = df.execute().fetch()
    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            {"prices": [100, 101, np.nan, 100, 89, 88]},
            index=pd.date_range("1/1/2010", periods=6, freq="D"),
        ),
    )

    s = np.random.rand(20)

    ps = pd.Series(s, index=[np.arange(20), np.arange(20, 0, -1)], name="a")
    series = md.Series(ps, chunk_size=7)
    result = series.execute().fetch()
    pd.testing.assert_series_equal(ps, result)

    series = md.Series(s, index=md.date_range("2020-1-1", periods=20))
    result = series.execute().fetch()
    pd.testing.assert_series_equal(
        result, pd.Series(s, index=pd.date_range("2020-1-1", periods=20))
    )

    pi = pd.IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)])
    index = md.Index(md.Index(pi))
    result = index.execute().fetch()
    pd.testing.assert_index_equal(pi, result)


def test_index_only(setup):
    df = md.DataFrame(index=[1, 2, 3])
    pd.testing.assert_frame_equal(df.execute().fetch(), pd.DataFrame(index=[1, 2, 3]))

    s = md.Series(index=[1, 2, 3])
    pd.testing.assert_series_equal(s.execute().fetch(), pd.Series(index=[1, 2, 3]))

    df = md.DataFrame(index=md.Index([1, 2, 3]))
    pd.testing.assert_frame_equal(df.execute().fetch(), pd.DataFrame(index=[1, 2, 3]))

    s = md.Series(index=md.Index([1, 2, 3]), dtype=object)
    pd.testing.assert_series_equal(
        s.execute().fetch(), pd.Series(index=[1, 2, 3], dtype=object)
    )


def test_series_from_tensor(setup):
    data = np.random.rand(10)
    series = md.Series(mt.tensor(data), name="a")
    pd.testing.assert_series_equal(series.execute().fetch(), pd.Series(data, name="a"))

    series = md.Series(mt.tensor(data, chunk_size=3))
    pd.testing.assert_series_equal(series.execute().fetch(), pd.Series(data))

    series = md.Series(mt.ones((10,), chunk_size=4))
    pd.testing.assert_series_equal(
        series.execute().fetch(),
        pd.Series(np.ones(10)),
    )

    index_data = np.random.rand(10)
    series = md.Series(
        mt.tensor(data, chunk_size=3),
        name="a",
        index=mt.tensor(index_data, chunk_size=4),
    )
    pd.testing.assert_series_equal(
        series.execute().fetch(), pd.Series(data, name="a", index=index_data)
    )

    series = md.Series(
        mt.tensor(data, chunk_size=3),
        name="a",
        index=md.date_range("2020-1-1", periods=10),
    )
    pd.testing.assert_series_equal(
        series.execute().fetch(),
        pd.Series(data, name="a", index=pd.date_range("2020-1-1", periods=10)),
    )


def test_from_tensor_execution(setup):
    tensor = mt.random.rand(10, 10, chunk_size=5)
    df = dataframe_from_tensor(tensor)
    tensor_res = tensor.execute().fetch()
    pdf_expected = pd.DataFrame(tensor_res)
    df_result = df.execute().fetch()
    pd.testing.assert_index_equal(df_result.index, pd.RangeIndex(0, 10))
    pd.testing.assert_index_equal(df_result.columns, pd.RangeIndex(0, 10))
    pd.testing.assert_frame_equal(df_result, pdf_expected)

    # test from tensor with unknown shape
    tensor2 = tensor[tensor[:, 0] < 0.9]
    df = dataframe_from_tensor(tensor2)
    df_result = df.execute().fetch()
    tensor_res = tensor2.execute().fetch()
    pdf_expected = pd.DataFrame(tensor_res)
    pd.testing.assert_frame_equal(df_result.reset_index(drop=True), pdf_expected)

    # test converted with specified index_value and columns
    tensor2 = mt.random.rand(2, 2, chunk_size=1)
    df2 = dataframe_from_tensor(
        tensor2, index=pd.Index(["a", "b"]), columns=pd.Index([3, 4])
    )
    df_result = df2.execute().fetch()
    pd.testing.assert_index_equal(df_result.index, pd.Index(["a", "b"]))
    pd.testing.assert_index_equal(df_result.columns, pd.Index([3, 4]))

    # test converted from 1-d tensor
    tensor3 = mt.array([1, 2, 3])
    df3 = dataframe_from_tensor(tensor3)
    result3 = df3.execute().fetch()
    pdf_expected = pd.DataFrame(np.array([1, 2, 3]))
    pd.testing.assert_frame_equal(pdf_expected, result3)

    # test converted from identical chunks
    tensor4 = mt.ones((10, 10), chunk_size=3)
    df4 = dataframe_from_tensor(tensor4)
    result4 = df4.execute().fetch()
    pdf_expected = pd.DataFrame(tensor4.execute().fetch())
    pd.testing.assert_frame_equal(pdf_expected, result4)

    # from tensor with given index
    tensor5 = mt.ones((10, 10), chunk_size=3)
    df5 = dataframe_from_tensor(tensor5, index=np.arange(0, 20, 2))
    result5 = df5.execute().fetch()
    pdf_expected = pd.DataFrame(tensor5.execute().fetch(), index=np.arange(0, 20, 2))
    pd.testing.assert_frame_equal(pdf_expected, result5)

    # from tensor with given index that is a tensor
    raw7 = np.random.rand(10, 10)
    tensor7 = mt.tensor(raw7, chunk_size=3)
    index_raw7 = np.random.rand(10)
    index7 = mt.tensor(index_raw7, chunk_size=4)
    df7 = dataframe_from_tensor(tensor7, index=index7)
    result7 = df7.execute().fetch()
    pdf_expected = pd.DataFrame(raw7, index=index_raw7)
    pd.testing.assert_frame_equal(pdf_expected, result7)

    # from tensor with given index is a md.Index
    raw10 = np.random.rand(10, 10)
    tensor10 = mt.tensor(raw10, chunk_size=3)
    index10 = md.date_range("2020-1-1", periods=10, chunk_size=3)
    df10 = dataframe_from_tensor(tensor10, index=index10)
    result10 = df10.execute().fetch()
    pdf_expected = pd.DataFrame(raw10, index=pd.date_range("2020-1-1", periods=10))
    pd.testing.assert_frame_equal(pdf_expected, result10)

    # from tensor with given columns
    tensor6 = mt.ones((10, 10), chunk_size=3)
    df6 = dataframe_from_tensor(tensor6, columns=list("abcdefghij"))
    result6 = df6.execute().fetch()
    pdf_expected = pd.DataFrame(tensor6.execute().fetch(), columns=list("abcdefghij"))
    pd.testing.assert_frame_equal(pdf_expected, result6)

    # from 1d tensors
    raws8 = [
        ("a", np.random.rand(8)),
        ("b", np.random.randint(10, size=8)),
        ("c", ["".join(np.random.choice(list(printable), size=6)) for _ in range(8)]),
    ]
    tensors8 = OrderedDict((r[0], mt.tensor(r[1], chunk_size=3)) for r in raws8)
    raws8.append(("d", 1))
    raws8.append(("e", pd.date_range("2020-1-1", periods=8)))
    tensors8["d"] = 1
    tensors8["e"] = raws8[-1][1]
    df8 = dataframe_from_1d_tileables(tensors8, columns=[r[0] for r in raws8])
    result = df8.execute().fetch()
    pdf_expected = pd.DataFrame(OrderedDict(raws8))
    pd.testing.assert_frame_equal(result, pdf_expected)

    # from 1d tensors and specify index with a tensor
    index_raw9 = np.random.rand(8)
    index9 = mt.tensor(index_raw9, chunk_size=4)
    df9 = dataframe_from_1d_tileables(
        tensors8, columns=[r[0] for r in raws8], index=index9
    )
    result = df9.execute().fetch()
    pdf_expected = pd.DataFrame(OrderedDict(raws8), index=index_raw9)
    pd.testing.assert_frame_equal(result, pdf_expected)

    # from 1d tensors and specify index
    df11 = dataframe_from_1d_tileables(
        tensors8,
        columns=[r[0] for r in raws8],
        index=md.date_range("2020-1-1", periods=8),
    )
    result = df11.execute().fetch()
    pdf_expected = pd.DataFrame(
        OrderedDict(raws8), index=pd.date_range("2020-1-1", periods=8)
    )
    pd.testing.assert_frame_equal(result, pdf_expected)


def test_from_records_execution(setup):
    dtype = np.dtype([("x", "int"), ("y", "double"), ("z", "<U16")])

    ndarr = np.ones((10,), dtype=dtype)
    pdf_expected = pd.DataFrame.from_records(ndarr, index=pd.RangeIndex(10))

    # from structured array of mars
    tensor = mt.ones((10,), dtype=dtype, chunk_size=3)
    df1 = from_records(tensor)
    df1_result = df1.execute().fetch()
    pd.testing.assert_frame_equal(df1_result, pdf_expected)

    # from structured array of numpy
    df2 = from_records(ndarr)
    df2_result = df2.execute().fetch()
    pd.testing.assert_frame_equal(df2_result, pdf_expected)


def test_read_csv_execution(setup):
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")

        df = pd.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
            columns=["a", "b", "c"],
        )
        df.to_csv(file_path)

        pdf = pd.read_csv(file_path, index_col=0)
        r = md.read_csv(file_path, index_col=0)
        mdf = r.execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf)
        # size_res = self.executor.execute_dataframe(r, mock=True)
        # assert sum(s[0] for s in size_res) == os.stat(file_path).st_size

        mdf2 = md.read_csv(file_path, index_col=0, chunk_bytes=10).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf2)

        mdf = md.read_csv(file_path, index_col=0, nrows=1).execute().fetch()
        pd.testing.assert_frame_equal(df[:1], mdf)

    # test names and usecols
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")
        df = pd.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
            columns=["a", "b", "c"],
        )
        df.to_csv(file_path, index=False)

        mdf = md.read_csv(file_path, usecols=["c", "b"]).execute().fetch()
        pd.testing.assert_frame_equal(pd.read_csv(file_path, usecols=["c", "b"]), mdf)

        mdf = (
            md.read_csv(file_path, names=["a", "b", "c"], usecols=["c", "b"])
            .execute()
            .fetch()
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(file_path, names=["a", "b", "c"], usecols=["c", "b"]), mdf
        )

        mdf = (
            md.read_csv(file_path, names=["a", "b", "c"], usecols=["a", "c"])
            .execute()
            .fetch()
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(file_path, names=["a", "b", "c"], usecols=["a", "c"]), mdf
        )

        mdf = md.read_csv(file_path, usecols=["a", "c"]).execute().fetch()
        pd.testing.assert_frame_equal(pd.read_csv(file_path, usecols=["a", "c"]), mdf)

    # test sep
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")

        df = pd.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
        )
        df.to_csv(file_path, sep=";")

        pdf = pd.read_csv(file_path, sep=";", index_col=0)
        mdf = md.read_csv(file_path, sep=";", index_col=0).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf)

        mdf2 = (
            md.read_csv(file_path, sep=";", index_col=0, chunk_bytes=10)
            .execute()
            .fetch()
        )
        pd.testing.assert_frame_equal(pdf, mdf2)

    # test missing value
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")

        df = pd.DataFrame(
            {
                "c1": [np.nan, "a", "b", "c"],
                "c2": [1, 2, 3, np.nan],
                "c3": [np.nan, np.nan, 3.4, 2.2],
            }
        )
        df.to_csv(file_path)

        pdf = pd.read_csv(file_path, index_col=0)
        mdf = md.read_csv(file_path, index_col=0).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf)

        mdf2 = md.read_csv(file_path, index_col=0, chunk_bytes=12).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf2)

    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")

        index = pd.date_range(start="1/1/2018", periods=100)
        df = pd.DataFrame(
            {
                "col1": np.random.rand(100),
                "col2": np.random.choice(["a", "b", "c"], (100,)),
                "col3": np.arange(100),
            },
            index=index,
        )
        df.to_csv(file_path)

        pdf = pd.read_csv(file_path, index_col=0)
        mdf = md.read_csv(file_path, index_col=0).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf)

        mdf2 = md.read_csv(file_path, index_col=0, chunk_bytes=100).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf2)

    # test nan
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")

        df = pd.DataFrame(
            {
                "col1": np.random.rand(100),
                "col2": np.random.choice(["a", "b", "c"], (100,)),
                "col3": np.arange(100),
            }
        )
        df.iloc[20:, :] = pd.NA
        df.to_csv(file_path)

        pdf = pd.read_csv(file_path, index_col=0)
        mdf = md.read_csv(file_path, index_col=0, head_lines=10, chunk_bytes=200)
        result = mdf.execute().fetch()
        pd.testing.assert_frame_equal(pdf, result)

        # dtypes is inferred as expected
        pd.testing.assert_series_equal(
            mdf.dtypes, pd.Series(["float64", "object", "int64"], index=df.columns)
        )

    # test compression
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.gzip")

        index = pd.date_range(start="1/1/2018", periods=100)
        df = pd.DataFrame(
            {
                "col1": np.random.rand(100),
                "col2": np.random.choice(["a", "b", "c"], (100,)),
                "col3": np.arange(100),
            },
            index=index,
        )
        df.to_csv(file_path, compression="gzip")

        pdf = pd.read_csv(file_path, compression="gzip", index_col=0)
        mdf = md.read_csv(file_path, compression="gzip", index_col=0).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf)

        mdf2 = (
            md.read_csv(file_path, compression="gzip", index_col=0, chunk_bytes="1k")
            .execute()
            .fetch()
        )
        pd.testing.assert_frame_equal(pdf, mdf2)

    # test multiple files
    for merge_small_file_option in [{"n_sample_file": 1}, None]:
        with tempfile.TemporaryDirectory() as tempdir:
            df = pd.DataFrame(np.random.rand(300, 3), columns=["a", "b", "c"])

            file_paths = [os.path.join(tempdir, f"test{i}.csv") for i in range(3)]
            df[:100].to_csv(file_paths[0])
            df[100:200].to_csv(file_paths[1])
            df[200:].to_csv(file_paths[2])

            mdf = (
                md.read_csv(
                    file_paths,
                    index_col=0,
                    merge_small_file_options=merge_small_file_option,
                )
                .execute()
                .fetch()
            )
            pd.testing.assert_frame_equal(df, mdf)

            mdf2 = (
                md.read_csv(file_paths, index_col=0, chunk_bytes=50).execute().fetch()
            )
            pd.testing.assert_frame_equal(df, mdf2)

    # test wildcards in path
    with tempfile.TemporaryDirectory() as tempdir:
        df = pd.DataFrame(np.random.rand(300, 3), columns=["a", "b", "c"])

        file_paths = [os.path.join(tempdir, f"test{i}.csv") for i in range(3)]
        df[:100].to_csv(file_paths[0])
        df[100:200].to_csv(file_paths[1])
        df[200:].to_csv(file_paths[2])

        # As we can not guarantee the order in which these files are processed,
        # the result may not keep the original order.
        mdf = md.read_csv(f"{tempdir}/*.csv", index_col=0).execute().fetch()
        pd.testing.assert_frame_equal(df, mdf.sort_index())

        mdf2 = (
            md.read_csv(f"{tempdir}/*.csv", index_col=0, chunk_bytes=50)
            .execute()
            .fetch()
        )
        pd.testing.assert_frame_equal(df, mdf2.sort_index())

    # test read directory
    with tempfile.TemporaryDirectory() as tempdir:
        testdir = os.path.join(tempdir, "test_dir")
        os.makedirs(testdir, exist_ok=True)

        df = pd.DataFrame(np.random.rand(300, 3), columns=["a", "b", "c"])

        file_paths = [os.path.join(testdir, f"test{i}.csv") for i in range(3)]
        df[:100].to_csv(file_paths[0])
        df[100:200].to_csv(file_paths[1])
        df[200:].to_csv(file_paths[2])

        # As we can not guarantee the order in which these files are processed,
        # the result may not keep the original order.
        mdf = md.read_csv(testdir, index_col=0).execute().fetch()
        pd.testing.assert_frame_equal(df, mdf.sort_index())

        mdf2 = md.read_csv(testdir, index_col=0, chunk_bytes=50).execute().fetch()
        pd.testing.assert_frame_equal(df, mdf2.sort_index())


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_read_csv_use_arrow_dtype(setup):
    rs = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "col1": rs.rand(100),
            "col2": rs.choice(["a" * 2, "b" * 3, "c" * 4], (100,)),
            "col3": np.arange(100),
        }
    )
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")
        df.to_csv(file_path, index=False)

        pdf = pd.read_csv(file_path)
        mdf = md.read_csv(file_path, use_arrow_dtype=True)
        result = mdf.execute().fetch()
        assert isinstance(mdf.dtypes.iloc[1], md.ArrowStringDtype)
        assert isinstance(result.dtypes.iloc[1], md.ArrowStringDtype)
        pd.testing.assert_frame_equal(arrow_array_to_objects(result), pdf)

    with tempfile.TemporaryDirectory() as tempdir:
        with option_context({"dataframe.use_arrow_dtype": True}):
            file_path = os.path.join(tempdir, "test.csv")
            df.to_csv(file_path, index=False)

            pdf = pd.read_csv(file_path)
            mdf = md.read_csv(file_path)
            result = mdf.execute().fetch()
            assert isinstance(mdf.dtypes.iloc[1], md.ArrowStringDtype)
            assert isinstance(result.dtypes.iloc[1], md.ArrowStringDtype)
            pd.testing.assert_frame_equal(arrow_array_to_objects(result), pdf)

    # test compression
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.gzip")
        df.to_csv(file_path, compression="gzip", index=False)

        pdf = pd.read_csv(file_path, compression="gzip")
        mdf = md.read_csv(file_path, compression="gzip", use_arrow_dtype=True)
        result = mdf.execute().fetch()
        assert isinstance(mdf.dtypes.iloc[1], md.ArrowStringDtype)
        assert isinstance(result.dtypes.iloc[1], md.ArrowStringDtype)
        pd.testing.assert_frame_equal(arrow_array_to_objects(result), pdf)


@require_cudf
def test_read_csv_gpu_execution(setup_gpu):
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")

        df = pd.DataFrame(
            {
                "col1": np.random.rand(100),
                "col2": np.random.choice(["a", "b", "c"], (100,)),
                "col3": np.arange(100),
            }
        )
        df.to_csv(file_path, index=False)

        pdf = pd.read_csv(file_path)
        mdf = md.read_csv(file_path, gpu=True).execute().fetch()
        pd.testing.assert_frame_equal(
            pdf.reset_index(drop=True), mdf.to_pandas().reset_index(drop=True)
        )

        mdf2 = md.read_csv(file_path, gpu=True, chunk_bytes=200).execute().fetch()
        pd.testing.assert_frame_equal(
            pdf.reset_index(drop=True), mdf2.to_pandas().reset_index(drop=True)
        )


def test_read_csv_without_index(setup):
    # test csv file without storing index
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")

        df = pd.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
        )
        df.to_csv(file_path, index=False)

        pdf = pd.read_csv(file_path)
        mdf = md.read_csv(file_path).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf)

        mdf2 = md.read_csv(file_path, chunk_bytes=10).execute().fetch()
        pd.testing.assert_frame_equal(pdf, mdf2)

        file_path2 = os.path.join(tempdir, "test.csv")
        df = pd.DataFrame(
            np.random.RandomState(0).rand(100, 10),
            columns=[f"col{i}" for i in range(10)],
        )
        df.to_csv(file_path2, index=False)

        mdf3 = md.read_csv(file_path2, chunk_bytes=os.stat(file_path2).st_size / 5)
        result = mdf3.execute().fetch()
        expected = pd.read_csv(file_path2)
        pd.testing.assert_frame_equal(result, expected)

        # test incremental_index = False
        mdf4 = md.read_csv(
            file_path2,
            chunk_bytes=os.stat(file_path2).st_size / 5,
            incremental_index=False,
        )
        result = mdf4.execute().fetch()
        assert not result.index.is_monotonic_increasing
        expected = pd.read_csv(file_path2)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


@pytest.mark.skipif(sqlalchemy is None, reason="sqlalchemy not installed")
def test_read_sql_execution(setup):
    import sqlalchemy as sa

    rs = np.random.RandomState(0)
    test_df = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
            "c": rs.rand(10),
            "d": [
                datetime.fromtimestamp(time.time() + 3600 * (i - 5)) for i in range(10)
            ],
        }
    )

    with tempfile.TemporaryDirectory() as d:
        table_name = "test"
        table_name2 = "test2"
        uri = "sqlite:///" + os.path.join(d, "test.db")

        test_df.to_sql(table_name, uri, index=False)

        # test read with table name
        r = md.read_sql_table("test", uri, chunk_size=4)
        result = r.execute().fetch()
        pd.testing.assert_frame_equal(result, test_df)

        # test read with sql string and offset method
        r = md.read_sql_query(
            "select * from test where c > 0.5", uri, parse_dates=["d"], chunk_size=4
        )
        result = r.execute().fetch()
        pd.testing.assert_frame_equal(
            result, test_df[test_df.c > 0.5].reset_index(drop=True)
        )

        # test read with sql string and partition method with integer cols
        r = md.read_sql(
            "select * from test where b > 's5'",
            uri,
            parse_dates=["d"],
            partition_col="a",
            num_partitions=3,
        )
        result = r.execute().fetch()
        pd.testing.assert_frame_equal(
            result, test_df[test_df.b > "s5"].reset_index(drop=True)
        )

        # test read with sql string and partition method with datetime cols
        r = md.read_sql_query(
            "select * from test where b > 's5'",
            uri,
            parse_dates={"d": "%Y-%m-%d %H:%M:%S"},
            partition_col="d",
            num_partitions=3,
        )
        result = r.execute().fetch()
        pd.testing.assert_frame_equal(
            result, test_df[test_df.b > "s5"].reset_index(drop=True)
        )

        # test read with sql string and partition method with datetime cols
        r = md.read_sql_query(
            "select * from test where b > 's5'",
            uri,
            parse_dates=["d"],
            partition_col="d",
            num_partitions=3,
            index_col="d",
        )
        result = r.execute().fetch()
        pd.testing.assert_frame_equal(result, test_df[test_df.b > "s5"].set_index("d"))

        # test SQL that return no result
        r = md.read_sql_query("select * from test where a > 1000", uri)
        result = r.execute().fetch()
        pd.testing.assert_frame_equal(result, pd.DataFrame(columns=test_df.columns))

        engine = sa.create_engine(uri)
        m = sa.MetaData()
        try:
            # test index_col and columns
            r = md.read_sql_table(
                "test",
                engine.connect(),
                chunk_size=4,
                index_col="a",
                columns=["b", "d"],
            )
            result = r.execute().fetch()
            expected = test_df.copy(deep=True)
            expected.set_index("a", inplace=True)
            del expected["c"]
            pd.testing.assert_frame_equal(result, expected)

            # do not specify chunk_size
            r = md.read_sql_table(
                "test", engine.connect(), index_col="a", columns=["b", "d"]
            )
            result = r.execute().fetch()
            pd.testing.assert_frame_equal(result, expected)

            table = sa.Table(table_name, m, autoload=True, autoload_with=engine)
            r = md.read_sql_table(
                table,
                engine,
                chunk_size=4,
                index_col=[table.columns["a"], table.columns["b"]],
                columns=[table.columns["c"], "d"],
            )
            result = r.execute().fetch()
            expected = test_df.copy(deep=True)
            expected.set_index(["a", "b"], inplace=True)
            pd.testing.assert_frame_equal(result, expected)

            # test table with primary key
            sa.Table(
                table_name2,
                m,
                sa.Column("id", sa.Integer, primary_key=True),
                sa.Column("a", sa.Integer),
                sa.Column("b", sa.String),
                sa.Column("c", sa.Float),
                sa.Column("d", sa.DateTime),
            )
            m.create_all(engine)
            test_df = test_df.copy(deep=True)
            test_df.index.name = "id"
            test_df.to_sql(table_name2, uri, if_exists="append")

            r = md.read_sql_table(table_name2, engine, chunk_size=4, index_col="id")
            result = r.execute().fetch()
            pd.testing.assert_frame_equal(result, test_df)
        finally:
            engine.dispose()


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_read_sql_use_arrow_dtype(setup):
    rs = np.random.RandomState(0)
    test_df = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
            "c": rs.rand(10),
            "d": [
                datetime.fromtimestamp(time.time() + 3600 * (i - 5)) for i in range(10)
            ],
        }
    )

    with tempfile.TemporaryDirectory() as d:
        table_name = "test"
        uri = "sqlite:///" + os.path.join(d, "test.db")

        test_df.to_sql(table_name, uri, index=False)

        r = md.read_sql_table("test", uri, chunk_size=4, use_arrow_dtype=True)
        result = r.execute().fetch()
        assert isinstance(r.dtypes.iloc[1], md.ArrowStringDtype)
        assert isinstance(result.dtypes.iloc[1], md.ArrowStringDtype)
        pd.testing.assert_frame_equal(arrow_array_to_objects(result), test_df)

        # test read with sql string and offset method
        r = md.read_sql_query(
            "select * from test where c > 0.5",
            uri,
            parse_dates=["d"],
            chunk_size=4,
            use_arrow_dtype=True,
        )
        result = r.execute().fetch()
        assert isinstance(r.dtypes.iloc[1], md.ArrowStringDtype)
        assert isinstance(result.dtypes.iloc[1], md.ArrowStringDtype)
        pd.testing.assert_frame_equal(
            arrow_array_to_objects(result),
            test_df[test_df.c > 0.5].reset_index(drop=True),
        )


def test_date_range_execution(setup):
    for closed in [None, "left", "right"]:
        # start, periods, freq
        dr = md.date_range("2020-1-1", periods=10, chunk_size=3, closed=closed)

        result = dr.execute().fetch()
        expected = pd.date_range("2020-1-1", periods=10, closed=closed)
        pd.testing.assert_index_equal(result, expected)

        # end, periods, freq
        dr = md.date_range(end="2020-1-10", periods=10, chunk_size=3, closed=closed)

        result = dr.execute().fetch()
        expected = pd.date_range(end="2020-1-10", periods=10, closed=closed)
        pd.testing.assert_index_equal(result, expected)

        # start, end, freq
        dr = md.date_range("2020-1-1", "2020-1-10", chunk_size=3, closed=closed)

        result = dr.execute().fetch()
        expected = pd.date_range("2020-1-1", "2020-1-10", closed=closed)
        pd.testing.assert_index_equal(result, expected)

        # start, end and periods
        dr = md.date_range(
            "2020-1-1", "2020-1-10", periods=19, chunk_size=3, closed=closed
        )

        result = dr.execute().fetch()
        expected = pd.date_range("2020-1-1", "2020-1-10", periods=19, closed=closed)
        pd.testing.assert_index_equal(result, expected)

        # start, end and freq
        dr = md.date_range(
            "2020-1-1", "2020-1-10", freq="12H", chunk_size=3, closed=closed
        )

        result = dr.execute().fetch()
        expected = pd.date_range("2020-1-1", "2020-1-10", freq="12H", closed=closed)
        pd.testing.assert_index_equal(result, expected)

    # test timezone
    dr = md.date_range("2020-1-1", periods=10, tz="Asia/Shanghai", chunk_size=7)

    result = dr.execute().fetch()
    expected = pd.date_range("2020-1-1", periods=10, tz="Asia/Shanghai")
    pd.testing.assert_index_equal(result, expected)

    # test periods=0
    dr = md.date_range("2020-1-1", periods=0)

    result = dr.execute().fetch()
    expected = pd.date_range("2020-1-1", periods=0)
    pd.testing.assert_index_equal(result, expected)

    # test start == end
    dr = md.date_range("2020-1-1", "2020-1-1", periods=1)

    result = dr.execute().fetch()
    expected = pd.date_range("2020-1-1", "2020-1-1", periods=1)
    pd.testing.assert_index_equal(result, expected)

    # test normalize=True
    dr = md.date_range("2020-1-1", periods=10, normalize=True, chunk_size=4)

    result = dr.execute().fetch()
    expected = pd.date_range("2020-1-1", periods=10, normalize=True)
    pd.testing.assert_index_equal(result, expected)

    # test freq
    dr = md.date_range(start="1/1/2018", periods=5, freq="M", chunk_size=3)

    result = dr.execute().fetch()
    expected = pd.date_range(start="1/1/2018", periods=5, freq="M")
    pd.testing.assert_index_equal(result, expected)


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_read_parquet_arrow(setup):
    test_df = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
            "c": np.random.rand(10),
        }
    )

    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")
        test_df.to_parquet(file_path)

        df = md.read_parquet(file_path)
        result = df.execute().fetch()
        pd.testing.assert_frame_equal(result, test_df)
        # size_res = self.executor.execute_dataframe(df, mock=True)
        # assert sum(s[0] for s in size_res) > test_df.memory_usage(deep=True).sum()

    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.parquet")
        test_df.to_parquet(file_path, row_group_size=3)

        df = md.read_parquet(file_path, groups_as_chunks=True, columns=["a", "b"])
        result = df.execute().fetch()
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), test_df[["a", "b"]]
        )

    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.parquet")
        test_df.to_parquet(file_path, row_group_size=5)

        df = md.read_parquet(
            file_path,
            groups_as_chunks=True,
            use_arrow_dtype=True,
            incremental_index=True,
        )
        result = df.execute().fetch()
        assert isinstance(df.dtypes.iloc[1], md.ArrowStringDtype)
        assert isinstance(result.dtypes.iloc[1], md.ArrowStringDtype)
        pd.testing.assert_frame_equal(arrow_array_to_objects(result), test_df)

    # test wildcards in path
    for merge_small_file_option in [{"n_sample_file": 1}, None]:
        with tempfile.TemporaryDirectory() as tempdir:
            df = pd.DataFrame(
                {
                    "a": np.arange(300).astype(np.int64, copy=False),
                    "b": [f"s{i}" for i in range(300)],
                    "c": np.random.rand(300),
                }
            )

            file_paths = [os.path.join(tempdir, f"test{i}.parquet") for i in range(3)]
            df[:100].to_parquet(file_paths[0], row_group_size=50)
            df[100:200].to_parquet(file_paths[1], row_group_size=30)
            df[200:].to_parquet(file_paths[2])

            mdf = md.read_parquet(f"{tempdir}/*.parquet")
            r = mdf.execute().fetch()
            pd.testing.assert_frame_equal(df, r.sort_values("a").reset_index(drop=True))

            mdf = md.read_parquet(
                f"{tempdir}/*.parquet",
                groups_as_chunks=True,
                merge_small_file_options=merge_small_file_option,
            )
            r = mdf.execute().fetch()
            pd.testing.assert_frame_equal(df, r.sort_values("a").reset_index(drop=True))


@pytest.mark.skipif(fastparquet is None, reason="fastparquet not installed")
def test_read_parquet_fast_parquet(setup):
    test_df = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
            "c": np.random.rand(10),
        }
    )

    # test fastparquet engine
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")
        test_df.to_parquet(file_path, compression=None)

        df = md.read_parquet(file_path, engine="fastparquet")
        result = df.execute().fetch()
        pd.testing.assert_frame_equal(result, test_df)
        # size_res = self.executor.execute_dataframe(df, mock=True)
        # assert sum(s[0] for s in size_res) > test_df.memory_usage(deep=True).sum()


@require_ray
def test_read_raydataset(ray_start_regular, ray_create_mars_cluster):
    test_df1 = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
        }
    )
    test_df2 = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
        }
    )
    df = pd.concat([test_df1, test_df2])
    ds = ray.data.from_pandas_refs([ray.put(test_df1), ray.put(test_df2)])
    mdf = md.read_raydataset(ds)
    assert df.equals(mdf.execute().fetch())


@require_ray
def test_read_ray_mldataset(ray_start_regular, ray_create_mars_cluster):
    test_dfs = [
        pd.DataFrame(
            {
                "a": np.arange(i * 10, (i + 1) * 10).astype(np.int64, copy=False),
                "b": [f"s{j}" for j in range(i * 10, (i + 1) * 10)],
            }
        )
        for i in range(5)
    ]
    import ray.util.iter
    from ray.util.data import from_parallel_iter

    ml_dataset = from_parallel_iter(
        ray.util.iter.from_items(test_dfs, num_shards=4), need_convert=False
    )
    dfs = []
    for shard in ml_dataset.shards():
        dfs.extend(list(shard))
    df = pd.concat(dfs).reset_index(drop=True)
    mdf = md.read_ray_mldataset(ml_dataset)
    pd.testing.assert_frame_equal(df, mdf.execute().fetch())
    pd.testing.assert_frame_equal(df.head(5), mdf.head(5).execute().fetch())
    pd.testing.assert_frame_equal(df.head(15), mdf.head(15).execute().fetch())
