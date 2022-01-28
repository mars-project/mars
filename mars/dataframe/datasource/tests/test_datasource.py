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
import shutil
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from .... import tensor as mt
from ....core import tile
from ....tests.core import require_ray
from ....utils import lazy_import
from ...core import IndexValue, DatetimeIndex, Int64Index, Float64Index
from ..core import merge_small_files
from ..dataframe import from_pandas as from_pandas_df
from ..date_range import date_range
from ..from_records import from_records
from ..from_tensor import (
    dataframe_from_tensor,
    series_from_tensor,
    dataframe_from_1d_tileables,
)
from ..index import from_pandas as from_pandas_index, from_tileable
from ..read_csv import read_csv, DataFrameReadCSV
from ..read_sql import read_sql_table, read_sql_query, DataFrameReadSQL
from ..read_raydataset import (
    read_raydataset,
    DataFrameReadRayDataset,
    read_ray_mldataset,
    DataFrameReadMLDataset,
)
from ..series import from_pandas as from_pandas_series


ray = lazy_import("ray")


def test_from_pandas_dataframe():
    data = pd.DataFrame(
        np.random.rand(10, 10), columns=["c" + str(i) for i in range(10)]
    )
    df = from_pandas_df(data, chunk_size=4)

    pd.testing.assert_series_equal(df.op.dtypes, data.dtypes)
    assert isinstance(df.index_value._index_value, IndexValue.RangeIndex)
    assert df.index_value._index_value._slice == slice(0, 10, 1)
    assert df.index_value.is_monotonic_increasing is True
    assert df.index_value.is_monotonic_decreasing is False
    assert df.index_value.is_unique is True
    assert df.index_value.min_val == 0
    assert df.index_value.max_val == 9
    np.testing.assert_equal(df.columns_value._index_value._data, data.columns.values)

    df = tile(df)

    assert len(df.chunks) == 9
    pd.testing.assert_frame_equal(df.chunks[0].op.data, df.op.data.iloc[:4, :4])
    assert df.chunks[0].index_value._index_value._slice == slice(0, 4, 1)
    assert df.chunks[0].index_value._index_value._is_monotonic_increasing is True
    assert df.chunks[0].index_value._index_value._is_monotonic_decreasing is False
    assert df.chunks[0].index_value._index_value._is_unique is True
    pd.testing.assert_frame_equal(df.chunks[1].op.data, df.op.data.iloc[:4, 4:8])
    assert df.chunks[1].index_value._index_value._slice == slice(0, 4, 1)
    assert df.chunks[1].index_value._index_value._is_monotonic_increasing is True
    assert df.chunks[1].index_value._index_value._is_monotonic_decreasing is False
    assert df.chunks[1].index_value._index_value._is_unique is True
    pd.testing.assert_frame_equal(df.chunks[2].op.data, df.op.data.iloc[:4, 8:])
    assert df.chunks[2].index_value._index_value._slice == slice(0, 4, 1)
    assert df.chunks[2].index_value._index_value._is_monotonic_increasing is True
    assert df.chunks[2].index_value._index_value._is_monotonic_decreasing is False
    assert df.chunks[2].index_value._index_value._is_unique is True
    pd.testing.assert_frame_equal(df.chunks[3].op.data, df.op.data.iloc[4:8, :4])
    assert df.chunks[3].index_value._index_value._slice == slice(4, 8, 1)
    assert df.chunks[3].index_value._index_value._is_monotonic_increasing is True
    assert df.chunks[3].index_value._index_value._is_monotonic_decreasing is False
    assert df.chunks[3].index_value._index_value._is_unique is True
    pd.testing.assert_frame_equal(df.chunks[4].op.data, df.op.data.iloc[4:8, 4:8])
    assert df.chunks[4].index_value._index_value._slice == slice(4, 8, 1)
    assert df.chunks[4].index_value._index_value._is_monotonic_increasing is True
    assert df.chunks[4].index_value._index_value._is_monotonic_decreasing is False
    assert df.chunks[4].index_value._index_value._is_unique is True
    pd.testing.assert_frame_equal(df.chunks[5].op.data, df.op.data.iloc[4:8, 8:])
    assert df.chunks[5].index_value._index_value._slice == slice(4, 8, 1)
    assert df.chunks[5].index_value._index_value._is_monotonic_increasing is True
    assert df.chunks[5].index_value._index_value._is_monotonic_decreasing is False
    assert df.chunks[5].index_value._index_value._is_unique is True
    pd.testing.assert_frame_equal(df.chunks[6].op.data, df.op.data.iloc[8:, :4])
    assert df.chunks[6].index_value._index_value._slice == slice(8, 10, 1)
    assert df.chunks[6].index_value._index_value._is_monotonic_increasing is True
    assert df.chunks[6].index_value._index_value._is_monotonic_decreasing is False
    assert df.chunks[6].index_value._index_value._is_unique is True
    pd.testing.assert_frame_equal(df.chunks[7].op.data, df.op.data.iloc[8:, 4:8])
    assert df.chunks[7].index_value._index_value._slice == slice(8, 10, 1)
    assert df.chunks[7].index_value._index_value._is_monotonic_increasing is True
    assert df.chunks[7].index_value._index_value._is_monotonic_decreasing is False
    assert df.chunks[7].index_value._index_value._is_unique is True
    pd.testing.assert_frame_equal(df.chunks[8].op.data, df.op.data.iloc[8:, 8:])
    assert df.chunks[8].index_value._index_value._slice == slice(8, 10, 1)
    assert df.chunks[8].index_value._index_value._is_monotonic_increasing is True
    assert df.chunks[8].index_value._index_value._is_monotonic_decreasing is False
    assert df.chunks[8].index_value._index_value._is_unique is True

    data2 = data[::2]
    df2 = from_pandas_df(data2, chunk_size=4)

    pd.testing.assert_series_equal(df.op.dtypes, data2.dtypes)
    assert isinstance(df2.index_value._index_value, IndexValue.RangeIndex)
    assert df2.index_value._index_value._slice == slice(0, 10, 2)

    df2 = tile(df2)

    assert len(df2.chunks) == 6
    pd.testing.assert_frame_equal(df2.chunks[0].op.data, df2.op.data.iloc[:4, :4])
    assert df2.chunks[0].index_value._index_value._slice == slice(0, 8, 2)
    pd.testing.assert_frame_equal(df2.chunks[1].op.data, df2.op.data.iloc[:4, 4:8])
    assert df2.chunks[1].index_value._index_value._slice == slice(0, 8, 2)
    pd.testing.assert_frame_equal(df2.chunks[2].op.data, df2.op.data.iloc[:4, 8:])
    assert df2.chunks[2].index_value._index_value._slice == slice(0, 8, 2)
    pd.testing.assert_frame_equal(df2.chunks[3].op.data, df2.op.data.iloc[4:, :4])
    assert df2.chunks[3].index_value._index_value._slice == slice(8, 10, 2)
    pd.testing.assert_frame_equal(df2.chunks[4].op.data, df2.op.data.iloc[4:, 4:8])
    assert df2.chunks[3].index_value._index_value._slice == slice(8, 10, 2)
    pd.testing.assert_frame_equal(df2.chunks[5].op.data, df2.op.data.iloc[4:, 8:])
    assert df2.chunks[3].index_value._index_value._slice == slice(8, 10, 2)


def test_from_pandas_series():
    data = pd.Series(np.random.rand(10), name="a")
    series = from_pandas_series(data, chunk_size=4)

    assert series.name == data.name
    assert isinstance(series.index_value._index_value, IndexValue.RangeIndex)
    assert series.index_value._index_value._slice == slice(0, 10, 1)
    assert series.index_value.is_monotonic_increasing is True
    assert series.index_value.is_monotonic_decreasing is False
    assert series.index_value.is_unique is True
    assert series.index_value.min_val == 0
    assert series.index_value.max_val == 9

    series = tile(series)

    assert len(series.chunks) == 3
    pd.testing.assert_series_equal(series.chunks[0].op.data, series.op.data.iloc[:4])
    assert series.chunks[0].index_value._index_value._slice == slice(0, 4, 1)
    assert series.chunks[0].index_value._index_value._is_monotonic_increasing is True
    assert series.chunks[0].index_value._index_value._is_monotonic_decreasing is False
    assert series.chunks[0].index_value._index_value._is_unique is True
    pd.testing.assert_series_equal(series.chunks[1].op.data, series.op.data.iloc[4:8])
    assert series.chunks[1].index_value._index_value._slice == slice(4, 8, 1)
    assert series.chunks[1].index_value._index_value._is_monotonic_increasing is True
    assert series.chunks[1].index_value._index_value._is_monotonic_decreasing is False
    assert series.chunks[1].index_value._index_value._is_unique is True
    pd.testing.assert_series_equal(series.chunks[2].op.data, series.op.data.iloc[8:])
    assert series.chunks[2].index_value._index_value._slice == slice(8, 10, 1)
    assert series.chunks[2].index_value._index_value._is_monotonic_increasing is True
    assert series.chunks[2].index_value._index_value._is_monotonic_decreasing is False
    assert series.chunks[2].index_value._index_value._is_unique is True


def test_from_pandas_index():
    data = pd.date_range("2020-1-1", periods=10, name="date")
    index = from_pandas_index(data, chunk_size=4)

    assert isinstance(index, DatetimeIndex)
    assert index.name == data.name
    assert index.dtype == data.dtype
    assert isinstance(index.index_value.value, IndexValue.DatetimeIndex)

    index = tile(index)

    for i, c in enumerate(index.chunks):
        assert c.name == data.name
        pd.testing.assert_index_equal(c.op.data, data[i * 4 : (i + 1) * 4])
        assert c.dtype == data.dtype
        assert isinstance(c.index_value.value, IndexValue.DatetimeIndex)


def test_from_tileable_index():
    t = mt.random.rand(10, 4)

    with pytest.raises(ValueError):
        from_tileable(t)

    pd_df = pd.DataFrame(
        np.random.rand(10, 4), index=np.arange(10, 0, -1).astype(np.int64)
    )
    pd_df.index.name = "ind"
    df = from_pandas_df(pd_df, chunk_size=6)

    for o in [df, df[0]]:
        index = o.index
        assert isinstance(index, Int64Index)
        assert index.dtype == np.int64
        assert index.name == pd_df.index.name
        assert isinstance(index.index_value.value, IndexValue.Int64Index)

        index = tile(index)

        assert len(index.chunks) == 2
        for c in index.chunks:
            assert c.dtype == np.int64
            assert c.name == pd_df.index.name
            assert isinstance(c.index_value.value, IndexValue.Int64Index)

    t = mt.random.rand(10, chunk_size=6)
    index = from_tileable(t, name="new_name")

    assert isinstance(index, Float64Index)
    assert index.dtype == np.float64
    assert index.name == "new_name"
    assert isinstance(index.index_value.value, IndexValue.Float64Index)

    index = tile(index)

    assert len(index.chunks) == 2
    for c in index.chunks:
        assert c.dtype == np.float64
        assert c.name == "new_name"
        assert isinstance(c.index_value.value, IndexValue.Float64Index)


def test_from_tensor():
    tensor = mt.random.rand(10, 10, chunk_size=5)
    df = dataframe_from_tensor(tensor)
    assert isinstance(df.index_value._index_value, IndexValue.RangeIndex)
    assert df.op.dtypes[0] == tensor.dtype

    df = tile(df)
    assert len(df.chunks) == 4
    assert isinstance(df.chunks[0].index_value._index_value, IndexValue.RangeIndex)
    assert isinstance(df.chunks[0].index_value, IndexValue)

    # test converted from 1-d tensor
    tensor2 = mt.array([1, 2, 3])
    # in fact, tensor3 is (3,1)
    tensor3 = mt.array([tensor2]).T

    df2 = dataframe_from_tensor(tensor2)
    df3 = dataframe_from_tensor(tensor3)
    df2 = tile(df2)
    df3 = tile(df3)
    np.testing.assert_equal(df2.chunks[0].index, (0, 0))
    np.testing.assert_equal(df3.chunks[0].index, (0, 0))

    # test converted from scalar
    scalar = mt.array(1)
    np.testing.assert_equal(scalar.ndim, 0)
    with pytest.raises(TypeError):
        dataframe_from_tensor(scalar)

    # from tensor with given index
    df = dataframe_from_tensor(tensor, index=np.arange(0, 20, 2))
    df = tile(df)
    pd.testing.assert_index_equal(
        df.chunks[0].index_value.to_pandas(), pd.Index(np.arange(0, 10, 2))
    )
    pd.testing.assert_index_equal(
        df.chunks[1].index_value.to_pandas(), pd.Index(np.arange(0, 10, 2))
    )
    pd.testing.assert_index_equal(
        df.chunks[2].index_value.to_pandas(), pd.Index(np.arange(10, 20, 2))
    )
    pd.testing.assert_index_equal(
        df.chunks[3].index_value.to_pandas(), pd.Index(np.arange(10, 20, 2))
    )

    # from tensor with index that is a tensor as well
    df = dataframe_from_tensor(tensor, index=mt.arange(0, 20, 2))
    df = tile(df)
    assert len(df.chunks[0].inputs) == 2
    assert df.chunks[0].index_value.has_value() is False

    # from tensor with given columns
    df = dataframe_from_tensor(tensor, columns=list("abcdefghij"))
    df = tile(df)
    pd.testing.assert_index_equal(df.dtypes.index, pd.Index(list("abcdefghij")))
    pd.testing.assert_index_equal(
        df.chunks[0].columns_value.to_pandas(), pd.Index(["a", "b", "c", "d", "e"])
    )
    pd.testing.assert_index_equal(
        df.chunks[0].dtypes.index, pd.Index(["a", "b", "c", "d", "e"])
    )
    pd.testing.assert_index_equal(
        df.chunks[1].columns_value.to_pandas(), pd.Index(["f", "g", "h", "i", "j"])
    )
    pd.testing.assert_index_equal(
        df.chunks[1].dtypes.index, pd.Index(["f", "g", "h", "i", "j"])
    )
    pd.testing.assert_index_equal(
        df.chunks[2].columns_value.to_pandas(), pd.Index(["a", "b", "c", "d", "e"])
    )
    pd.testing.assert_index_equal(
        df.chunks[2].dtypes.index, pd.Index(["a", "b", "c", "d", "e"])
    )
    pd.testing.assert_index_equal(
        df.chunks[3].columns_value.to_pandas(), pd.Index(["f", "g", "h", "i", "j"])
    )
    pd.testing.assert_index_equal(
        df.chunks[3].dtypes.index, pd.Index(["f", "g", "h", "i", "j"])
    )

    # test series from tensor
    tensor = mt.random.rand(10, chunk_size=4)
    series = series_from_tensor(tensor, name="a")

    assert series.dtype == tensor.dtype
    assert series.name == "a"
    pd.testing.assert_index_equal(series.index_value.to_pandas(), pd.RangeIndex(10))

    series = tile(series)
    assert len(series.chunks) == 3
    pd.testing.assert_index_equal(
        series.chunks[0].index_value.to_pandas(), pd.RangeIndex(0, 4)
    )
    assert series.chunks[0].name == "a"
    pd.testing.assert_index_equal(
        series.chunks[1].index_value.to_pandas(), pd.RangeIndex(4, 8)
    )
    assert series.chunks[1].name == "a"
    pd.testing.assert_index_equal(
        series.chunks[2].index_value.to_pandas(), pd.RangeIndex(8, 10)
    )
    assert series.chunks[2].name == "a"

    d = OrderedDict(
        [(0, mt.tensor(np.random.rand(4))), (1, mt.tensor(np.random.rand(4)))]
    )
    df = dataframe_from_1d_tileables(d)
    pd.testing.assert_index_equal(df.columns_value.to_pandas(), pd.RangeIndex(2))

    df = tile(df)

    pd.testing.assert_index_equal(
        df.chunks[0].index_value.to_pandas(), pd.RangeIndex(4)
    )

    series = series_from_tensor(mt.random.rand(4))
    pd.testing.assert_index_equal(series.index_value.to_pandas(), pd.RangeIndex(4))

    series = series_from_tensor(mt.random.rand(4), index=[1, 2, 3])
    pd.testing.assert_index_equal(series.index_value.to_pandas(), pd.Index([1, 2, 3]))

    series = series_from_tensor(
        mt.random.rand(4), index=pd.Index([1, 2, 3], name="my_index")
    )
    pd.testing.assert_index_equal(
        series.index_value.to_pandas(), pd.Index([1, 2, 3], name="my_index")
    )
    assert series.index_value.name == "my_index"

    with pytest.raises(TypeError):
        series_from_tensor(mt.ones((10, 10)))

    # index has wrong shape
    with pytest.raises(ValueError):
        dataframe_from_tensor(mt.random.rand(4, 3), index=mt.random.rand(5))

    # columns have wrong shape
    with pytest.raises(ValueError):
        dataframe_from_tensor(mt.random.rand(4, 3), columns=["a", "b"])

    # index should be 1-d
    with pytest.raises(ValueError):
        dataframe_from_tensor(
            mt.tensor(np.random.rand(3, 2)), index=mt.tensor(np.random.rand(3, 2))
        )

    # 1-d tensors should have same shape
    with pytest.raises(ValueError):
        dataframe_from_1d_tileables(
            OrderedDict(
                [(0, mt.tensor(np.random.rand(3))), (1, mt.tensor(np.random.rand(2)))]
            )
        )

    # index has wrong shape
    with pytest.raises(ValueError):
        dataframe_from_1d_tileables(
            {0: mt.tensor(np.random.rand(3))}, index=mt.tensor(np.random.rand(2))
        )

    # columns have wrong shape
    with pytest.raises(ValueError):
        dataframe_from_1d_tileables(
            {0: mt.tensor(np.random.rand(3))}, columns=["a", "b"]
        )

    # index should be 1-d
    with pytest.raises(ValueError):
        series_from_tensor(mt.random.rand(4), index=mt.random.rand(4, 3))


def test_from_records():
    dtype = np.dtype([("x", "int"), ("y", "double"), ("z", "<U16")])

    tensor = mt.ones((10,), dtype=dtype, chunk_size=3)
    df = from_records(tensor)
    df = tile(df)

    assert df.chunk_shape == (4, 1)
    assert df.chunks[0].shape == (3, 3)
    assert df.chunks[1].shape == (3, 3)
    assert df.chunks[2].shape == (3, 3)
    assert df.chunks[3].shape == (1, 3)

    assert df.chunks[0].inputs[0].shape == (3,)
    assert df.chunks[1].inputs[0].shape == (3,)
    assert df.chunks[2].inputs[0].shape == (3,)
    assert df.chunks[3].inputs[0].shape == (1,)

    assert df.chunks[0].op.extra_params == {"begin_index": 0, "end_index": 3}
    assert df.chunks[1].op.extra_params == {"begin_index": 3, "end_index": 6}
    assert df.chunks[2].op.extra_params == {"begin_index": 6, "end_index": 9}
    assert df.chunks[3].op.extra_params == {"begin_index": 9, "end_index": 10}

    names = pd.Index(["x", "y", "z"])
    dtypes = pd.Series(
        {"x": np.dtype("int"), "y": np.dtype("double"), "z": np.dtype("<U16")}
    )
    for chunk in df.chunks:
        pd.testing.assert_index_equal(chunk.columns_value.to_pandas(), names)
        pd.testing.assert_series_equal(chunk.dtypes, dtypes)

    pd.testing.assert_index_equal(
        df.chunks[0].index_value.to_pandas(), pd.RangeIndex(0, 3)
    )
    pd.testing.assert_index_equal(
        df.chunks[1].index_value.to_pandas(), pd.RangeIndex(3, 6)
    )
    pd.testing.assert_index_equal(
        df.chunks[2].index_value.to_pandas(), pd.RangeIndex(6, 9)
    )
    pd.testing.assert_index_equal(
        df.chunks[3].index_value.to_pandas(), pd.RangeIndex(9, 10)
    )


def test_read_csv():
    tempdir = tempfile.mkdtemp()
    file_path = os.path.join(tempdir, "test.csv")
    try:
        df = pd.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            columns=["a", "b", "c"],
            dtype=np.int64,
        )
        df.to_csv(file_path)
        mdf = read_csv(file_path, index_col=0, chunk_bytes=10)
        assert isinstance(mdf.op, DataFrameReadCSV)
        assert mdf.shape[1] == 3
        pd.testing.assert_index_equal(df.columns, mdf.columns_value.to_pandas())

        mdf = tile(mdf)
        assert len(mdf.chunks) == 4
        index_keys = set()
        for chunk in mdf.chunks:
            index_keys.add(chunk.index_value.key)
            pd.testing.assert_index_equal(df.columns, chunk.columns_value.to_pandas())
            pd.testing.assert_series_equal(df.dtypes, chunk.dtypes)
        assert len(index_keys) > 1
    finally:
        shutil.rmtree(tempdir)


def test_read_sql():
    test_df = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
        }
    )

    with tempfile.TemporaryDirectory() as d:
        table_name = "test"
        uri = "sqlite:///" + os.path.join(d, "test.db")

        test_df.to_sql(table_name, uri, index=False)

        df = read_sql_table(table_name, uri, chunk_size=4)

        assert df.shape == test_df.shape
        pd.testing.assert_index_equal(df.index_value.to_pandas(), test_df.index)
        pd.testing.assert_series_equal(df.dtypes, test_df.dtypes)

        df = tile(df)
        assert df.nsplits == ((4, 4, 2), (2,))
        for c in df.chunks:
            assert isinstance(c.op, DataFrameReadSQL)
            assert c.op.offset is not None

        with pytest.raises(NotImplementedError):
            read_sql_table(table_name, uri, chunksize=4, index_col=b"a")
        with pytest.raises(TypeError):
            read_sql_table(table_name, uri, chunk_size=4, index_col=b"a")
        with pytest.raises(TypeError):
            read_sql_query("select * from " + table_name, uri, partition_col="b")


@require_ray
def test_read_raydataset(ray_start_regular):
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
    mdf = read_raydataset(ds)

    assert mdf.shape[1] == 2
    pd.testing.assert_index_equal(df.columns, mdf.columns_value.to_pandas())
    pd.testing.assert_series_equal(df.dtypes, mdf.dtypes)

    mdf = tile(mdf)
    assert len(mdf.chunks) == 2
    for chunk in mdf.chunks:
        assert isinstance(chunk.op, DataFrameReadRayDataset)


def test_date_range():
    with pytest.raises(TypeError):
        _ = date_range("2020-1-1", periods="2")

    with pytest.raises(ValueError):
        _ = date_range("2020-1-1", "2020-1-10", periods=10, freq="D")

    with pytest.raises(ValueError):
        _ = date_range(pd.NaT, periods=10)

    expected = pd.date_range("2020-1-1", periods=9.0, name="date")

    dr = date_range("2020-1-1", periods=9.0, name="date", chunk_size=3)
    assert isinstance(dr, DatetimeIndex)
    assert dr.shape == (9,)
    assert dr.dtype == expected.dtype
    assert isinstance(dr.index_value.value, IndexValue.DatetimeIndex)
    assert dr.index_value.min_val == expected.min()
    assert dr.index_value.min_val_close is True
    assert dr.index_value.max_val == expected.max()
    assert dr.index_value.max_val_close is True
    assert dr.index_value.is_unique == expected.is_unique
    assert dr.index_value.is_monotonic_increasing == expected.is_monotonic_increasing
    assert dr.name == expected.name

    dr = tile(dr)

    for i, c in enumerate(dr.chunks):
        ec = expected[i * 3 : (i + 1) * 3]
        assert c.shape == (3,)
        assert c.dtype == ec.dtype
        assert isinstance(c.index_value.value, IndexValue.DatetimeIndex)
        assert c.index_value.min_val == ec.min()
        assert c.index_value.min_val_close is True
        assert c.index_value.max_val == ec.max()
        assert c.index_value.max_val_close is True
        assert c.index_value.is_unique == ec.is_unique
        assert c.index_value.is_monotonic_increasing == ec.is_monotonic_increasing
        assert c.name == ec.name


@require_ray
def test_read_ray_mldataset(ray_start_regular):
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
    import ray.util.iter
    from ray.util.data import from_parallel_iter

    ml_dataset = from_parallel_iter(
        ray.util.iter.from_items([test_df1, test_df2], num_shards=2), need_convert=False
    )
    mdf = read_ray_mldataset(ml_dataset)

    assert mdf.shape[1] == 2
    pd.testing.assert_index_equal(df.columns, mdf.columns_value.to_pandas())
    pd.testing.assert_series_equal(df.dtypes, mdf.dtypes)

    mdf = tile(mdf)
    assert len(mdf.chunks) == 2
    for chunk in mdf.chunks:
        assert isinstance(chunk.op, DataFrameReadMLDataset)


def test_merge_small_files():
    raw = pd.DataFrame(np.random.rand(16, 4))
    df = tile(from_pandas_df(raw, chunk_size=4))

    chunk_size = 4 * 4 * 8
    # number of chunks < 10
    assert df is merge_small_files(df, n_sample_file=10)
    # merged_chunk_size
    assert df is merge_small_files(
        df, n_sample_file=2, merged_file_size=chunk_size + 0.1
    )

    df2 = merge_small_files(df, n_sample_file=2, merged_file_size=2 * chunk_size)
    assert len(df2.chunks) == 2
    assert df2.chunks[0].shape == (8, 4)
    pd.testing.assert_index_equal(
        df2.chunks[0].index_value.to_pandas(), pd.RangeIndex(8)
    )
    assert df2.chunks[1].shape == (8, 4)
    pd.testing.assert_index_equal(
        df2.chunks[1].index_value.to_pandas(), pd.RangeIndex(8, 16)
    )
    assert df2.nsplits == ((8, 8), (4,))
