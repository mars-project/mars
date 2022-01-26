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

from .... import dataframe as md
from .... import tensor as mt
from ....core import tile
from ....tensor.core import TENSOR_CHUNK_TYPE, TENSOR_TYPE, Tensor
from ...core import (
    SERIES_CHUNK_TYPE,
    SERIES_TYPE,
    Series,
    DATAFRAME_TYPE,
    DataFrame,
    DATAFRAME_CHUNK_TYPE,
)
from ..iloc import (
    DataFrameIlocGetItem,
    DataFrameIlocSetItem,
    IndexingError,
    HeadTailOptimizedOperandMixin,
)
from ..loc import DataFrameLocGetItem


def test_set_index():
    df1 = pd.DataFrame(
        [[1, 3, 3], [4, 2, 6], [7, 8, 9]],
        index=["a1", "a2", "a3"],
        columns=["x", "y", "z"],
    )
    df2 = md.DataFrame(df1, chunk_size=2)

    df3 = df2.set_index("y", drop=True)
    df3 = tile(df3)
    assert df3.chunk_shape == (2, 2)
    pd.testing.assert_index_equal(
        df3.chunks[0].columns_value.to_pandas(), pd.Index(["x"])
    )
    pd.testing.assert_index_equal(
        df3.chunks[1].columns_value.to_pandas(), pd.Index(["z"])
    )

    df4 = df2.set_index("y", drop=False)
    df4 = tile(df4)
    assert df4.chunk_shape == (2, 2)
    pd.testing.assert_index_equal(
        df4.chunks[0].columns_value.to_pandas(), pd.Index(["x", "y"])
    )
    pd.testing.assert_index_equal(
        df4.chunks[1].columns_value.to_pandas(), pd.Index(["z"])
    )


def test_iloc_getitem():
    df1 = pd.DataFrame(
        [[1, 3, 3], [4, 2, 6], [7, 8, 9]],
        index=["a1", "a2", "a3"],
        columns=["x", "y", "z"],
    )
    df2 = md.DataFrame(df1, chunk_size=2)

    with pytest.raises(IndexingError):
        _ = df2.iloc[1, 1, 1]

    # index cannot be tuple
    with pytest.raises(IndexingError):
        _ = df2.iloc[((1,),)]

    # index wrong type
    with pytest.raises(TypeError):
        _ = df2.iloc["a1":]

    with pytest.raises(NotImplementedError):
        _ = df2.iloc[0, md.Series(["a2", "a3"])]

    # fancy index should be 1-d
    with pytest.raises(ValueError):
        _ = df2.iloc[[[0, 1], [1, 2]]]

    with pytest.raises(ValueError):
        _ = df2.iloc[1, ...]

    with pytest.raises(IndexError):
        _ = df2.iloc[-4]

    with pytest.raises(IndexError):
        _ = df2.iloc[3]

    # plain index
    df3 = df2.iloc[1]
    df3 = tile(df3)
    assert isinstance(df3, SERIES_TYPE)
    assert isinstance(df3.op, DataFrameIlocGetItem)
    assert df3.shape == (3,)
    assert df3.chunk_shape == (2,)
    assert df3.chunks[0].shape == (2,)
    assert df3.chunks[1].shape == (1,)
    assert df3.chunks[0].op.indexes == [1, slice(None, None, None)]
    assert df3.chunks[1].op.indexes == [1, slice(None, None, None)]
    assert df3.chunks[0].inputs[0].index == (0, 0)
    assert df3.chunks[0].inputs[0].shape == (2, 2)
    assert df3.chunks[1].inputs[0].index == (0, 1)
    assert df3.chunks[1].inputs[0].shape == (2, 1)

    # slice index
    df4 = df2.iloc[:, 2:4]
    df4 = tile(df4)
    assert isinstance(df4, DATAFRAME_TYPE)
    assert isinstance(df4.op, DataFrameIlocGetItem)
    assert df4.index_value.key == df2.index_value.key
    assert df4.shape == (3, 1)
    assert df4.chunk_shape == (2, 1)
    assert df4.chunks[0].shape == (2, 1)
    pd.testing.assert_index_equal(
        df4.chunks[0].columns_value.to_pandas(), df1.columns[2:3]
    )
    pd.testing.assert_series_equal(df4.chunks[0].dtypes, df1.dtypes[2:3])
    assert isinstance(df4.chunks[0].index_value.to_pandas(), type(df1.index))
    assert df4.chunks[1].shape == (1, 1)
    pd.testing.assert_index_equal(
        df4.chunks[1].columns_value.to_pandas(), df1.columns[2:3]
    )
    pd.testing.assert_series_equal(df4.chunks[1].dtypes, df1.dtypes[2:3])
    assert df4.chunks[0].index_value.key != df4.chunks[1].index_value.key
    assert isinstance(df4.chunks[1].index_value.to_pandas(), type(df1.index))
    assert df4.chunks[0].op.indexes == [
        slice(None, None, None),
        slice(None, None, None),
    ]
    assert df4.chunks[1].op.indexes == [
        slice(None, None, None),
        slice(None, None, None),
    ]
    assert df4.chunks[0].inputs[0].index == (0, 1)
    assert df4.chunks[0].inputs[0].shape == (2, 1)
    assert df4.chunks[1].inputs[0].index == (1, 1)
    assert df4.chunks[1].inputs[0].shape == (1, 1)

    # plain fancy index
    df5 = df2.iloc[[0], [0, 1, 2]]
    df5 = tile(df5)
    assert isinstance(df5, DATAFRAME_TYPE)
    assert isinstance(df5.op, DataFrameIlocGetItem)
    assert df5.shape == (1, 3)
    assert df5.chunk_shape == (1, 2)
    assert df5.chunks[0].shape == (1, 2)
    pd.testing.assert_index_equal(
        df5.chunks[0].columns_value.to_pandas(), df1.columns[:2]
    )
    pd.testing.assert_series_equal(df5.chunks[0].dtypes, df1.dtypes[:2])
    assert isinstance(df5.chunks[0].index_value.to_pandas(), type(df1.index))
    assert df5.chunks[1].shape == (1, 1)
    pd.testing.assert_index_equal(
        df5.chunks[1].columns_value.to_pandas(), df1.columns[2:]
    )
    pd.testing.assert_series_equal(df5.chunks[1].dtypes, df1.dtypes[2:])
    assert isinstance(df5.chunks[1].index_value.to_pandas(), type(df1.index))
    np.testing.assert_array_equal(df5.chunks[0].op.indexes[0], [0])
    np.testing.assert_array_equal(df5.chunks[0].op.indexes[1], [0, 1])
    np.testing.assert_array_equal(df5.chunks[1].op.indexes[0], [0])
    np.testing.assert_array_equal(df5.chunks[1].op.indexes[1], [0])
    assert df5.chunks[0].inputs[0].index == (0, 0)
    assert df5.chunks[0].inputs[0].shape == (2, 2)
    assert df5.chunks[1].inputs[0].index == (0, 1)
    assert df5.chunks[1].inputs[0].shape == (2, 1)

    # fancy index
    df6 = df2.iloc[[1, 2], [0, 1, 2]]
    df6 = tile(df6)
    assert isinstance(df6, DATAFRAME_TYPE)
    assert isinstance(df6.op, DataFrameIlocGetItem)
    assert df6.shape == (2, 3)
    assert df6.chunk_shape == (2, 2)
    assert df6.chunks[0].shape == (1, 2)
    assert df6.chunks[1].shape == (1, 1)
    assert df6.chunks[2].shape == (1, 2)
    assert df6.chunks[3].shape == (1, 1)
    np.testing.assert_array_equal(df6.chunks[0].op.indexes[0], [1])
    np.testing.assert_array_equal(df6.chunks[0].op.indexes[1], [0, 1])
    np.testing.assert_array_equal(df6.chunks[1].op.indexes[0], [1])
    np.testing.assert_array_equal(df6.chunks[1].op.indexes[1], [0])
    np.testing.assert_array_equal(df6.chunks[2].op.indexes[0], [0])
    np.testing.assert_array_equal(df6.chunks[2].op.indexes[1], [0, 1])
    np.testing.assert_array_equal(df6.chunks[3].op.indexes[0], [0])
    np.testing.assert_array_equal(df6.chunks[3].op.indexes[1], [0])
    assert df6.chunks[0].inputs[0].index == (0, 0)
    assert df6.chunks[0].inputs[0].shape == (2, 2)
    assert df6.chunks[1].inputs[0].index == (0, 1)
    assert df6.chunks[1].inputs[0].shape == (2, 1)
    assert df6.chunks[2].inputs[0].index == (1, 0)
    assert df6.chunks[2].inputs[0].shape == (1, 2)
    assert df6.chunks[3].inputs[0].index == (1, 1)
    assert df6.chunks[3].inputs[0].shape == (1, 1)

    # plain index
    df7 = df2.iloc[1, 2]
    df7 = tile(df7)
    assert isinstance(df7, TENSOR_TYPE)  # scalar
    assert isinstance(df7.op, DataFrameIlocGetItem)
    assert df7.shape == ()
    assert df7.chunk_shape == ()
    assert df7.chunks[0].dtype == df7.dtype
    assert df7.chunks[0].shape == ()
    assert df7.chunks[0].op.indexes == [1, 0]
    assert df7.chunks[0].inputs[0].index == (0, 1)
    assert df7.chunks[0].inputs[0].shape == (2, 1)

    # test Series iloc getitem

    # slice
    series = md.Series(pd.Series(np.arange(10)), chunk_size=3).iloc[4:8]
    series = tile(series)

    assert series.shape == (4,)

    assert len(series.chunks) == 2
    assert series.chunks[0].shape == (2,)
    assert series.chunks[0].index == (0,)
    assert series.chunks[0].op.indexes == [slice(1, 3, 1)]
    assert series.chunks[1].shape == (2,)
    assert series.chunks[1].op.indexes == [slice(0, 2, 1)]
    assert series.chunks[1].index == (1,)

    # fancy index
    series = md.Series(pd.Series(np.arange(10)), chunk_size=3).iloc[[2, 4, 8]]
    series = tile(series)

    assert series.shape == (3,)

    assert len(series.chunks) == 3
    assert series.chunks[0].shape == (1,)
    assert series.chunks[0].index == (0,)
    assert series.chunks[0].op.indexes[0] == [2]
    assert series.chunks[1].shape == (1,)
    assert series.chunks[1].op.indexes[0] == [1]
    assert series.chunks[1].index == (1,)
    assert series.chunks[2].shape == (1,)
    assert series.chunks[2].op.indexes[0] == [2]
    assert series.chunks[2].index == (2,)


def test_iloc_setitem():
    df1 = pd.DataFrame(
        [[1, 3, 3], [4, 2, 6], [7, 8, 9]],
        index=["a1", "a2", "a3"],
        columns=["x", "y", "z"],
    )
    df2 = md.DataFrame(df1, chunk_size=2)
    df2 = tile(df2)

    # plain index
    df3 = md.DataFrame(df1, chunk_size=2)
    df3.iloc[1] = 100
    df3 = tile(df3)
    assert isinstance(df3.op, DataFrameIlocSetItem)
    assert df3.chunk_shape == df2.chunk_shape
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df3.index_value.to_pandas()
    )
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), df3.columns_value.to_pandas()
    )
    for c1, c2 in zip(df2.chunks, df3.chunks):
        assert c1.shape == c2.shape
        pd.testing.assert_index_equal(
            c1.index_value.to_pandas(), c2.index_value.to_pandas()
        )
        pd.testing.assert_index_equal(
            c1.columns_value.to_pandas(), c2.columns_value.to_pandas()
        )
        if isinstance(c2.op, DataFrameIlocSetItem):
            assert c1.key == c2.inputs[0].key
        else:
            assert c1.key == c2.key
    assert df3.chunks[0].op.indexes == [1, slice(None, None, None)]
    assert df3.chunks[1].op.indexes == [1, slice(None, None, None)]

    # # slice index
    df4 = md.DataFrame(df1, chunk_size=2)
    df4.iloc[:, 2:4] = 1111
    df4 = tile(df4)
    assert isinstance(df4.op, DataFrameIlocSetItem)
    assert df4.chunk_shape == df2.chunk_shape
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df4.index_value.to_pandas()
    )
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), df4.columns_value.to_pandas()
    )
    for c1, c2 in zip(df2.chunks, df4.chunks):
        assert c1.shape == c2.shape
        pd.testing.assert_index_equal(
            c1.index_value.to_pandas(), c2.index_value.to_pandas()
        )
        pd.testing.assert_index_equal(
            c1.columns_value.to_pandas(), c2.columns_value.to_pandas()
        )
        if isinstance(c2.op, DataFrameIlocSetItem):
            assert c1.key == c2.inputs[0].key
        else:
            assert c1.key == c2.key
    assert df4.chunks[1].op.indexes == [
        slice(None, None, None),
        slice(None, None, None),
    ]
    assert df4.chunks[3].op.indexes == [
        slice(None, None, None),
        slice(None, None, None),
    ]

    # plain fancy index
    df5 = md.DataFrame(df1, chunk_size=2)
    df5.iloc[[0], [0, 1, 2]] = 2222
    df5 = tile(df5)
    assert isinstance(df5.op, DataFrameIlocSetItem)
    assert df5.chunk_shape == df2.chunk_shape
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df5.index_value.to_pandas()
    )
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), df5.columns_value.to_pandas()
    )
    for c1, c2 in zip(df2.chunks, df5.chunks):
        assert c1.shape == c2.shape
        pd.testing.assert_index_equal(
            c1.index_value.to_pandas(), c2.index_value.to_pandas()
        )
        pd.testing.assert_index_equal(
            c1.columns_value.to_pandas(), c2.columns_value.to_pandas()
        )
        if isinstance(c2.op, DataFrameIlocSetItem):
            assert c1.key == c2.inputs[0].key
        else:
            assert c1.key == c2.key
    np.testing.assert_array_equal(df5.chunks[0].op.indexes[0], [0])
    np.testing.assert_array_equal(df5.chunks[0].op.indexes[1], [0, 1])
    np.testing.assert_array_equal(df5.chunks[1].op.indexes[0], [0])
    np.testing.assert_array_equal(df5.chunks[1].op.indexes[1], [0])

    # fancy index
    df6 = md.DataFrame(df1, chunk_size=2)
    df6.iloc[[1, 2], [0, 1, 2]] = 3333
    df6 = tile(df6)
    assert isinstance(df6.op, DataFrameIlocSetItem)
    assert df6.chunk_shape == df2.chunk_shape
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df6.index_value.to_pandas()
    )
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), df6.columns_value.to_pandas()
    )
    for c1, c2 in zip(df2.chunks, df6.chunks):
        assert c1.shape == c2.shape
        pd.testing.assert_index_equal(
            c1.index_value.to_pandas(), c2.index_value.to_pandas()
        )
        pd.testing.assert_index_equal(
            c1.columns_value.to_pandas(), c2.columns_value.to_pandas()
        )
        if isinstance(c2.op, DataFrameIlocSetItem):
            assert c1.key == c2.inputs[0].key
        else:
            assert c1.key == c2.key
    np.testing.assert_array_equal(df6.chunks[0].op.indexes[0], [1])
    np.testing.assert_array_equal(df6.chunks[0].op.indexes[1], [0, 1])
    np.testing.assert_array_equal(df6.chunks[1].op.indexes[0], [1])
    np.testing.assert_array_equal(df6.chunks[1].op.indexes[1], [0])
    np.testing.assert_array_equal(df6.chunks[2].op.indexes[0], [0])
    np.testing.assert_array_equal(df6.chunks[2].op.indexes[1], [0, 1])
    np.testing.assert_array_equal(df6.chunks[3].op.indexes[0], [0])
    np.testing.assert_array_equal(df6.chunks[3].op.indexes[1], [0])

    # plain index
    df7 = md.DataFrame(df1, chunk_size=2)
    df7.iloc[1, 2] = 4444
    df7 = tile(df7)
    assert isinstance(df7.op, DataFrameIlocSetItem)
    assert df7.chunk_shape == df2.chunk_shape
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df7.index_value.to_pandas()
    )
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), df7.columns_value.to_pandas()
    )
    for c1, c2 in zip(df2.chunks, df7.chunks):
        assert c1.shape == c2.shape
        pd.testing.assert_index_equal(
            c1.index_value.to_pandas(), c2.index_value.to_pandas()
        )
        pd.testing.assert_index_equal(
            c1.columns_value.to_pandas(), c2.columns_value.to_pandas()
        )
        if isinstance(c2.op, DataFrameIlocSetItem):
            assert c1.key == c2.inputs[0].key
        else:
            assert c1.key == c2.key
    assert df7.chunks[1].op.indexes == [1, 0]

    # test Series

    # slice
    series = md.Series(pd.Series(np.arange(10)), chunk_size=3)
    series.iloc[:4] = 2
    series = tile(series)

    assert series.shape == (10,)
    assert len(series.chunks) == 4

    assert series.chunks[0].op.indexes == [
        slice(None, None, None),
    ]
    assert series.chunks[0].op.value == 2
    assert series.chunks[1].op.indexes == [
        slice(0, 1, 1),
    ]
    assert series.chunks[1].op.value == 2

    raw = pd.DataFrame(
        np.random.rand(9, 2),
        index=["a1", "a2", "a3"] * 3,
        columns=["x", "y"],
    )
    df = md.DataFrame(raw, chunk_size=4)
    iloc_df = df.iloc[:, 1:]
    tiled_df, tiled_iloc_df = tile(df, iloc_df)
    # for full slice, index_value should be same as input chunk
    for loc_chunk, chunk in zip(tiled_iloc_df.chunks, tiled_df.chunks):
        assert loc_chunk.index_value.key == chunk.index_value.key

    # fancy index
    series = md.Series(pd.Series(np.arange(10)), chunk_size=3)
    series.iloc[[2, 4, 9]] = 3
    series = tile(series)

    assert series.shape == (10,)

    assert len(series.chunks) == 4
    assert series.chunks[0].index == (0,)
    assert series.chunks[0].op.indexes[0].tolist() == [2]
    assert series.chunks[0].op.value == 3
    assert series.chunks[1].index == (1,)
    assert series.chunks[1].op.indexes[0].tolist() == [1]
    assert series.chunks[1].op.value == 3
    assert series.chunks[3].index == (3,)
    assert series.chunks[3].op.indexes[0].tolist() == [0]
    assert series.chunks[3].op.value == 3


def test_dataframe_loc():
    raw = pd.DataFrame(
        [[1, 3, 3], [4, 2, 6], [7, 8, 9]],
        index=["a1", "a2", "a3"],
        columns=["x", "y", "z"],
    )
    df = md.DataFrame(raw, chunk_size=2)
    raw2 = raw.copy()
    raw2.reset_index(inplace=True, drop=True)
    df3 = md.DataFrame(raw2, chunk_size=2)
    s = pd.Series([1, 3, 5], index=["a1", "a2", "a3"])
    series = md.Series(s, chunk_size=2)

    # test return scalar
    df2 = df.loc["a1", "z"]
    assert isinstance(df2, Tensor)
    assert df2.shape == ()
    assert df2.dtype == raw["z"].dtype

    df2 = tile(df2)
    assert len(df2.chunks) == 1
    assert isinstance(df2.chunks[0], TENSOR_CHUNK_TYPE)

    # test return series for index axis
    df2 = df.loc[:, "y"]
    assert isinstance(df2, Series)
    assert df2.shape == (3,)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df.index_value.to_pandas()
    )
    assert df2.name == "y"
    assert df2.index_value.key == df.index_value.key

    df2 = tile(df2)
    assert len(df2.chunks) == 2
    for c in df2.chunks:
        assert isinstance(c, SERIES_CHUNK_TYPE)
        assert isinstance(c.index_value.to_pandas(), type(raw.index))
        assert c.name == "y"
        assert c.dtype == raw["y"].dtype

    # test return series for column axis
    df2 = df.loc["a2", :]
    assert isinstance(df2, Series)
    assert df2.shape == (3,)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df.columns_value.to_pandas()
    )
    assert df2.name == "a2"

    df2 = tile(df2)
    assert len(df2.chunks) == 2
    for c in df2.chunks:
        assert isinstance(c, SERIES_CHUNK_TYPE)
        assert isinstance(c.index_value.to_pandas(), type(raw.columns))
        assert c.name == "a2"
        assert c.dtype == raw.loc["a2"].dtype

    # test slice
    df2 = df.loc["a2":"a3", "y":"z"]
    assert isinstance(df2, DataFrame)
    assert df2.shape == (np.nan, 2)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df.index_value.to_pandas()
    )
    assert df2.index_value.key != df.index_value.key
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), raw.loc[:, "y":"z"].columns
    )
    pd.testing.assert_series_equal(df2.dtypes, raw.loc[:, "y":"z"].dtypes)

    # test fancy index on index axis
    df2 = df.loc[["a3", "a2"], [True, False, True]]
    assert isinstance(df2, DataFrame)
    assert df2.shape == (2, 2)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df.index_value.to_pandas()
    )
    assert df2.index_value.key != df.index_value.key
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), raw.loc[:, [True, False, True]].columns
    )
    pd.testing.assert_series_equal(df2.dtypes, raw.loc[:, [True, False, True]].dtypes)

    # test fancy index which is md.Series on index axis
    df2 = df.loc[md.Series(["a3", "a2"]), [True, False, True]]
    assert isinstance(df2, DataFrame)
    assert df2.shape == (2, 2)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df.index_value.to_pandas()
    )
    assert df2.index_value.key != df.index_value.key
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), raw.loc[:, [True, False, True]].columns
    )
    pd.testing.assert_series_equal(df2.dtypes, raw.loc[:, [True, False, True]].dtypes)

    # test fancy index on columns axis
    df2 = df.loc[[True, False, True], ["z", "x", "y"]]
    assert isinstance(df2, DataFrame)
    assert df2.shape == (2, 3)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df.index_value.to_pandas()
    )
    assert df2.index_value.key != df.index_value.key
    pd.testing.assert_index_equal(
        df2.columns_value.to_pandas(), raw.loc[:, ["z", "x", "y"]].columns
    )
    pd.testing.assert_series_equal(df2.dtypes, raw.loc[:, ["z", "x", "y"]].dtypes)

    df2 = tile(df2)
    assert len(df2.chunks) == 2
    for c in df2.chunks:
        assert isinstance(c, DATAFRAME_CHUNK_TYPE)
        pd.testing.assert_index_equal(
            c.index_value.to_pandas(), df.index_value.to_pandas()
        )
        assert c.index_value.key != df.index_value.key
        pd.testing.assert_index_equal(
            c.columns_value.to_pandas(), raw.loc[:, ["z", "x", "y"]].columns
        )
        pd.testing.assert_series_equal(c.dtypes, raw.loc[:, ["z", "x", "y"]].dtypes)

    df2 = df.loc[md.Series([True, False, True])]
    assert isinstance(df2, DataFrame)
    assert df2.shape == (np.nan, 3)
    pd.testing.assert_index_equal(
        df2.index_value.to_pandas(), df.index_value.to_pandas()
    )
    assert df2.index_value.key != df.index_value.key
    pd.testing.assert_index_equal(df2.columns_value.to_pandas(), raw.columns)
    pd.testing.assert_series_equal(df2.dtypes, raw.dtypes)

    df2 = df3.loc[md.Series([True, False, True])]
    assert isinstance(df2, DataFrame)
    assert df2.shape == (np.nan, 3)
    assert isinstance(
        df2.index_value.to_pandas(), type(raw.loc[[True, False, True]].index)
    )
    assert df2.index_value.key != df3.index_value.key
    pd.testing.assert_index_equal(df2.columns_value.to_pandas(), raw.columns)
    pd.testing.assert_series_equal(df2.dtypes, raw.dtypes)

    df2 = df3.loc[md.Series([2, 1])]
    assert isinstance(df2, DataFrame)
    assert df2.shape == (2, 3)
    assert isinstance(df2.index_value.to_pandas(), type(raw2.loc[[2, 1]].index))
    assert df2.index_value.key != df3.index_value.key
    pd.testing.assert_index_equal(df2.columns_value.to_pandas(), raw.columns)
    pd.testing.assert_series_equal(df2.dtypes, raw.dtypes)

    series2 = series.loc["a2"]
    assert isinstance(series2, Tensor)
    assert series2.shape == ()
    assert series2.dtype == s.dtype

    series2 = series.loc[["a2", "a3"]]
    assert isinstance(series2, Series)
    assert series2.shape == (2,)
    assert series2.dtype == s.dtype
    assert series2.name == s.name

    with pytest.raises(IndexingError):
        _ = df.loc["a1", "z", ...]

    with pytest.raises(NotImplementedError):
        _ = df.loc[:, md.Series([True, False, True])]

    with pytest.raises(KeyError):
        _ = df.loc[:, ["non_exist"]]

    # test loc chunk's index_value
    raw = pd.DataFrame(
        np.random.rand(9, 2),
        index=["a1", "a2", "a3"] * 3,
        columns=["x", "y"],
    )
    df = md.DataFrame(raw, chunk_size=4)
    loc_df = df.loc[:, ["x"]]
    tiled_df, tiled_loc_df = tile(df, loc_df)
    # for full slice, index_value should be same as input chunk
    for loc_chunk, chunk in zip(tiled_loc_df.chunks, tiled_df.chunks):
        assert loc_chunk.index_value.key == chunk.index_value.key


def test_loc_use_iloc():
    raw = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]], columns=["x", "y", "z"])
    df = md.DataFrame(raw, chunk_size=2)

    assert isinstance(df.loc[:3].op, DataFrameIlocGetItem)
    assert isinstance(df.loc[1:3].op, DataFrameIlocGetItem)
    assert isinstance(df.loc[1].op, DataFrameIlocGetItem)
    # negative
    assert isinstance(df.loc[:-3].op, DataFrameLocGetItem)
    with pytest.raises(KeyError):
        _ = df.loc[-3]
    # index 1 not None
    assert isinstance(df.loc[:3, :"y"].op, DataFrameLocGetItem)
    # index 1 not slice
    assert isinstance(df.loc[:3, [True, False, True]].op, DataFrameLocGetItem)
    assert isinstance(df.loc[[True, False, True]].op, DataFrameLocGetItem)

    raw2 = raw.copy()
    raw2.index = pd.RangeIndex(1, 4)
    df2 = md.DataFrame(raw2, chunk_size=2)

    assert isinstance(df2.loc[:3].op, DataFrameLocGetItem)
    assert isinstance(df2.loc["a3":].op, DataFrameLocGetItem)

    raw2 = raw.copy()
    raw2.index = [f"a{i}" for i in range(3)]
    df2 = md.DataFrame(raw2, chunk_size=2)

    assert isinstance(df2.loc[:3].op, DataFrameLocGetItem)


def test_dataframe_getitem():
    data = pd.DataFrame(np.random.rand(10, 5), columns=["c1", "c2", "c3", "c4", "c5"])
    df = md.DataFrame(data, chunk_size=2)

    series = df["c3"]
    assert isinstance(series, Series)
    assert series.shape == (10,)
    assert series.name == "c3"
    assert series.dtype == data["c3"].dtype
    assert series.index_value == df.index_value

    series = tile(series)
    assert isinstance(series, SERIES_TYPE)
    assert all(not i.is_coarse() for i in series.inputs) is True
    assert series.nsplits == ((2, 2, 2, 2, 2),)
    assert len(series.chunks) == 5
    for i, c in enumerate(series.chunks):
        assert isinstance(c, SERIES_CHUNK_TYPE)
        assert c.index == (i,)
        assert c.shape == (2,)

    df1 = df[["c1", "c2", "c3"]]
    assert isinstance(df1, DataFrame)
    assert df1.shape == (10, 3)
    assert df1.index_value == df.index_value
    pd.testing.assert_index_equal(
        df1.columns_value.to_pandas(), data[["c1", "c2", "c3"]].columns
    )
    pd.testing.assert_series_equal(df1.dtypes, data[["c1", "c2", "c3"]].dtypes)

    df1 = tile(df1)
    assert df1.nsplits == ((2, 2, 2, 2, 2), (2, 1))
    assert len(df1.chunks) == 10
    for i, c in enumerate(df1.chunks[slice(0, 10, 2)]):
        assert isinstance(c, DATAFRAME_CHUNK_TYPE)
        assert c.index == (i, 0)
        assert c.shape == (2, 2)
    for i, c in enumerate(df1.chunks[slice(1, 10, 2)]):
        assert isinstance(c, DATAFRAME_CHUNK_TYPE)
        assert c.index == (i, 1)
        assert c.shape == (2, 1)


def test_dataframe_getitem_bool():
    data = pd.DataFrame(
        np.random.rand(10, 5),
        columns=["c1", "c2", "c3", "c4", "c5"],
        index=pd.RangeIndex(10, name="i"),
    )
    df = md.DataFrame(data, chunk_size=2)

    mask_data1 = data.c1 > 0.5
    mask_data2 = data.c1 < 0.5
    mask1 = md.Series(mask_data1, chunk_size=2)
    mask2 = md.Series(mask_data2, chunk_size=2)

    r1 = df[mask1]
    r2 = df[mask2]
    r3 = df[mask1]

    assert r1.index_value.key != df.index_value.key
    assert r1.index_value.key != mask1.index_value.key
    assert r1.columns_value.key == df.columns_value.key
    assert r1.columns_value is df.columns_value
    assert r1.index_value.name == "i"

    assert r1.index_value.key != r2.index_value.key
    assert r1.columns_value.key == r2.columns_value.key
    assert r1.columns_value is r2.columns_value

    assert r1.index_value.key == r3.index_value.key
    assert r1.columns_value.key == r3.columns_value.key
    assert r1.columns_value is r3.columns_value


def test_series_getitem():
    data = pd.Series(np.random.rand(10), name="a")
    series = md.Series(data, chunk_size=3)

    result1 = series[2]
    assert result1.shape == ()

    result1 = tile(result1)
    assert result1.nsplits == ()
    assert len(result1.chunks) == 1
    assert isinstance(result1.chunks[0], TENSOR_CHUNK_TYPE)
    assert result1.chunks[0].shape == ()
    assert result1.chunks[0].dtype == data.dtype

    result2 = series[[4, 5, 1, 2, 3]]
    assert result2.shape == (5,)

    result2 = tile(result2)
    assert result2.nsplits == ((2, 2, 1),)
    assert len(result2.chunks) == 3
    assert result2.chunks[0].op.labels == [4, 5]
    assert result2.chunks[1].op.labels == [1, 2]
    assert result2.chunks[2].op.labels == [3]

    data = pd.Series(np.random.rand(10), index=["i" + str(i) for i in range(10)])
    series = md.Series(data, chunk_size=3)

    result1 = series["i2"]
    assert result1.shape == ()

    result1 = tile(result1)
    assert result1.nsplits == ()
    assert result1.chunks[0].dtype == data.dtype
    assert result1.chunks[0].op.labels == "i2"

    result2 = series[["i2", "i4"]]
    assert result2.shape == (2,)

    result2 = tile(result2)
    assert result2.nsplits == ((2,),)
    assert result2.chunks[0].dtype == data.dtype
    assert result2.chunks[0].op.labels == ["i2", "i4"]


def test_setitem():
    data = pd.DataFrame(np.random.rand(10, 2), columns=["c1", "c2"])
    df = md.DataFrame(data, chunk_size=4)

    df["new"] = 1
    assert df.shape == (10, 3)
    pd.testing.assert_series_equal(df.inputs[0].dtypes, data.dtypes)

    tiled = tile(df)
    assert tiled.chunks[0].shape == (4, 3)
    pd.testing.assert_series_equal(tiled.inputs[0].dtypes, data.dtypes)
    assert tiled.chunks[1].shape == (4, 3)
    pd.testing.assert_series_equal(tiled.inputs[0].dtypes, data.dtypes)
    assert tiled.chunks[2].shape == (2, 3)
    pd.testing.assert_series_equal(tiled.inputs[0].dtypes, data.dtypes)

    for c in tiled.chunks:
        pd.testing.assert_series_equal(c.inputs[0].dtypes, data.dtypes)


def test_reset_index():
    data = pd.DataFrame(
        [("bird", 389.0), ("bird", 24.0), ("mammal", 80.5), ("mammal", np.nan)],
        index=["falcon", "parrot", "lion", "monkey"],
        columns=("class", "max_speed"),
    )
    df = md.DataFrame(data, chunk_size=2).reset_index()
    r = data.reset_index()

    assert df.shape == (4, 3)
    pd.testing.assert_series_equal(df.dtypes, r.dtypes)
    pd.testing.assert_index_equal(df.columns_value.to_pandas(), r.columns)

    df2 = tile(df)

    assert len(df2.chunks) == 2
    assert df2.chunks[0].shape == (2, 3)
    pd.testing.assert_index_equal(
        df2.chunks[0].index_value.to_pandas(), pd.RangeIndex(2)
    )
    pd.testing.assert_series_equal(df2.chunks[0].dtypes, r.dtypes)
    assert df2.chunks[1].shape == (2, 3)
    pd.testing.assert_index_equal(
        df2.chunks[1].index_value.to_pandas(), pd.RangeIndex(2, 4)
    )
    pd.testing.assert_series_equal(df2.chunks[1].dtypes, r.dtypes)

    df = md.DataFrame(data, chunk_size=1).reset_index(drop=True)
    r = data.reset_index(drop=True)

    assert df.shape == (4, 2)
    pd.testing.assert_series_equal(df.dtypes, r.dtypes)

    df2 = tile(df)

    assert len(df2.chunks) == 8

    for c in df2.chunks:
        assert c.shape == (1, 1)
        pd.testing.assert_index_equal(
            c.index_value.to_pandas(), pd.RangeIndex(c.index[0], c.index[0] + 1)
        )
        pd.testing.assert_series_equal(c.dtypes, r.dtypes[c.index[1] : c.index[1] + 1])

    # test Series
    series_data = pd.Series(
        [1, 2, 3, 4], name="foo", index=pd.Index(["a", "b", "c", "d"], name="idx")
    )
    s = md.Series(series_data, chunk_size=2).reset_index()
    r = series_data.reset_index()

    assert s.shape == (4, 2)
    pd.testing.assert_series_equal(s.dtypes, r.dtypes)

    s2 = tile(s)
    assert len(s2.chunks) == 2
    assert s2.chunks[0].shape == (2, 2)
    pd.testing.assert_index_equal(
        s2.chunks[0].index_value.to_pandas(), pd.RangeIndex(2)
    )
    assert s2.chunks[1].shape == (2, 2)
    pd.testing.assert_index_equal(
        s2.chunks[1].index_value.to_pandas(), pd.RangeIndex(2, 4)
    )

    with pytest.raises(TypeError):
        md.Series(series_data, chunk_size=2).reset_index(inplace=True)


def test_head_tail_optimize():
    raw = pd.DataFrame(np.random.rand(4, 3))

    df = md.DataFrame(raw, chunk_size=2)

    # no nan chunk shape
    assert (
        HeadTailOptimizedOperandMixin._need_tile_head_tail(tile(df).head(2).op) is False
    )

    df2 = tile(df[df[0] < 0.5])
    # chunk shape on axis 1 greater than 1
    assert HeadTailOptimizedOperandMixin._need_tile_head_tail(df2.head(2).op) is False

    df = md.DataFrame(raw, chunk_size=(2, 3))
    df2 = tile(df[df[0] < 0.5])
    # not slice
    assert HeadTailOptimizedOperandMixin._need_tile_head_tail(df2.iloc[2].op) is False
    # step not None
    assert (
        HeadTailOptimizedOperandMixin._need_tile_head_tail(df2.iloc[:2:2].op) is False
    )
    # not head or tail
    assert HeadTailOptimizedOperandMixin._need_tile_head_tail(df2.iloc[1:3].op) is False
    # slice 1 is not slice(None)
    assert (
        HeadTailOptimizedOperandMixin._need_tile_head_tail(df2.iloc[:3, :2].op) is False
    )


def test_reindex():
    raw = pd.DataFrame(np.random.rand(4, 3))

    df = md.DataFrame(raw, chunk_size=2)

    with pytest.raises(TypeError):
        df.reindex(unknown_arg=1)

    with pytest.raises(ValueError):
        df.reindex([1, 2], fill_value=mt.tensor([1, 2]))
