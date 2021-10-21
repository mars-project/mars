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

from ....core import tile
from ....core.operand import OperandStage
from ...indexing.getitem import DataFrameIndex
from ...initializer import DataFrame
from ..sort_index import sort_index, DataFrameSortIndex
from ..sort_values import dataframe_sort_values, DataFrameSortValues


def test_sort_values():
    raw = pd.DataFrame(
        {
            "a": np.random.rand(10),
            "b": np.random.randint(1000, size=10),
            "c": np.random.rand(10),
            "d": [np.random.bytes(10) for _ in range(10)],
            "e": [pd.Timestamp(f"201{i}") for i in range(10)],
            "f": [pd.Timedelta(f"{i} days") for i in range(10)],
        },
    )
    df = DataFrame(raw)
    sorted_df = dataframe_sort_values(df, by="c")

    assert sorted_df.shape == raw.shape
    assert isinstance(sorted_df.op, DataFrameSortValues)

    tiled = tile(sorted_df)

    assert len(tiled.chunks) == 1
    assert isinstance(tiled.chunks[0].op, DataFrameSortValues)

    df = DataFrame(raw, chunk_size=6)
    sorted_df = dataframe_sort_values(df, by="c")

    assert sorted_df.shape == raw.shape
    assert isinstance(sorted_df.op, DataFrameSortValues)

    tiled = tile(sorted_df)

    assert len(tiled.chunks) == 2
    assert tiled.chunks[0].op.stage == OperandStage.reduce

    df = DataFrame(raw, chunk_size=3)
    sorted_df = dataframe_sort_values(df, by=["a", "c"])

    assert sorted_df.shape == raw.shape
    assert isinstance(sorted_df.op, DataFrameSortValues)

    tiled = tile(sorted_df)

    assert len(tiled.chunks) == 3
    assert tiled.chunks[0].op.stage == OperandStage.reduce
    pd.testing.assert_series_equal(tiled.chunks[0].dtypes, raw.dtypes)
    assert tiled.chunks[1].op.stage == OperandStage.reduce
    pd.testing.assert_series_equal(tiled.chunks[1].dtypes, raw.dtypes)
    assert tiled.chunks[2].op.stage == OperandStage.reduce
    pd.testing.assert_series_equal(tiled.chunks[2].dtypes, raw.dtypes)


def test_sort_index():
    raw = pd.DataFrame(
        np.random.rand(10, 10), columns=np.random.rand(10), index=np.random.rand(10)
    )
    df = DataFrame(raw)
    sorted_df = sort_index(df)

    assert sorted_df.shape == raw.shape
    assert isinstance(sorted_df.op, DataFrameSortIndex)

    tiled = tile(sorted_df)

    assert len(tiled.chunks) == 1
    assert isinstance(tiled.chunks[0].op, DataFrameSortIndex)

    df = DataFrame(raw, chunk_size=6)
    sorted_df = sort_index(df)

    assert sorted_df.shape == raw.shape
    assert isinstance(sorted_df.op, DataFrameSortIndex)

    tiled = tile(sorted_df)

    assert len(tiled.chunks) == 2
    assert tiled.chunks[0].op.stage == OperandStage.reduce

    df = DataFrame(raw, chunk_size=3)
    sorted_df = sort_index(df)

    assert sorted_df.shape == raw.shape
    assert isinstance(sorted_df.op, DataFrameSortIndex)

    tiled = tile(sorted_df)

    assert len(tiled.chunks) == 3
    assert tiled.chunks[0].op.stage == OperandStage.reduce
    assert tiled.chunks[1].op.stage == OperandStage.reduce
    assert tiled.chunks[2].op.stage == OperandStage.reduce

    # support on axis 1
    df = DataFrame(raw, chunk_size=4)
    sorted_df = sort_index(df, axis=1)

    assert sorted_df.shape == raw.shape
    assert isinstance(sorted_df.op, DataFrameSortIndex)

    tiled = tile(sorted_df)

    assert all(isinstance(c.op, DataFrameIndex) for c in tiled.chunks) is True
