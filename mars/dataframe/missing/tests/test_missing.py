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

import random

import numpy as np
import pandas as pd
import pytest

from .... import dataframe as md
from ....core import tile
from ....core.operand import OperandStage
from ....utils import pd_release_version

_drop_na_enable_no_default = pd_release_version[:2] >= (1, 5)


def test_fill_na():
    df_raw = pd.DataFrame(np.nan, index=range(0, 20), columns=list("ABCDEFGHIJ"))
    for _ in range(20):
        df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)
    value_df_raw = pd.DataFrame(
        np.random.randint(0, 100, (10, 7)).astype(np.float32), columns=list("ABCDEFG")
    )
    series_raw = pd.Series(np.nan, index=range(20))
    for _ in range(3):
        series_raw.iloc[random.randint(0, 19)] = random.randint(0, 99)
    value_series_raw = pd.Series(
        np.random.randint(0, 100, (10,)).astype(np.float32), index=list("ABCDEFGHIJ")
    )

    df = md.DataFrame(df_raw)
    series = md.Series(series_raw)

    # when nothing supplied, raise
    with pytest.raises(ValueError):
        df.fillna()
    # when both values and methods supplied, raises
    with pytest.raises(ValueError):
        df.fillna(value=1, method="ffill")
    # when call on series, cannot supply DataFrames
    with pytest.raises(ValueError):
        series.fillna(value=df)
    with pytest.raises(ValueError):
        series.fillna(value=df_raw)
    with pytest.raises(NotImplementedError):
        series.fillna(value=series_raw, downcast="infer")
    with pytest.raises(NotImplementedError):
        series.ffill(limit=1)

    df2 = tile(df.fillna(value_series_raw))
    assert len(df2.chunks) == 1
    assert df2.chunks[0].shape == df2.shape
    assert df2.chunks[0].op.stage is None

    series2 = tile(series.fillna(value_series_raw))
    assert len(series2.chunks) == 1
    assert series2.chunks[0].shape == series2.shape
    assert series2.chunks[0].op.stage is None

    df = md.DataFrame(df_raw, chunk_size=5)
    df2 = tile(df.fillna(value_series_raw))
    assert len(df2.chunks) == 8
    assert df2.chunks[0].shape == (5, 5)
    assert df2.chunks[0].op.stage is None

    series = md.Series(series_raw, chunk_size=5)
    series2 = tile(series.fillna(value_series_raw))
    assert len(series2.chunks) == 4
    assert series2.chunks[0].shape == (5,)
    assert series2.chunks[0].op.stage is None

    df2 = tile(df.ffill(axis="columns"))
    assert len(df2.chunks) == 8
    assert df2.chunks[0].shape == (5, 5)
    assert df2.chunks[0].op.axis == 1
    assert df2.chunks[0].op.stage == OperandStage.combine
    assert df2.chunks[0].op.method == "ffill"
    assert df2.chunks[0].op.limit is None

    series2 = tile(series.bfill())
    assert len(series2.chunks) == 4
    assert series2.chunks[0].shape == (5,)
    assert series2.chunks[0].op.stage == OperandStage.combine
    assert series2.chunks[0].op.method == "bfill"
    assert series2.chunks[0].op.limit is None

    value_df = md.DataFrame(value_df_raw, chunk_size=7)
    value_series = md.Series(value_series_raw, chunk_size=7)

    df2 = tile(df.fillna(value_df))
    assert df2.shape == df.shape
    assert df2.chunks[0].op.stage is None

    df2 = tile(df.fillna(value_series))
    assert df2.shape == df.shape
    assert df2.chunks[0].op.stage is None

    value_series_raw.index = list(range(10))
    value_series = md.Series(value_series_raw)
    series2 = tile(series.fillna(value_series))
    assert series2.shape == series.shape
    assert series2.chunks[0].op.stage is None


def test_drop_na():
    # dataframe cases
    df_raw = pd.DataFrame(np.nan, index=range(0, 20), columns=list("ABCDEFGHIJ"))
    for _ in range(30):
        df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)
    for rowid in range(random.randint(1, 5)):
        row = random.randint(0, 19)
        for idx in range(0, 10):
            df_raw.iloc[row, idx] = random.randint(0, 99)

    # not supporting drop with axis=1
    with pytest.raises(NotImplementedError):
        md.DataFrame(df_raw).dropna(axis=1)

    if _drop_na_enable_no_default:
        with pytest.raises(TypeError):
            md.DataFrame(df_raw).dropna(how="any", thresh=0)

    # only one chunk in columns, can run dropna directly
    r = tile(md.DataFrame(df_raw, chunk_size=(4, 10)).dropna())
    assert r.shape == (np.nan, 10)
    assert r.nsplits == ((np.nan,) * 5, (10,))
    for c in r.chunks:
        assert isinstance(c.op, type(r.op))
        assert len(c.inputs) == 1
        assert len(c.inputs[0].inputs) == 0
        assert c.shape == (np.nan, 10)

    # multiple chunks in columns, count() will be called first
    r = tile(md.DataFrame(df_raw, chunk_size=4).dropna())
    assert r.shape == (np.nan, 10)
    assert r.nsplits == ((np.nan,) * 5, (4, 4, 2))
    for c in r.chunks:
        assert isinstance(c.op, type(r.op))
        assert len(c.inputs) == 2
        assert len(c.inputs[0].inputs) == 0
        assert c.inputs[1].op.stage == OperandStage.agg
        assert np.isnan(c.shape[0])

    # series cases
    series_raw = pd.Series(np.nan, index=range(20))
    for _ in range(10):
        series_raw.iloc[random.randint(0, 19)] = random.randint(0, 99)

    r = tile(md.Series(series_raw, chunk_size=4).dropna())
    assert r.shape == (np.nan,)
    assert r.nsplits == ((np.nan,) * 5,)
    for c in r.chunks:
        assert isinstance(c.op, type(r.op))
        assert len(c.inputs) == 1
        assert len(c.inputs[0].inputs) == 0
        assert c.shape == (np.nan,)


def test_replace():
    # dataframe cases
    df_raw = pd.DataFrame(-1, index=range(0, 20), columns=list("ABCDEFGHIJ"))
    for _ in range(30):
        df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)
    for rowid in range(random.randint(1, 5)):
        row = random.randint(0, 19)
        for idx in range(0, 10):
            df_raw.iloc[row, idx] = random.randint(0, 99)

    # not supporting fill with limit
    df = md.DataFrame(df_raw, chunk_size=4)
    with pytest.raises(NotImplementedError):
        df.replace(-1, method="ffill", limit=5)

    r = tile(df.replace(-1, method="ffill"))
    assert len(r.chunks) == 15
    assert r.chunks[0].shape == (4, 4)
    assert r.chunks[0].op.stage == OperandStage.combine
    assert r.chunks[0].op.method == "ffill"
    assert r.chunks[0].op.limit is None
    assert r.chunks[-1].inputs[-1].shape == (1, 2)
    assert r.chunks[-1].inputs[-1].op.stage == OperandStage.map
    assert r.chunks[-1].inputs[-1].op.method == "ffill"
    assert r.chunks[-1].inputs[-1].op.limit is None

    r = tile(df.replace(-1, 99))
    assert len(r.chunks) == 15
    assert r.chunks[0].shape == (4, 4)
    assert r.chunks[0].op.stage is None
    assert r.chunks[0].op.limit is None

    # series cases
    series_raw = pd.Series(-1, index=range(20))
    for _ in range(10):
        series_raw.iloc[random.randint(0, 19)] = random.randint(0, 99)
    series = md.Series(series_raw, chunk_size=4)

    r = tile(series.replace(-1, method="ffill"))
    assert len(r.chunks) == 5
    assert r.chunks[0].shape == (4,)
    assert r.chunks[0].op.stage == OperandStage.combine
    assert r.chunks[0].op.method == "ffill"
    assert r.chunks[0].op.limit is None
    assert r.chunks[-1].inputs[-1].shape == (1,)
    assert r.chunks[-1].inputs[-1].op.stage == OperandStage.map
    assert r.chunks[-1].inputs[-1].op.method == "ffill"
    assert r.chunks[-1].inputs[-1].op.limit is None

    r = tile(series.replace(-1, 99))
    assert len(r.chunks) == 5
    assert r.chunks[0].shape == (4,)
    assert r.chunks[0].op.stage is None
    assert r.chunks[0].op.limit is None
