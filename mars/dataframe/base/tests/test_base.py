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

from .... import opcodes
from ....config import options, option_context
from ....core import OutputType, tile
from ....core.operand import OperandStage
from ....tensor.core import TENSOR_TYPE
from ... import eval as mars_eval, cut, get_dummies, to_numeric
from ...core import (
    DATAFRAME_TYPE,
    SERIES_TYPE,
    SERIES_CHUNK_TYPE,
    INDEX_TYPE,
    CATEGORICAL_TYPE,
    CATEGORICAL_CHUNK_TYPE,
)
from ...datasource.dataframe import from_pandas as from_pandas_df
from ...datasource.series import from_pandas as from_pandas_series
from ...datasource.index import from_pandas as from_pandas_index
from .. import to_gpu, to_cpu, astype


def test_to_gpu():
    # test dataframe
    data = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.random.randint(-100, 100, size=(10,)),
        columns=[np.random.bytes(10) for _ in range(10)],
    )
    df = from_pandas_df(data)
    cdf = to_gpu(df)

    assert df.index_value == cdf.index_value
    assert df.columns_value == cdf.columns_value
    assert cdf.op.gpu is True
    pd.testing.assert_series_equal(df.dtypes, cdf.dtypes)

    df, cdf = tile(df, cdf)

    assert df.nsplits == cdf.nsplits
    assert df.chunks[0].index_value == cdf.chunks[0].index_value
    assert df.chunks[0].columns_value == cdf.chunks[0].columns_value
    assert cdf.chunks[0].op.gpu is True
    pd.testing.assert_series_equal(df.chunks[0].dtypes, cdf.chunks[0].dtypes)

    assert cdf is to_gpu(cdf)

    # test series
    sdata = data.iloc[:, 0]
    series = from_pandas_series(sdata)
    cseries = to_gpu(series)

    assert series.index_value == cseries.index_value
    assert cseries.op.gpu is True

    series, cseries = tile(series, cseries)

    assert series.nsplits == cseries.nsplits
    assert series.chunks[0].index_value == cseries.chunks[0].index_value
    assert cseries.chunks[0].op.gpu is True

    assert cseries is to_gpu(cseries)


def test_to_cpu():
    data = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.random.randint(-100, 100, size=(10,)),
        columns=[np.random.bytes(10) for _ in range(10)],
    )
    df = from_pandas_df(data)
    cdf = to_gpu(df)
    df2 = to_cpu(cdf)

    assert df.index_value == df2.index_value
    assert df.columns_value == df2.columns_value
    assert df2.op.gpu is False
    pd.testing.assert_series_equal(df.dtypes, df2.dtypes)

    df, df2 = tile(df, df2)

    assert df.nsplits == df2.nsplits
    assert df.chunks[0].index_value == df2.chunks[0].index_value
    assert df.chunks[0].columns_value == df2.chunks[0].columns_value
    assert df2.chunks[0].op.gpu is False
    pd.testing.assert_series_equal(df.chunks[0].dtypes, df2.chunks[0].dtypes)

    assert df2 is to_cpu(df2)


def test_rechunk():
    raw = pd.DataFrame(np.random.rand(10, 10))
    df = from_pandas_df(raw, chunk_size=3)
    df2 = tile(df.rechunk(4))

    assert df2.shape == (10, 10)
    assert len(df2.chunks) == 9

    assert df2.chunks[0].shape == (4, 4)
    pd.testing.assert_index_equal(
        df2.chunks[0].index_value.to_pandas(), pd.RangeIndex(4)
    )
    pd.testing.assert_index_equal(
        df2.chunks[0].columns_value.to_pandas(), pd.RangeIndex(4)
    )
    pd.testing.assert_series_equal(df2.chunks[0].dtypes, raw.dtypes[:4])

    assert df2.chunks[2].shape == (4, 2)
    pd.testing.assert_index_equal(
        df2.chunks[2].index_value.to_pandas(), pd.RangeIndex(4)
    )
    pd.testing.assert_index_equal(
        df2.chunks[2].columns_value.to_pandas(), pd.RangeIndex(8, 10)
    )
    pd.testing.assert_series_equal(df2.chunks[2].dtypes, raw.dtypes[-2:])

    assert df2.chunks[-1].shape == (2, 2)
    pd.testing.assert_index_equal(
        df2.chunks[-1].index_value.to_pandas(), pd.RangeIndex(8, 10)
    )
    pd.testing.assert_index_equal(
        df2.chunks[-1].columns_value.to_pandas(), pd.RangeIndex(8, 10)
    )
    pd.testing.assert_series_equal(df2.chunks[-1].dtypes, raw.dtypes[-2:])

    for c in df2.chunks:
        assert c.shape[1] == len(c.dtypes)
        assert len(c.columns_value.to_pandas()) == len(c.dtypes)

    columns = [np.random.bytes(10) for _ in range(10)]
    index = np.random.randint(-100, 100, size=(4,))
    raw = pd.DataFrame(np.random.rand(4, 10), index=index, columns=columns)
    df = from_pandas_df(raw, chunk_size=3)
    df2 = tile(df.rechunk(6))

    assert df2.shape == (4, 10)
    assert len(df2.chunks) == 2

    assert df2.chunks[0].shape == (4, 6)
    pd.testing.assert_index_equal(
        df2.chunks[0].index_value.to_pandas(), df.index_value.to_pandas()
    )
    pd.testing.assert_index_equal(
        df2.chunks[0].columns_value.to_pandas(), pd.Index(columns[:6])
    )
    pd.testing.assert_series_equal(df2.chunks[0].dtypes, raw.dtypes[:6])

    assert df2.chunks[1].shape == (4, 4)
    pd.testing.assert_index_equal(
        df2.chunks[1].index_value.to_pandas(), df.index_value.to_pandas()
    )
    pd.testing.assert_index_equal(
        df2.chunks[1].columns_value.to_pandas(), pd.Index(columns[6:])
    )
    pd.testing.assert_series_equal(df2.chunks[1].dtypes, raw.dtypes[-4:])

    for c in df2.chunks:
        assert c.shape[1] == len(c.dtypes)
        assert len(c.columns_value.to_pandas()) == len(c.dtypes)

    # test Series rechunk
    series = from_pandas_series(pd.Series(np.random.rand(10)), chunk_size=3)
    series2 = tile(series.rechunk(4))

    assert series2.shape == (10,)
    assert len(series2.chunks) == 3
    pd.testing.assert_index_equal(series2.index_value.to_pandas(), pd.RangeIndex(10))

    assert series2.chunk_shape == (3,)
    assert series2.nsplits == ((4, 4, 2),)
    assert series2.chunks[0].shape == (4,)
    pd.testing.assert_index_equal(
        series2.chunks[0].index_value.to_pandas(), pd.RangeIndex(4)
    )
    assert series2.chunks[1].shape == (4,)
    pd.testing.assert_index_equal(
        series2.chunks[1].index_value.to_pandas(), pd.RangeIndex(4, 8)
    )
    assert series2.chunks[2].shape == (2,)
    pd.testing.assert_index_equal(
        series2.chunks[2].index_value.to_pandas(), pd.RangeIndex(8, 10)
    )

    series2 = tile(series.rechunk(1))

    assert series2.shape == (10,)
    assert len(series2.chunks) == 10
    pd.testing.assert_index_equal(series2.index_value.to_pandas(), pd.RangeIndex(10))

    assert series2.chunk_shape == (10,)
    assert series2.nsplits == ((1,) * 10,)
    assert series2.chunks[0].shape == (1,)
    pd.testing.assert_index_equal(
        series2.chunks[0].index_value.to_pandas(), pd.RangeIndex(1)
    )

    # no need to rechunk
    series2 = tile(series.rechunk(3))
    series = tile(series)
    assert series2.chunk_shape == series.chunk_shape
    assert series2.nsplits == series.nsplits


def test_data_frame_apply():
    cols = [chr(ord("A") + i) for i in range(10)]
    df_raw = pd.DataFrame(dict((c, [i**2 for i in range(20)]) for c in cols))

    old_chunk_store_limit = options.chunk_store_limit
    try:
        options.chunk_store_limit = 20

        df = from_pandas_df(df_raw, chunk_size=5)

        def df_func_with_err(v):
            assert len(v) > 2
            return v.sort_values()

        with pytest.raises(TypeError):
            df.apply(df_func_with_err)

        r = df.apply(df_func_with_err, output_type="dataframe", dtypes=df_raw.dtypes)
        assert r.shape == (np.nan, df.shape[-1])
        assert r.op._op_type_ == opcodes.APPLY
        assert r.op.output_types[0] == OutputType.dataframe
        assert r.op.elementwise is False

        r = df.apply("ffill")
        assert r.op._op_type_ == opcodes.FILL_NA

        r = tile(df.apply(np.sqrt))
        assert all(v == np.dtype("float64") for v in r.dtypes) is True
        assert r.shape == df.shape
        assert r.op._op_type_ == opcodes.APPLY
        assert r.op.output_types[0] == OutputType.dataframe
        assert r.op.elementwise is True

        r = tile(df.apply(lambda x: pd.Series([1, 2])))
        assert all(v == np.dtype("int64") for v in r.dtypes) is True
        assert r.shape == (np.nan, df.shape[1])
        assert r.op.output_types[0] == OutputType.dataframe
        assert r.chunks[0].shape == (np.nan, 1)
        assert r.chunks[0].inputs[0].shape[0] == df_raw.shape[0]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE
        assert r.op.elementwise is False

        r = tile(df.apply(np.sum, axis="index"))
        assert np.dtype("int64") == r.dtype
        assert r.shape == (df.shape[1],)
        assert r.op.output_types[0] == OutputType.series
        assert r.chunks[0].shape == (20 // df.shape[0],)
        assert r.chunks[0].inputs[0].shape[0] == df_raw.shape[0]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE
        assert r.op.elementwise is False

        r = tile(df.apply(np.sum, axis="columns"))
        assert np.dtype("int64") == r.dtype
        assert r.shape == (df.shape[0],)
        assert r.op.output_types[0] == OutputType.series
        assert r.chunks[0].shape == (20 // df.shape[1],)
        assert r.chunks[0].inputs[0].shape[1] == df_raw.shape[1]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE
        assert r.op.elementwise is False

        r = tile(df.apply(lambda x: pd.Series([1, 2], index=["foo", "bar"]), axis=1))
        assert all(v == np.dtype("int64") for v in r.dtypes) is True
        assert r.shape == (df.shape[0], np.nan)
        assert r.op.output_types[0] == OutputType.dataframe
        assert r.chunks[0].shape == (20 // df.shape[1], np.nan)
        assert r.chunks[0].inputs[0].shape[1] == df_raw.shape[1]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE
        assert r.op.elementwise is False

        r = tile(df.apply(lambda x: [1, 2], axis=1, result_type="expand"))
        assert all(v == np.dtype("int64") for v in r.dtypes) is True
        assert r.shape == (df.shape[0], np.nan)
        assert r.op.output_types[0] == OutputType.dataframe
        assert r.chunks[0].shape == (20 // df.shape[1], np.nan)
        assert r.chunks[0].inputs[0].shape[1] == df_raw.shape[1]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE
        assert r.op.elementwise is False

        r = tile(df.apply(lambda x: list(range(10)), axis=1, result_type="reduce"))
        assert np.dtype("object") == r.dtype
        assert r.shape == (df.shape[0],)
        assert r.op.output_types[0] == OutputType.series
        assert r.chunks[0].shape == (20 // df.shape[1],)
        assert r.chunks[0].inputs[0].shape[1] == df_raw.shape[1]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE
        assert r.op.elementwise is False

        r = tile(df.apply(lambda x: list(range(10)), axis=1, result_type="broadcast"))
        assert all(v == np.dtype("int64") for v in r.dtypes) is True
        assert r.shape == (df.shape[0], np.nan)
        assert r.op.output_types[0] == OutputType.dataframe
        assert r.chunks[0].shape == (20 // df.shape[1], np.nan)
        assert r.chunks[0].inputs[0].shape[1] == df_raw.shape[1]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE
        assert r.op.elementwise is False
    finally:
        options.chunk_store_limit = old_chunk_store_limit

    raw = pd.DataFrame({"a": [np.array([1, 2, 3]), np.array([4, 5, 6])]})
    df = from_pandas_df(raw)
    df2 = df.apply(
        lambda x: x["a"].astype(pd.Series),
        axis=1,
        output_type="dataframe",
        dtypes=pd.Series([np.dtype(float)] * 3),
    )
    assert df2.ndim == 2


def test_series_apply():
    idxes = [chr(ord("A") + i) for i in range(20)]
    s_raw = pd.Series([i**2 for i in range(20)], index=idxes)

    series = from_pandas_series(s_raw, chunk_size=5)

    r = tile(series.apply("add", args=(1,)))
    assert r.op._op_type_ == opcodes.ADD

    r = tile(series.apply(np.sqrt))
    assert np.dtype("float64") == r.dtype
    assert r.shape == series.shape
    assert r.index_value is series.index_value
    assert r.op._op_type_ == opcodes.APPLY
    assert r.op.output_types[0] == OutputType.series
    assert r.chunks[0].shape == (5,)
    assert r.chunks[0].inputs[0].shape == (5,)

    r = tile(series.apply("sqrt"))
    assert np.dtype("float64") == r.dtype
    assert r.shape == series.shape
    assert r.op._op_type_ == opcodes.APPLY
    assert r.op.output_types[0] == OutputType.series
    assert r.chunks[0].shape == (5,)
    assert r.chunks[0].inputs[0].shape == (5,)

    r = tile(series.apply(lambda x: [x, x + 1], convert_dtype=False))
    assert np.dtype("object") == r.dtype
    assert r.shape == series.shape
    assert r.op._op_type_ == opcodes.APPLY
    assert r.op.output_types[0] == OutputType.series
    assert r.chunks[0].shape == (5,)
    assert r.chunks[0].inputs[0].shape == (5,)

    s_raw2 = pd.Series([np.array([1, 2, 3]), np.array([4, 5, 6])])
    series = from_pandas_series(s_raw2)

    r = series.apply(np.sum)
    assert r.dtype == np.dtype(object)

    r = series.apply(lambda x: pd.Series([1]), output_type="dataframe")
    expected = s_raw2.apply(lambda x: pd.Series([1]))
    pd.testing.assert_series_equal(r.dtypes, expected.dtypes)

    dtypes = pd.Series([np.dtype(float)] * 3)
    r = series.apply(pd.Series, output_type="dataframe", dtypes=dtypes)
    assert r.ndim == 2
    pd.testing.assert_series_equal(r.dtypes, dtypes)
    assert r.shape == (2, 3)

    r = series.apply(
        pd.Series, output_type="dataframe", dtypes=dtypes, index=pd.RangeIndex(2)
    )
    assert r.ndim == 2
    pd.testing.assert_series_equal(r.dtypes, dtypes)
    assert r.shape == (2, 3)

    with pytest.raises(AttributeError, match="abc"):
        series.apply("abc")

    with pytest.raises(TypeError):
        # dtypes not provided
        series.apply(lambda x: x.tolist(), output_type="dataframe")


def test_transform():
    cols = [chr(ord("A") + i) for i in range(10)]
    df_raw = pd.DataFrame(dict((c, [i**2 for i in range(20)]) for c in cols))
    df = from_pandas_df(df_raw, chunk_size=5)

    idxes = [chr(ord("A") + i) for i in range(20)]
    s_raw = pd.Series([i**2 for i in range(20)], index=idxes)
    series = from_pandas_series(s_raw, chunk_size=5)

    def rename_fn(f, new_name):
        f.__name__ = new_name
        return f

    old_chunk_store_limit = options.chunk_store_limit
    try:
        options.chunk_store_limit = 20

        # DATAFRAME CASES

        # test transform with infer failure
        def transform_df_with_err(v):
            assert len(v) > 2
            return v.sort_values()

        with pytest.raises(TypeError):
            df.transform(transform_df_with_err)

        r = tile(df.transform(transform_df_with_err, dtypes=df_raw.dtypes))
        assert r.shape == df.shape
        assert r.op._op_type_ == opcodes.TRANSFORM
        assert r.op.output_types[0] == OutputType.dataframe
        assert r.chunks[0].shape == (df.shape[0], 20 // df.shape[0])
        assert r.chunks[0].inputs[0].shape[0] == df_raw.shape[0]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE

        # test transform scenarios on data frames
        r = tile(df.transform(lambda x: list(range(len(x)))))
        assert all(v == np.dtype("int64") for v in r.dtypes) is True
        assert r.shape == df.shape
        assert r.op._op_type_ == opcodes.TRANSFORM
        assert r.op.output_types[0] == OutputType.dataframe
        assert r.chunks[0].shape == (df.shape[0], 20 // df.shape[0])
        assert r.chunks[0].inputs[0].shape[0] == df_raw.shape[0]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE

        r = tile(df.transform(lambda x: list(range(len(x))), axis=1))
        assert all(v == np.dtype("int64") for v in r.dtypes) is True
        assert r.shape == df.shape
        assert r.op._op_type_ == opcodes.TRANSFORM
        assert r.op.output_types[0] == OutputType.dataframe
        assert r.chunks[0].shape == (20 // df.shape[1], df.shape[1])
        assert r.chunks[0].inputs[0].shape[1] == df_raw.shape[1]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE

        r = tile(df.transform(["cumsum", "cummax", lambda x: x + 1]))
        assert all(v == np.dtype("int64") for v in r.dtypes) is True
        assert r.shape == (df.shape[0], df.shape[1] * 3)
        assert r.op._op_type_ == opcodes.TRANSFORM
        assert r.op.output_types[0] == OutputType.dataframe
        assert r.chunks[0].shape == (df.shape[0], 20 // df.shape[0] * 3)
        assert r.chunks[0].inputs[0].shape[0] == df_raw.shape[0]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE

        r = tile(
            df.transform(
                {"A": "cumsum", "D": ["cumsum", "cummax"], "F": lambda x: x + 1}
            )
        )
        assert all(v == np.dtype("int64") for v in r.dtypes) is True
        assert r.shape == (df.shape[0], 4)
        assert r.op._op_type_ == opcodes.TRANSFORM
        assert r.op.output_types[0] == OutputType.dataframe
        assert r.chunks[0].shape == (df.shape[0], 1)
        assert r.chunks[0].inputs[0].shape[0] == df_raw.shape[0]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE

        # test agg scenarios on series
        r = tile(df.transform(lambda x: x.iloc[:-1], _call_agg=True))
        assert all(v == np.dtype("int64") for v in r.dtypes) is True
        assert r.shape == (np.nan, df.shape[1])
        assert r.op._op_type_ == opcodes.TRANSFORM
        assert r.op.output_types[0] == OutputType.dataframe
        assert r.chunks[0].shape == (np.nan, 1)
        assert r.chunks[0].inputs[0].shape[0] == df_raw.shape[0]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE

        r = tile(df.transform(lambda x: x.iloc[:-1], axis=1, _call_agg=True))
        assert all(v == np.dtype("int64") for v in r.dtypes) is True
        assert r.shape == (df.shape[0], np.nan)
        assert r.op._op_type_ == opcodes.TRANSFORM
        assert r.op.output_types[0] == OutputType.dataframe
        assert r.chunks[0].shape == (2, np.nan)
        assert r.chunks[0].inputs[0].shape[1] == df_raw.shape[1]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE

        fn_list = [
            rename_fn(lambda x: x.iloc[1:].reset_index(drop=True), "f1"),
            lambda x: x.iloc[:-1].reset_index(drop=True),
        ]
        r = tile(df.transform(fn_list, _call_agg=True))
        assert all(v == np.dtype("int64") for v in r.dtypes) is True
        assert r.shape == (np.nan, df.shape[1] * 2)
        assert r.op._op_type_ == opcodes.TRANSFORM
        assert r.op.output_types[0] == OutputType.dataframe
        assert r.chunks[0].shape == (np.nan, 2)
        assert r.chunks[0].inputs[0].shape[0] == df_raw.shape[0]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE

        r = tile(df.transform(lambda x: x.sum(), _call_agg=True))
        assert r.dtype == np.dtype("int64")
        assert r.shape == (df.shape[1],)
        assert r.op._op_type_ == opcodes.TRANSFORM
        assert r.op.output_types[0] == OutputType.series
        assert r.chunks[0].shape == (20 // df.shape[0],)
        assert r.chunks[0].inputs[0].shape[0] == df_raw.shape[0]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE

        fn_dict = {
            "A": rename_fn(lambda x: x.iloc[1:].reset_index(drop=True), "f1"),
            "D": [
                rename_fn(lambda x: x.iloc[1:].reset_index(drop=True), "f1"),
                lambda x: x.iloc[:-1].reset_index(drop=True),
            ],
            "F": lambda x: x.iloc[:-1].reset_index(drop=True),
        }
        r = tile(df.transform(fn_dict, _call_agg=True))
        assert all(v == np.dtype("int64") for v in r.dtypes) is True
        assert r.shape == (np.nan, 4)
        assert r.op._op_type_ == opcodes.TRANSFORM
        assert r.op.output_types[0] == OutputType.dataframe
        assert r.chunks[0].shape == (np.nan, 1)
        assert r.chunks[0].inputs[0].shape[0] == df_raw.shape[0]
        assert r.chunks[0].inputs[0].op._op_type_ == opcodes.CONCATENATE

        # SERIES CASES
        # test transform scenarios on series
        r = tile(series.transform(lambda x: x + 1))
        assert np.dtype("int64") == r.dtype
        assert r.shape == series.shape
        assert r.op._op_type_ == opcodes.TRANSFORM
        assert r.op.output_types[0] == OutputType.series
        assert r.chunks[0].shape == (5,)
        assert r.chunks[0].inputs[0].shape == (5,)
    finally:
        options.chunk_store_limit = old_chunk_store_limit


def test_string_method():
    s = pd.Series(["a", "b", "c"], name="s")
    series = from_pandas_series(s, chunk_size=2)

    with pytest.raises(AttributeError):
        _ = series.str.non_exist

    r = series.str.contains("c")
    assert r.dtype == np.bool_
    assert r.name == s.name
    pd.testing.assert_index_equal(r.index_value.to_pandas(), s.index)
    assert r.shape == s.shape

    r = tile(r)
    for i, c in enumerate(r.chunks):
        assert c.index == (i,)
        assert c.dtype == np.bool_
        assert c.name == s.name
        pd.testing.assert_index_equal(
            c.index_value.to_pandas(), s.index[i * 2 : (i + 1) * 2]
        )
        assert c.shape == (2,) if i == 0 else (1,)

    r = series.str.split(",", expand=True, n=1)
    assert r.op.output_types[0] == OutputType.dataframe
    assert r.shape == (3, 2)
    pd.testing.assert_index_equal(r.index_value.to_pandas(), s.index)
    pd.testing.assert_index_equal(r.columns_value.to_pandas(), pd.RangeIndex(2))

    r = tile(r)
    for i, c in enumerate(r.chunks):
        assert c.index == (i, 0)
        pd.testing.assert_index_equal(
            c.index_value.to_pandas(), s.index[i * 2 : (i + 1) * 2]
        )
        pd.testing.assert_index_equal(c.columns_value.to_pandas(), pd.RangeIndex(2))
        assert c.shape == (2, 2) if i == 0 else (1, 2)

    with pytest.raises(TypeError):
        _ = series.str.cat([["1", "2"]])

    with pytest.raises(ValueError):
        _ = series.str.cat(["1", "2"])

    with pytest.raises(ValueError):
        _ = series.str.cat(",")

    with pytest.raises(TypeError):
        _ = series.str.cat({"1", "2", "3"})

    r = series.str.cat(sep=",")
    assert r.op.output_types[0] == OutputType.scalar
    assert r.dtype == s.dtype

    r = tile(r)
    assert len(r.chunks) == 1
    assert r.chunks[0].op.output_types[0] == OutputType.scalar
    assert r.chunks[0].dtype == s.dtype

    r = series.str.extract(r"[ab](\d)", expand=False)
    assert r.op.output_types[0] == OutputType.series
    assert r.dtype == s.dtype

    r = tile(r)
    for i, c in enumerate(r.chunks):
        assert c.index == (i,)
        assert c.dtype == s.dtype
        assert c.name == s.name
        pd.testing.assert_index_equal(
            c.index_value.to_pandas(), s.index[i * 2 : (i + 1) * 2]
        )
        assert c.shape == (2,) if i == 0 else (1,)

    r = series.str.extract(r"[ab](\d)", expand=True)
    assert r.op.output_types[0] == OutputType.dataframe
    assert r.shape == (3, 1)
    pd.testing.assert_index_equal(r.index_value.to_pandas(), s.index)
    pd.testing.assert_index_equal(r.columns_value.to_pandas(), pd.RangeIndex(1))

    r = tile(r)
    for i, c in enumerate(r.chunks):
        assert c.index == (i, 0)
        pd.testing.assert_index_equal(
            c.index_value.to_pandas(), s.index[i * 2 : (i + 1) * 2]
        )
        pd.testing.assert_index_equal(c.columns_value.to_pandas(), pd.RangeIndex(1))
        assert c.shape == (2, 1) if i == 0 else (1, 1)

    assert "lstrip" in dir(series.str)


def test_datetime_method():
    s = pd.Series(
        [pd.Timestamp("2020-1-1"), pd.Timestamp("2020-2-1"), pd.Timestamp("2020-3-1")],
        name="ss",
    )
    series = from_pandas_series(s, chunk_size=2)

    r = series.dt.year
    assert r.dtype == s.dt.year.dtype
    pd.testing.assert_index_equal(r.index_value.to_pandas(), s.index)
    assert r.shape == s.shape
    assert r.op.output_types[0] == OutputType.series
    assert r.name == s.dt.year.name

    r = tile(r)
    for i, c in enumerate(r.chunks):
        assert c.index == (i,)
        assert c.dtype == s.dt.year.dtype
        assert c.op.output_types[0] == OutputType.series
        assert r.name == s.dt.year.name
        pd.testing.assert_index_equal(
            c.index_value.to_pandas(), s.index[i * 2 : (i + 1) * 2]
        )
        assert c.shape == (2,) if i == 0 else (1,)

    with pytest.raises(AttributeError):
        _ = series.dt.non_exist

    assert "ceil" in dir(series.dt)


def test_series_isin():
    # one chunk in multiple chunks
    a = from_pandas_series(pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), chunk_size=10)
    b = from_pandas_series(pd.Series([2, 1, 9, 3]), chunk_size=2)

    r = tile(a.isin(b))
    for i, c in enumerate(r.chunks):
        assert c.index == (i,)
        assert c.dtype == np.dtype("bool")
        assert c.shape == (10,)
        assert len(c.op.inputs) == 2
        assert c.op.output_types[0] == OutputType.series
        assert c.op.inputs[0].index == (i,)
        assert c.op.inputs[0].shape == (10,)
        assert c.op.inputs[1].index == (0,)
        assert c.op.inputs[1].shape == (4,)  # has been rechunked

    # multiple chunk in one chunks
    a = from_pandas_series(pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), chunk_size=2)
    b = from_pandas_series(pd.Series([2, 1, 9, 3]), chunk_size=4)

    r = tile(a.isin(b))
    for i, c in enumerate(r.chunks):
        assert c.index == (i,)
        assert c.dtype == np.dtype("bool")
        assert c.shape == (2,)
        assert len(c.op.inputs) == 2
        assert c.op.output_types[0] == OutputType.series
        assert c.op.inputs[0].index == (i,)
        assert c.op.inputs[0].shape == (2,)
        assert c.op.inputs[1].index == (0,)
        assert c.op.inputs[1].shape == (4,)

    # multiple chunk in multiple chunks
    a = from_pandas_series(pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), chunk_size=2)
    b = from_pandas_series(pd.Series([2, 1, 9, 3]), chunk_size=2)

    r = tile(a.isin(b))
    for i, c in enumerate(r.chunks):
        assert c.index == (i,)
        assert c.dtype == np.dtype("bool")
        assert c.shape == (2,)
        assert len(c.op.inputs) == 2
        assert c.op.output_types[0] == OutputType.series
        assert c.op.inputs[0].index == (i,)
        assert c.op.inputs[0].shape == (2,)
        assert c.op.inputs[1].index == (0,)
        assert c.op.inputs[1].shape == (4,)  # has been rechunked

    with pytest.raises(TypeError):
        _ = a.isin("sth")

    with pytest.raises(TypeError):
        _ = a.to_frame().isin("sth")


def test_cut():
    s = from_pandas_series(pd.Series([1.0, 2.0, 3.0, 4.0]), chunk_size=2)

    with pytest.raises(ValueError):
        _ = cut(s, -1)

    with pytest.raises(ValueError):
        _ = cut([[1, 2], [3, 4]], 3)

    with pytest.raises(ValueError):
        _ = cut([], 3)

    r, b = cut(s, [1.5, 2.5], retbins=True)
    assert isinstance(r, SERIES_TYPE)
    assert isinstance(b, TENSOR_TYPE)

    r = tile(r)

    assert len(r.chunks) == 2
    for c in r.chunks:
        assert isinstance(c, SERIES_CHUNK_TYPE)
        assert c.shape == (2,)

    r = cut(s.to_tensor(), [1.5, 2.5])
    assert isinstance(r, CATEGORICAL_TYPE)
    assert len(r) == len(s)
    assert "Categorical" in repr(r)

    r = tile(r)

    assert len(r.chunks) == 2
    for c in r.chunks:
        assert isinstance(c, CATEGORICAL_CHUNK_TYPE)
        assert c.shape == (2,)
        assert c.ndim == 1

    r = cut([0, 1, 1, 2], bins=4, labels=False)
    assert isinstance(r, TENSOR_TYPE)
    e = pd.cut([0, 1, 1, 2], bins=4, labels=False)
    assert r.dtype == e.dtype


def test_transpose():
    s = pd.DataFrame({"a": [1, 2, 3], "b": ["5", "-6", "7"], "c": [1, 2, 3]})
    df = from_pandas_df(s, chunk_size=2)

    r = tile(df.transpose())
    assert len(r.chunks) == 4
    assert isinstance(r, DATAFRAME_TYPE)

    r = tile(df.T)
    assert len(r.chunks) == 4
    assert isinstance(r, DATAFRAME_TYPE)


def test_to_numeric():
    raw = pd.DataFrame({"a": [1.0, 2, 3, -3]})
    df = from_pandas_df(raw, chunk_size=2)

    with pytest.raises(ValueError):
        _ = to_numeric(df)

    with pytest.raises(ValueError):
        _ = to_numeric([["1.0", 1]])

    with pytest.raises(ValueError):
        _ = to_numeric([])

    s = from_pandas_series(pd.Series(["1.0", "2.0", 1, -2]), chunk_size=2)
    r = tile(to_numeric(s))
    assert len(r.chunks) == 2
    assert isinstance(r, SERIES_TYPE)

    r = tile(to_numeric(["1.0", "2.0", 1, -2]))
    assert isinstance(r, TENSOR_TYPE)


def test_astype():
    s = from_pandas_series(pd.Series([1, 2, 1, 2], name="a"), chunk_size=2)
    with pytest.raises(KeyError):
        astype(s, {"b": "str"})

    df = from_pandas_df(
        pd.DataFrame({"a": [1, 2, 1, 2], "b": ["a", "b", "a", "b"]}), chunk_size=2
    )

    with pytest.raises(KeyError):
        astype(df, {"c": "str", "a": "str"})


def test_get_dummies():
    raw = pd.DataFrame(
        {
            "a": [1.1, 2.1, 3.1],
            "b": ["5", "-6", "-7"],
            "c": [1, 2, 3],
            "d": ["2", "3", "4"],
        }
    )
    df = from_pandas_df(raw, chunk_size=2)

    with pytest.raises(TypeError):
        _ = get_dummies(df, columns="a")

    with pytest.raises(ValueError):
        _ = get_dummies(df, prefix=["col1"])

    with pytest.raises(ValueError):
        _ = get_dummies(df, columns=["a"], prefix={"a": "col1", "c": "col2"})

    with pytest.raises(KeyError):
        _ = get_dummies(df, columns=["a", "b"], prefix={"a": "col1", "c": "col2"})

    r = get_dummies(df)
    assert isinstance(r, DATAFRAME_TYPE)


def test_drop():
    # test dataframe drop
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        rs.randint(1000, size=(20, 8)), columns=["c" + str(i + 1) for i in range(8)]
    )

    df = from_pandas_df(raw, chunk_size=8)

    with pytest.raises(KeyError):
        df.drop(columns=["c9"])
    with pytest.raises(NotImplementedError):
        df.drop(columns=from_pandas_series(pd.Series(["c9"])))

    r = df.drop(columns=["c1"])
    pd.testing.assert_index_equal(r.index_value.to_pandas(), raw.index)

    tiled = tile(r)
    start = 0
    for c in tiled.chunks:
        raw_index = raw.index[start : start + c.shape[0]]
        start += c.shape[0]
        pd.testing.assert_index_equal(raw_index, c.index_value.to_pandas())

    df = from_pandas_df(raw, chunk_size=3)

    columns = ["c2", "c4", "c5", "c6"]
    index = [3, 6, 7]
    r = df.drop(columns=columns, index=index)
    assert isinstance(r, DATAFRAME_TYPE)

    # test series drop
    raw = pd.Series(rs.randint(1000, size=(20,)))
    series = from_pandas_series(raw, chunk_size=3)

    r = series.drop(index=index)
    assert isinstance(r, SERIES_TYPE)

    # test index drop
    ser = pd.Series(range(20))
    rs.shuffle(ser)
    raw = pd.Index(ser)

    idx = from_pandas_index(raw)

    r = idx.drop(index)
    assert isinstance(r, INDEX_TYPE)


def test_drop_duplicates():
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        rs.randint(1000, size=(20, 7)), columns=["c" + str(i + 1) for i in range(7)]
    )
    raw["c7"] = [f"s{j}" for j in range(20)]

    df = from_pandas_df(raw, chunk_size=10)
    with pytest.raises(ValueError):
        df.drop_duplicates(method="unknown")
    with pytest.raises(KeyError):
        df.drop_duplicates(subset="c8")

    # test auto method selection
    assert tile(df.drop_duplicates()).chunks[0].op.method == "tree"
    # subset size less than chunk_store_limit
    assert (
        tile(df.drop_duplicates(subset=["c1", "c3"])).chunks[0].op.method
        == "subset_tree"
    )
    with option_context({"chunk_store_limit": 5}):
        # subset size greater than chunk_store_limit
        assert (
            tile(df.drop_duplicates(subset=["c1", "c3"])).chunks[0].op.method == "tree"
        )
    assert tile(df.drop_duplicates(subset=["c1", "c7"])).chunks[0].op.method == "tree"
    assert tile(df["c7"].drop_duplicates()).chunks[0].op.method == "tree"

    s = df["c7"]
    with pytest.raises(ValueError):
        s.drop_duplicates(method="unknown")


def test_memory_usage():
    dtypes = ["int64", "float64", "complex128", "object", "bool"]
    data = dict([(t, np.ones(shape=500).astype(t)) for t in dtypes])
    raw = pd.DataFrame(data)

    df = from_pandas_df(raw, chunk_size=(500, 2))
    r = tile(df.memory_usage())

    assert isinstance(r, SERIES_TYPE)
    assert r.shape == (6,)
    assert len(r.chunks) == 3
    assert r.chunks[0].op.stage is None

    df = from_pandas_df(raw, chunk_size=(100, 3))
    r = tile(df.memory_usage(index=True))

    assert isinstance(r, SERIES_TYPE)
    assert r.shape == (6,)
    assert len(r.chunks) == 2
    assert r.chunks[0].op.stage == OperandStage.reduce

    r = tile(df.memory_usage(index=False))

    assert isinstance(r, SERIES_TYPE)
    assert r.shape == (5,)
    assert len(r.chunks) == 2
    assert r.chunks[0].op.stage == OperandStage.reduce

    raw = pd.Series(np.ones(shape=500).astype("object"), name="s")

    series = from_pandas_series(raw)
    r = tile(series.memory_usage())

    assert isinstance(r, TENSOR_TYPE)
    assert r.shape == ()
    assert len(r.chunks) == 1
    assert r.chunks[0].op.stage is None

    series = from_pandas_series(raw, chunk_size=100)
    r = tile(series.memory_usage())

    assert isinstance(r, TENSOR_TYPE)
    assert r.shape == ()
    assert len(r.chunks) == 1
    assert r.chunks[0].op.stage == OperandStage.reduce


def test_shift():
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        rs.randint(1000, size=(10, 8)),
        columns=["col" + str(i + 1) for i in range(8)],
        index=pd.date_range("2021-1-1", periods=10),
    )
    df = from_pandas_df(raw, chunk_size=5)

    df2 = df.shift(1)
    df2 = tile(df2)

    for c in df2.chunks:
        pd.testing.assert_index_equal(c.dtypes.index, c.columns_value.to_pandas())

    df2 = df.shift(1, freq="D")
    df2 = tile(df2)

    for c in df2.chunks:
        pd.testing.assert_index_equal(c.dtypes.index, c.columns_value.to_pandas())


def test_eval_query():
    rs = np.random.RandomState(0)
    raw = pd.DataFrame({"a": rs.rand(100), "b": rs.rand(100), "c c": rs.rand(100)})
    df = from_pandas_df(raw, chunk_size=(10, 2))

    with pytest.raises(NotImplementedError):
        mars_eval("df.a * 2", engine="numexpr")
    with pytest.raises(NotImplementedError):
        mars_eval("df.a * 2", parser="pandas")
    with pytest.raises(TypeError):
        df.eval(df)
    with pytest.raises(SyntaxError):
        df.query(
            """
        a + b
        a + `c c`
        """
        )
    with pytest.raises(SyntaxError):
        df.eval(
            """
        def a():
            return v
        a()
        """
        )
    with pytest.raises(SyntaxError):
        df.eval("a + `c")
    with pytest.raises(KeyError):
        df.eval("a + c")
    with pytest.raises(ValueError):
        df.eval("p, q = a + c")
    with pytest.raises(ValueError):
        df.query("p = a + c")


def test_empty():
    # for DataFrame
    assert from_pandas_df(pd.DataFrame()).empty == pd.DataFrame().empty
    assert from_pandas_df(pd.DataFrame({})).empty == pd.DataFrame({}).empty
    assert (
        from_pandas_df(pd.DataFrame({"a": []})).empty == pd.DataFrame({"a": []}).empty
    )
    assert (
        from_pandas_df(pd.DataFrame({"a": [1]})).empty == pd.DataFrame({"a": [1]}).empty
    )
    assert (
        from_pandas_df(pd.DataFrame({"a": [1], "b": [2]})).empty
        == pd.DataFrame({"a": [1], "b": [2]}).empty
    )
    assert (
        from_pandas_df(pd.DataFrame(np.empty(shape=(4, 0)))).empty
        == pd.DataFrame(np.empty(shape=(4, 0))).empty
    )

    # for Series
    assert from_pandas_series(pd.Series()).empty == pd.Series().empty
    assert from_pandas_series(pd.Series({})).empty == pd.Series({}).empty
    assert from_pandas_series(pd.Series({"a": []})).empty == pd.Series({"a": []}).empty
    assert (
        from_pandas_series(pd.Series({"a": [1]})).empty == pd.Series({"a": [1]}).empty
    )

    # Maybe fail due to lazy evaluation
    with pytest.raises(ValueError):
        a = from_pandas_df(pd.DataFrame(np.random.rand(10, 2)))
        assert a[a > 0].empty
    with pytest.raises(ValueError):
        a = from_pandas_series(pd.Series(np.random.rand(10)))
        assert a[a > 0].empty
