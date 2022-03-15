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

import numpy as np
import pandas as pd
import pytest

try:
    import vineyard
except ImportError:
    vineyard = None
try:
    import sqlalchemy
except ImportError:
    sqlalchemy = None
try:
    import pyarrow as pa
except ImportError:
    pa = None
try:
    import fastparquet
except ImportError:
    fastparquet = None
try:
    import vineyard
except ImportError:
    vineyard = None

from .... import dataframe as md
from ....tests.core import flaky
from ... import DataFrame


def test_to_csv_execution(setup):
    index = pd.RangeIndex(100, 0, -1, name="index")
    raw = pd.DataFrame(
        {
            "col1": np.random.rand(100),
            "col2": np.random.choice(["a", "b", "c"], (100,)),
            "col3": np.arange(100),
        },
        index=index,
    )
    df = DataFrame(raw, chunk_size=33)

    with tempfile.TemporaryDirectory() as base_path:
        # DATAFRAME TESTS
        # test one file with dataframe
        path = os.path.join(base_path, "out.csv")

        df.to_csv(path).execute()

        result = pd.read_csv(path, dtype=raw.dtypes.to_dict())
        result.set_index("index", inplace=True)
        pd.testing.assert_frame_equal(result, raw)

        # test multi files with dataframe
        path = os.path.join(base_path, "out-*.csv")
        df.to_csv(path).execute()

        dfs = [
            pd.read_csv(
                os.path.join(base_path, f"out-{i}.csv"), dtype=raw.dtypes.to_dict()
            )
            for i in range(4)
        ]
        result = pd.concat(dfs, axis=0)
        result.set_index("index", inplace=True)
        pd.testing.assert_frame_equal(result, raw)
        pd.testing.assert_frame_equal(dfs[1].set_index("index"), raw.iloc[33:66])

        # test df with unknown shape
        df2 = DataFrame(raw, chunk_size=(50, 2))
        df2 = df2[df2["col1"] < 1]
        path2 = os.path.join(base_path, "out2.csv")
        df2.to_csv(path2).execute()

        result = pd.read_csv(path2, dtype=raw.dtypes.to_dict())
        result.set_index("index", inplace=True)
        pd.testing.assert_frame_equal(result, raw)

        # SERIES TESTS
        series = md.Series(raw.col1, chunk_size=33)

        # test one file with series
        path = os.path.join(base_path, "out.csv")
        series.to_csv(path).execute()

        result = pd.read_csv(path, dtype=raw.dtypes.to_dict())
        result.set_index("index", inplace=True)
        pd.testing.assert_frame_equal(result, raw.col1.to_frame())

        # test multi files with series
        path = os.path.join(base_path, "out-*.csv")
        series.to_csv(path).execute()

        dfs = [
            pd.read_csv(
                os.path.join(base_path, f"out-{i}.csv"), dtype=raw.dtypes.to_dict()
            )
            for i in range(4)
        ]
        result = pd.concat(dfs, axis=0)
        result.set_index("index", inplace=True)
        pd.testing.assert_frame_equal(result, raw.col1.to_frame())
        pd.testing.assert_frame_equal(
            dfs[1].set_index("index"), raw.col1.to_frame().iloc[33:66]
        )


@pytest.mark.skipif(sqlalchemy is None, reason="sqlalchemy not installed")
def test_to_sql():
    index = pd.RangeIndex(100, 0, -1, name="index")
    raw = pd.DataFrame(
        {
            "col1": np.random.rand(100),
            "col2": np.random.choice(["a", "b", "c"], (100,)),
            "col3": np.arange(100).astype("int64"),
        },
        index=index,
    )

    with tempfile.TemporaryDirectory() as d:
        table_name1 = "test_table"
        table_name2 = "test_table2"
        uri = "sqlite:///" + os.path.join(d, "test.db")

        engine = sqlalchemy.create_engine(uri)

        # test write dataframe
        df = DataFrame(raw, chunk_size=33)
        df.to_sql(table_name1, con=engine).execute()

        written = pd.read_sql(table_name1, con=engine, index_col="index").sort_index(
            ascending=False
        )
        pd.testing.assert_frame_equal(raw, written)

        # test write with existing table
        with pytest.raises(ValueError):
            df.to_sql(table_name1, con=uri).execute()

        # test write series
        series = md.Series(raw.col1, chunk_size=33)
        with engine.connect() as conn:
            series.to_sql(table_name2, con=conn).execute()

        written = pd.read_sql(table_name2, con=engine, index_col="index").sort_index(
            ascending=False
        )
        pd.testing.assert_frame_equal(raw.col1.to_frame(), written)


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
@flaky(max_runs=3)
def test_to_parquet_arrow_execution(setup):
    raw = pd.DataFrame(
        {
            "col1": np.random.rand(100),
            "col2": np.arange(100),
            "col3": np.random.choice(["a", "b", "c"], (100,)),
        }
    )
    df = DataFrame(raw, chunk_size=33)

    with tempfile.TemporaryDirectory() as base_path:
        # DATAFRAME TESTS
        path = os.path.join(base_path, "out-*.parquet")
        df.to_parquet(path).execute()

        read_df = md.read_parquet(path)
        result = read_df.execute().fetch()
        result = result.sort_index()
        pd.testing.assert_frame_equal(result, raw)

        read_df = md.read_parquet(path)
        result = read_df.execute().fetch()
        result = result.sort_index()
        pd.testing.assert_frame_equal(result, raw)

        # test read_parquet then to_parquet
        read_df = md.read_parquet(path)
        read_df.to_parquet(path).execute()

        # test partition_cols
        path = os.path.join(base_path, "out-partitioned")
        df.to_parquet(path, partition_cols=["col3"]).execute()

        read_df = md.read_parquet(path)
        result = read_df.execute().fetch()
        result["col3"] = result["col3"].astype("object")
        pd.testing.assert_frame_equal(
            result.sort_values("col1").reset_index(drop=True),
            raw.sort_values("col1").reset_index(drop=True),
        )


@pytest.mark.skipif(fastparquet is None, reason="fastparquet not installed")
def test_to_parquet_fast_parquet_execution():
    raw = pd.DataFrame(
        {
            "col1": np.random.rand(100),
            "col2": np.arange(100),
            "col3": np.random.choice(["a", "b", "c"], (100,)),
        }
    )
    df = DataFrame(raw, chunk_size=33)

    with tempfile.TemporaryDirectory() as base_path:
        # test fastparquet
        path = os.path.join(base_path, "out-fastparquet-*.parquet")
        df.to_parquet(path, engine="fastparquet", compression="gzip").execute()


@pytest.mark.skipif(vineyard is None, reason="vineyard not installed")
def test_vineyard_execution(setup):
    raw = np.random.RandomState(0).rand(55, 55)

    extra_config = {
        "check_dtype": False,
        "check_nsplits": False,
        "check_shape": False,
        "check_dtypes": False,
        "check_columns_value": False,
        "check_index_value": False,
    }

    with vineyard.deploy.local.start_vineyardd() as (_, vineyard_socket, _):
        raw = pd.DataFrame({"a": np.arange(0, 55), "b": np.arange(55, 110)})
        a = md.DataFrame(raw, chunk_size=15)
        a.execute()  # n.b.: pre-execute

        b = a.to_vineyard(vineyard_socket=vineyard_socket)
        object_id = b.execute(extra_config=extra_config).fetch()[0][0]

        c = md.from_vineyard(object_id, vineyard_socket=vineyard_socket)
        df = c.execute(extra_config=extra_config).fetch()
        pd.testing.assert_frame_equal(df, raw)

        raw = pd.DataFrame({"a": np.arange(0, 55), "b": np.arange(55, 110)})
        a = md.DataFrame(raw, chunk_size=15)  # n.b.: no pre-execute

        b = a.to_vineyard(vineyard_socket=vineyard_socket)
        object_id = b.execute(extra_config=extra_config).fetch()[0][0]

        c = md.from_vineyard(object_id, vineyard_socket=vineyard_socket)
        df = c.execute(extra_config=extra_config).fetch()
        pd.testing.assert_frame_equal(df, raw)
