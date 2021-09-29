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
from io import BytesIO, StringIO

import pandas as pd
import numpy as np
import pytest

from .... import dataframe as md
from ....tests.core import require_hadoop


TEST_DIR = "/tmp/test"


@require_hadoop
@pytest.fixture(scope="module")
def setup_hdfs():
    import pyarrow

    hdfs = pyarrow.hdfs.connect(host="localhost", port=8020)
    if hdfs.exists(TEST_DIR):
        hdfs.rm(TEST_DIR, recursive=True)
    try:
        yield hdfs
    finally:
        if hdfs.exists(TEST_DIR):
            hdfs.rm(TEST_DIR, recursive=True)


@require_hadoop
def test_read_csv_execution(setup, setup_hdfs):
    hdfs = setup_hdfs

    with hdfs.open(f"{TEST_DIR}/simple_test.csv", "wb", replication=1) as f:
        f.write(b"name,amount,id\nAlice,100,1\nBob,200,2")

    df = md.read_csv(f"hdfs://localhost:8020{TEST_DIR}/simple_test.csv")
    expected = pd.read_csv(BytesIO(b"name,amount,id\nAlice,100,1\nBob,200,2"))
    res = df.to_pandas()
    pd.testing.assert_frame_equal(expected, res)

    test_df = pd.DataFrame(
        {
            "A": np.random.rand(20),
            "B": [
                pd.Timestamp("2020-01-01") + pd.Timedelta(days=random.randint(0, 31))
                for _ in range(20)
            ],
            "C": np.random.rand(20),
            "D": np.random.randint(0, 100, size=(20,)),
            "E": ["foo" + str(random.randint(0, 999999)) for _ in range(20)],
        }
    )
    buf = StringIO()
    test_df[:10].to_csv(buf)
    csv_content = buf.getvalue().encode()

    buf = StringIO()
    test_df[10:].to_csv(buf)
    csv_content2 = buf.getvalue().encode()

    with hdfs.open(f"{TEST_DIR}/chunk_test.csv", "wb", replication=1) as f:
        f.write(csv_content)

    df = md.read_csv(f"hdfs://localhost:8020{TEST_DIR}/chunk_test.csv", chunk_bytes=50)
    expected = pd.read_csv(BytesIO(csv_content))
    res = df.to_pandas()
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True), res.reset_index(drop=True)
    )

    test_read_dir = f"{TEST_DIR}/test_read_csv_directory"
    hdfs.mkdir(test_read_dir)
    with hdfs.open(f"{test_read_dir}/part.csv", "wb", replication=1) as f:
        f.write(csv_content)
    with hdfs.open(f"{test_read_dir}/part2.csv", "wb", replication=1) as f:
        f.write(csv_content2)

    df = md.read_csv(f"hdfs://localhost:8020{test_read_dir}", chunk_bytes=50)
    expected = pd.concat(
        [pd.read_csv(BytesIO(csv_content)), pd.read_csv(BytesIO(csv_content2))]
    )
    res = df.to_pandas()
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True), res.reset_index(drop=True)
    )


@require_hadoop
def test_read_parquet_execution(setup, setup_hdfs):
    hdfs = setup_hdfs

    test_df = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
            "c": np.random.rand(10),
        }
    )
    test_df2 = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
            "c": np.random.rand(10),
        }
    )

    with hdfs.open(f"{TEST_DIR}/test.parquet", "wb", replication=1) as f:
        test_df.to_parquet(f, row_group_size=3)

    df = md.read_parquet(f"hdfs://localhost:8020{TEST_DIR}/test.parquet")
    res = df.to_pandas()
    pd.testing.assert_frame_equal(res, test_df)

    hdfs.mkdir(f"{TEST_DIR}/test_partitioned")

    with hdfs.open(
        f"{TEST_DIR}/test_partitioned/file1.parquet", "wb", replication=1
    ) as f:
        test_df.to_parquet(f, row_group_size=3)
    with hdfs.open(
        f"{TEST_DIR}/test_partitioned/file2.parquet", "wb", replication=1
    ) as f:
        test_df2.to_parquet(f, row_group_size=3)

    df = md.read_parquet(f"hdfs://localhost:8020{TEST_DIR}/test_partitioned")
    res = df.to_pandas()
    pd.testing.assert_frame_equal(res, pd.concat([test_df, test_df2]))
