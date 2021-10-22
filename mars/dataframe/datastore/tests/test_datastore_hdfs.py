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

    yield hdfs

    if hdfs.exists(TEST_DIR):
        hdfs.rm(TEST_DIR, recursive=True)


@require_hadoop
def test_to_parquet_execution(setup, setup_hdfs):
    hdfs = setup_hdfs

    test_df = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
            "c": np.random.rand(10),
        }
    )
    df = md.DataFrame(test_df, chunk_size=5)

    dir_name = f"hdfs://localhost:8020{TEST_DIR}/test_to_parquet/"
    hdfs.mkdir(dir_name)
    df.to_parquet(dir_name).execute()

    result = md.read_parquet(dir_name).to_pandas()
    pd.testing.assert_frame_equal(result.reset_index(drop=True), test_df)

    # test wildcard
    dir_name = f"hdfs://localhost:8020{TEST_DIR}/test_to_parquet2/*.parquet"
    hdfs.mkdir(dir_name.rsplit("/", 1)[0])
    df.to_parquet(dir_name).execute()

    result = md.read_parquet(dir_name.rsplit("/", 1)[0]).to_pandas()
    pd.testing.assert_frame_equal(result.reset_index(drop=True), test_df)
