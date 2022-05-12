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

from ... import dataframe as md
from ... import tensor as mt
from ...tests.core import require_cudf, require_cupy
from ...utils import lazy_import

cupy = lazy_import("cupy")
cudf = lazy_import("cudf")


def test_dataframe_initializer(setup):
    # from tensor
    raw = np.random.rand(100, 10)
    tensor = mt.tensor(raw, chunk_size=7)
    r = md.DataFrame(tensor)
    result = r.execute().fetch()
    pd.testing.assert_frame_equal(result, pd.DataFrame(raw))

    r = md.DataFrame(tensor, chunk_size=13)
    result = r.execute().fetch()
    pd.testing.assert_frame_equal(result, pd.DataFrame(raw))

    # from Mars dataframe
    raw = pd.DataFrame(np.random.rand(100, 10), columns=list("ABCDEFGHIJ"))
    df = md.DataFrame(raw, chunk_size=15) * 2
    r = md.DataFrame(df, num_partitions=11)
    result = r.execute().fetch()
    pd.testing.assert_frame_equal(result, raw * 2)

    # from tileable dict
    raw_dict = {
        "C": np.random.choice(["u", "v", "w"], size=(100,)),
        "A": pd.Series(np.random.rand(100)),
        "B": np.random.randint(0, 10, size=(100,)),
    }
    m_dict = raw_dict.copy()
    m_dict["A"] = md.Series(m_dict["A"])
    m_dict["B"] = mt.tensor(m_dict["B"])
    r = md.DataFrame(m_dict, columns=list("ABC"))
    result = r.execute().fetch()
    pd.testing.assert_frame_equal(result, pd.DataFrame(raw_dict, columns=list("ABC")))

    r = md.DataFrame({"a": [mt.tensor([1, 2, 3]).sum() + 1]})
    result = r.execute().fetch()
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [7]}))

    # from tileable list
    raw_list = [
        np.random.choice(["u", "v", "w"], size=(3,)),
        pd.Series(np.random.rand(3)),
        np.random.randint(0, 10, size=(3,)),
    ]
    m_list = raw_list.copy()
    m_list[1] = md.Series(m_list[1])
    m_list[2] = mt.tensor(m_list[2])
    r = md.DataFrame(m_list, columns=list("ABC"))
    result = r.execute(extra_config={"check_dtypes": False}).fetch()
    pd.testing.assert_frame_equal(result, pd.DataFrame(raw_list, columns=list("ABC")))

    # from raw pandas initializer
    raw = pd.DataFrame(np.random.rand(100, 10), columns=list("ABCDEFGHIJ"))
    r = md.DataFrame(raw, num_partitions=10)
    result = r.execute().fetch()
    pd.testing.assert_frame_equal(result, raw)

    # from mars series
    raw_s = np.random.rand(100)
    s = md.Series(raw_s, chunk_size=20)
    r = md.DataFrame(s, num_partitions=10)
    result = r.execute().fetch()
    pd.testing.assert_frame_equal(result, pd.DataFrame(raw_s))

    # test check instance
    r = r * 2
    assert isinstance(r, md.DataFrame)


@require_cudf
@require_cupy
def test_dataframe_gpu_initializer(setup_gpu):
    # from raw cudf initializer
    raw = cudf.DataFrame(cupy.random.rand(100, 10), columns=list("ABCDEFGHIJ"))
    r = md.DataFrame(raw, chunk_size=13)
    result = r.execute().fetch()
    pd.testing.assert_frame_equal(result.to_pandas(), raw.to_pandas())

    raw = cupy.random.rand(100, 10)
    r = md.DataFrame(raw, columns=list("ABCDEFGHIJ"), chunk_size=13)
    result = r.execute().fetch()
    expected = cudf.DataFrame(raw, columns=list("ABCDEFGHIJ"))
    pd.testing.assert_frame_equal(result.to_pandas(), expected.to_pandas())


def test_series_initializer(setup):
    # from tensor
    raw = np.random.rand(100)
    tensor = mt.tensor(raw, chunk_size=7)
    r = md.Series(tensor)
    result = r.execute().fetch()
    pd.testing.assert_series_equal(result, pd.Series(raw))

    r = md.Series(tensor, chunk_size=13)
    result = r.execute().fetch()
    pd.testing.assert_series_equal(result, pd.Series(raw))

    # from index
    raw = np.arange(100)
    np.random.shuffle(raw)
    raw = pd.Index(raw, name="idx_name")
    idx = md.Index(raw, chunk_size=7)
    r = md.Series(idx)
    result = r.execute().fetch()
    pd.testing.assert_series_equal(result, pd.Series(raw))

    # from Mars series
    raw = pd.Series(np.random.rand(100), name="series_name")
    ms = md.Series(raw, chunk_size=15) * 2
    r = md.Series(ms, num_partitions=11)
    result = r.execute().fetch()
    pd.testing.assert_series_equal(result, raw * 2)

    # from raw pandas initializer
    raw = pd.Series(np.random.rand(100), name="series_name")
    r = md.Series(raw, num_partitions=10)
    result = r.execute().fetch()
    pd.testing.assert_series_equal(result, raw)

    # test check instance
    r = r * 2
    assert isinstance(r, md.Series)


@require_cudf
@require_cupy
def test_series_gpu_initializer(setup_gpu):
    # from raw cudf initializer
    raw = cudf.Series(cupy.random.rand(100), name="a")
    r = md.Series(raw, chunk_size=13)
    result = r.execute().fetch()
    pd.testing.assert_series_equal(result.to_pandas(), raw.to_pandas())

    raw = cupy.random.rand(100)
    r = md.Series(raw, name="a", chunk_size=13)
    result = r.execute().fetch()
    expected = cudf.Series(raw, name="a")
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


def test_index_initializer(setup):
    # from tensor
    raw = np.arange(100)
    np.random.shuffle(raw)
    tensor = mt.tensor(raw)
    r = md.Index(tensor, chunk_size=7)
    result = r.execute().fetch()
    pd.testing.assert_index_equal(result, pd.Index(raw))

    # from Mars index
    raw = np.arange(100)
    np.random.shuffle(raw)
    idx = md.Index(raw, chunk_size=7)
    r = md.Index(idx, num_partitions=11)
    result = r.execute().fetch()
    pd.testing.assert_index_equal(result, pd.Index(raw))

    # from pandas initializer
    raw = np.arange(100)
    np.random.shuffle(raw)
    raw_ser = pd.Series(raw, name="series_name")
    r = md.Index(raw_ser, chunk_size=7)
    result = r.execute().fetch()
    pd.testing.assert_index_equal(result, pd.Index(raw_ser))

    raw_idx = pd.Index(raw, name="idx_name")
    r = md.Index(raw_idx, num_partitions=10)
    result = r.execute().fetch()
    pd.testing.assert_index_equal(result, pd.Index(raw_idx))


@require_cudf
@require_cupy
def test_index_gpu_initializer(setup_gpu):
    # from raw cudf initializer
    raw = cudf.Index(cupy.random.rand(100), name="a")
    r = md.Index(raw, chunk_size=13)
    result = r.execute().fetch()
    pd.testing.assert_index_equal(result.to_pandas(), raw.to_pandas())

    raw = cupy.random.rand(100)
    r = md.Index(raw, name="a", chunk_size=13)
    result = r.execute().fetch()
    expected = cudf.Index(raw, name="a")
    pd.testing.assert_index_equal(result.to_pandas(), expected.to_pandas())
