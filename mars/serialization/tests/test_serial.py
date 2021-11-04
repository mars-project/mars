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

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

try:
    import pyarrow as pa
except ImportError:
    pa = None
try:
    import scipy.sparse as sps
except ImportError:
    sps = None

from ...lib.sparse import SparseMatrix
from ...tests.core import require_cupy, require_cudf
from ...utils import lazy_import
from .. import serialize, deserialize

cupy = lazy_import("cupy", globals=globals())
cudf = lazy_import("cudf", globals=globals())


class CustomList(list):
    pass


@pytest.mark.parametrize(
    "val",
    [
        False,
        123,
        3.567,
        3.5 + 4.3j,
        b"abcd",
        "abcd",
        ["uvw", ("mno", "sdaf"), 4, 6.7],
        CustomList([3, 4, CustomList([5, 6])]),
        {"abc": 5.6, "def": [3.4]},
        OrderedDict([("abcd", 5.6)]),
    ],
)
def test_core(val):
    deserialized = deserialize(*serialize(val))
    assert type(val) == type(deserialized)
    assert val == deserialized


def test_nested_list():
    val = ["a" * 100] * 100
    val[0] = val
    deserialized = deserialize(*serialize(val))
    assert deserialized[0] is deserialized
    assert val[1:] == deserialized[1:]


class KeyedDict(dict):
    def _skeys(self):
        return set(k for k in self.keys() if isinstance(k, str))

    def __hash__(self):
        return hash(frozenset(self._skeys()))

    def __eq__(self, other: "KeyedDict"):
        return self._skeys() == other._skeys()


def test_nested_dict():
    val = {i: "b" * 100 for i in range(10)}
    val[0] = val
    deserialized = deserialize(*serialize(val))
    assert deserialized[0] is deserialized

    val = KeyedDict(abcd="efgh")
    val[val] = val
    deserialized = deserialize(*serialize(val))
    assert deserialized[val] is deserialized


class DictWithoutInitArgs(dict):
    # dict inheritance without args in __init__
    def __init__(self):
        super().__init__()


def test_dict_without_init_args():
    val = DictWithoutInitArgs()
    val["a"] = "b"
    deserialized = deserialize(*serialize(val))
    assert deserialized == val


@pytest.mark.parametrize(
    "val",
    [
        np.array(np.random.rand(100, 100)),
        np.array(np.random.rand(100, 100).T),
        np.array(["a", "bcd", None]),
    ],
)
def test_numpy(val):
    deserialized = deserialize(*serialize(val))
    assert type(val) == type(deserialized)
    np.testing.assert_equal(val, deserialized)
    if val.flags.f_contiguous:
        assert deserialized.flags.f_contiguous


def test_pandas():
    val = pd.Series([1, 2, 3, 4])
    pd.testing.assert_series_equal(val, deserialize(*serialize(val)))

    val = pd.DataFrame(
        {
            "a": np.random.rand(1000),
            "b": np.random.choice(list("abcd"), size=(1000,)),
            "c": np.random.randint(0, 100, size=(1000,)),
        }
    )
    pd.testing.assert_frame_equal(val, deserialize(*serialize(val)))


@pytest.mark.skipif(pa is None, reason="need pyarrow to run the cases")
def test_arrow():
    test_df = pd.DataFrame(
        {
            "a": np.random.rand(1000),
            "b": np.random.choice(list("abcd"), size=(1000,)),
            "c": np.random.randint(0, 100, size=(1000,)),
        }
    )
    test_vals = [
        pa.RecordBatch.from_pandas(test_df),
        pa.Table.from_pandas(test_df),
    ]
    for val in test_vals:
        deserialized = deserialize(*serialize(val))
        assert type(val) is type(deserialized)
        np.testing.assert_equal(val, deserialized)


@pytest.mark.parametrize(
    "np_val",
    [np.random.rand(100, 100), np.random.rand(100, 100).T],
)
@require_cupy
def test_cupy(np_val):
    val = cupy.array(np_val)
    deserialized = deserialize(*serialize(val))
    assert type(val) is type(deserialized)
    cupy.testing.assert_array_equal(val, deserialized)


@require_cudf
def test_cudf():
    raw_df = pd.DataFrame(
        {
            "a": np.random.rand(1000),
            "b": np.random.choice(list("abcd"), size=(1000,)),
            "c": np.random.randint(0, 100, size=(1000,)),
        }
    )
    test_df = cudf.DataFrame(raw_df)
    cudf.testing.assert_frame_equal(test_df, deserialize(*serialize(test_df)))

    raw_df.columns = pd.MultiIndex.from_tuples([("a", "a"), ("a", "b"), ("b", "c")])
    test_df = cudf.DataFrame(raw_df)
    cudf.testing.assert_frame_equal(test_df, deserialize(*serialize(test_df)))


@pytest.mark.skipif(sps is None, reason="need scipy to run the test")
def test_scipy_sparse():
    val = sps.random(100, 100, 0.1, format="csr")
    deserial = deserialize(*serialize(val))
    assert (val != deserial).nnz == 0


@pytest.mark.skipif(sps is None, reason="need scipy to run the test")
def test_mars_sparse():
    val = SparseMatrix(sps.random(100, 100, 0.1, format="csr"))
    deserial = deserialize(*serialize(val))
    assert (val.spmatrix != deserial.spmatrix).nnz == 0
