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

try:
    import pyarrow as pa
except ImportError:
    pa = None

from ...config import option_context
from ...core import enter_mode
from .. import ArrowStringDtype, ArrowStringArray, ArrowListDtype, ArrowListArray
from ..arrays import _use_bool_any_all
from ..utils import arrow_table_to_pandas_dataframe


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_arrow_dtype():
    s = pa.array(["a", "b"])
    assert list(ArrowStringDtype().__from_arrow__(s)) == list(ArrowStringArray(s))

    assert ArrowStringDtype() == ArrowStringDtype.construct_from_string("Arrow[string]")

    assert ArrowListDtype(
        ArrowListDtype("string")
    ) == ArrowListDtype.construct_from_string("Arrow[List[string]]")

    assert repr(ArrowListDtype(np.int8)) == "Arrow[List[int8]]"

    with pytest.raises(TypeError):
        ArrowListDtype.construct_from_string("Arrow[string]")

    assert ArrowListDtype.is_dtype("Arrow[List[uint8]]") is True
    assert ArrowListDtype.is_dtype("List[int8]") is False
    assert ArrowListDtype.is_dtype(ArrowStringDtype()) is False

    assert ArrowListDtype(np.int8) != ArrowStringDtype()
    assert ArrowListDtype(np.int8).kind == np.dtype(object).kind

    assert ArrowListDtype(np.int8).arrow_type == pa.list_(pa.int8())


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_arrow_string_array_creation():
    # create from pandas Series
    series = pd.Series(["a", "bc", "de"])
    array = ArrowStringArray(series)
    assert isinstance(array._arrow_array, pa.ChunkedArray)

    if pd.__version__ >= "1.0.0":
        # test create from StringArray which occurs in pandas 1.0
        s = pd.arrays.StringArray(np.array(["a", "bc", "de"], dtype=object))
        array = ArrowStringArray(s)
        assert isinstance(array._arrow_array, pa.ChunkedArray)

    # create from list
    lst = ["a", "bc", "de"]
    array = ArrowStringArray(lst)
    assert isinstance(array._arrow_array, pa.ChunkedArray)

    # create from pyarrow Array
    a = pa.array(["a", "bc", "de"])
    array = ArrowStringArray(a)
    assert isinstance(array._arrow_array, pa.ChunkedArray)

    # create from ArrowStringArray
    array2 = ArrowStringArray(array)
    assert isinstance(array2._arrow_array, pa.ChunkedArray)

    # test copy
    arrow_array = array2._arrow_array
    array3 = ArrowStringArray(arrow_array, copy=True)
    assert array3._arrow_array is not arrow_array

    # test from_scalars
    array = ArrowStringArray.from_scalars([1, 2])
    assert isinstance(array._arrow_array, pa.ChunkedArray)
    assert isinstance(array._arrow_array.chunks[0], pa.StringArray)

    # test _from_sequence
    array = ArrowStringArray._from_sequence(["a", "b", "cc"])
    assert isinstance(array._arrow_array, pa.ChunkedArray)

    # test _from_sequence_of_strings
    array = ArrowStringArray._from_sequence_of_strings(["a", "b"])
    assert isinstance(array._arrow_array, pa.ChunkedArray)


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_arrow_list_array_creation():
    # create from pandas Series
    series = pd.Series([["a", "b"], ["c"], ["d", "e"]])
    array = ArrowListArray(series)
    assert isinstance(array.dtype, ArrowListDtype)
    assert isinstance(array.dtype.value_type, ArrowStringDtype)
    assert isinstance(array._arrow_array, pa.ChunkedArray)

    # create from list
    lst = [["a"], ["b", "c"], ["d", "e"]]
    array = ArrowListArray(lst)
    assert isinstance(array.dtype, ArrowListDtype)
    assert isinstance(array.dtype.value_type, ArrowStringDtype)
    assert isinstance(array._arrow_array, pa.ChunkedArray)

    # create from pyarrow Array
    a = pa.array([[1.0], [2.0, 3.0], [4.0]])
    array = ArrowListArray(a)
    assert isinstance(array.dtype, ArrowListDtype)
    assert array.dtype.value_type == np.float64
    assert isinstance(array._arrow_array, pa.ChunkedArray)

    # create from ArrowListArray
    array2 = ArrowListArray(array)
    assert isinstance(array2._arrow_array, pa.ChunkedArray)

    # test _from_sequence
    array = ArrowListArray._from_sequence([[1, 2], [3, 4], [5]])
    assert isinstance(array.dtype, ArrowListDtype)
    assert array.dtype.value_type == np.int64
    assert isinstance(array._arrow_array, pa.ChunkedArray)

    # test pandas_only
    with option_context({"dataframe.arrow_array.pandas_only": True}):
        array = ArrowListArray._from_sequence([[1, 2], [3, 4], [5]])
        assert isinstance(array.dtype, ArrowListDtype)
        assert isinstance(array._ndarray, np.ndarray)

    # test pandas_only and in kernel mode
    with enter_mode(kernel=True), option_context(
        {"dataframe.arrow_array.pandas_only": True}
    ), pytest.raises(ImportError):
        ArrowListArray._from_sequence([[1, 2], [3, 4], [5]])


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_arrow_string_array_functions():
    lst = np.array(["abc", "de", "eee", "中文"], dtype=object)
    # leverage string array to get the right answer
    string_array = pd.arrays.StringArray(lst)
    has_na_arrow_array = ArrowStringArray(["abc", None, "eee", "中文"])
    has_na_string_array = pd.arrays.StringArray(
        np.array(["abc", pd.NA, "eee", "中文"], dtype=object)
    )

    for pandas_only in [False, True]:
        with option_context({"dataframe.arrow_array.pandas_only": pandas_only}):
            arrow_array = ArrowStringArray(lst)

            # getitem, scalar
            assert arrow_array[1] == string_array[1]
            assert arrow_array[-1] == string_array[-1]
            # getitem, slice
            assert list(arrow_array[:2]) == list(string_array[:2])
            assert list(arrow_array[1:-1]) == list(string_array[1:-1])
            assert list(arrow_array[::2]) == list(string_array[::2])
            # getitem, boolean index
            cond = np.array([len(c) > 2 for c in lst])
            assert list(arrow_array[cond]) == list(string_array[cond])
            # getitem, fancy index
            selection = [3, 1, 2]
            assert list(arrow_array[selection]) == list(string_array[selection])
            selection = [3, -1, 2, -4]
            assert list(arrow_array[selection]) == list(string_array[selection])
            selection = np.array([3, -1, 2, -4])
            assert list(arrow_array[selection]) == list(string_array[selection])

            # setitem
            arrow_array2 = arrow_array.copy()
            string_array2 = string_array.copy()
            arrow_array2[0] = "ss"
            string_array2[0] = "ss"
            assert list(arrow_array2) == list(string_array2)
            arrow_array2[1:3] = ["ss1", "ss2"]
            string_array2[1:3] = ["ss1", "ss2"]
            assert list(arrow_array2) == list(string_array2)
            arrow_array2[1:3] = arrow_array2[2:4]
            string_array2[1:3] = string_array2[2:4]
            assert list(arrow_array2) == list(string_array2)
            arrow_array2[2:] = pd.Series(["ss3", "ss4"])
            string_array2[2:] = pd.Series(["ss3", "ss4"])
            assert list(arrow_array2) == list(string_array2)
            with pytest.raises(ValueError):
                arrow_array2[0] = ["a", "b"]
            arrow_array2[-1] = None
            string_array2[-1] = None
            assert list(arrow_array2)[:-1] == list(string_array2)[:-1]
            assert pd.isna(list(arrow_array2)[-1]) is True
            with pytest.raises(ValueError):
                arrow_array2[0] = 2
            with pytest.raises(ValueError):
                arrow_array2[:2] = [1, 2]

            # test to_numpy
            np.testing.assert_array_equal(
                arrow_array.to_numpy(), string_array.to_numpy()
            )
            np.testing.assert_array_equal(
                arrow_array.to_numpy(copy=True), string_array.to_numpy(copy=True)
            )
            np.testing.assert_array_equal(
                has_na_arrow_array.to_numpy(copy=True, na_value="ss"),
                has_na_string_array.to_numpy(copy=True, na_value="ss"),
            )

            # test fillna
            arrow_array3 = has_na_arrow_array.fillna("filled")
            string_array3 = has_na_string_array.fillna("filled")
            assert list(arrow_array3) == list(string_array3)

            # test astype
            arrow_array4 = ArrowStringArray(["1", "10", "100"])
            # leverage string array to get the right answer
            string_array4 = pd.arrays.StringArray(
                np.array(["1", "10", "100"], dtype=object)
            )
            np.testing.assert_array_equal(
                arrow_array4.astype(np.int64), string_array4.astype(np.int64)
            )
            np.testing.assert_almost_equal(
                arrow_array4.astype(float), string_array4.astype(float)
            )
            assert list(arrow_array4.astype(ArrowStringDtype(), copy=False)) == list(
                string_array4.astype(pd.StringDtype(), copy=False)
            )
            assert list(arrow_array4.astype(ArrowStringDtype(), copy=True)) == list(
                string_array4.astype(pd.StringDtype(), copy=True)
            )

            # test factorize
            codes, unique = arrow_array.factorize()
            codes2, unique2 = string_array.factorize()
            assert list(codes) == list(codes2)
            assert list(unique) == list(unique2)

            # test nbytes
            assert arrow_array.nbytes < pd.Series(
                string_array.astype(object)
            ).memory_usage(deep=True, index=False)

            # test memory_usage
            if pandas_only:
                assert arrow_array.memory_usage(deep=False) == pd.Series(
                    string_array
                ).memory_usage(index=False)
            else:
                assert arrow_array.memory_usage(deep=True) == arrow_array.nbytes

            # test unique
            assert arrow_array.unique() == pd.Series(string_array).unique()
            arrow_array2 = arrow_array.copy()
            arrow_array2._force_use_pandas = True
            assert arrow_array2.unique() == pd.Series(string_array).unique()

            # test isna
            np.testing.assert_array_equal(
                has_na_arrow_array.isna(), has_na_string_array.isna()
            )
            has_na_arrow_array2 = has_na_arrow_array.copy()
            has_na_arrow_array2._force_use_pandas = True
            np.testing.assert_array_equal(
                has_na_arrow_array2.isna(), has_na_string_array.isna()
            )

            # test take
            assert list(arrow_array.take([1, 2, -1])) == list(
                string_array.take([1, 2, -1])
            )
            assert list(
                arrow_array.take([1, 2, -1], allow_fill=True).fillna("aa")
            ) == list(string_array.take([1, 2, -1], allow_fill=True).fillna("aa"))
            assert list(
                arrow_array.take([1, 2, -1], allow_fill=True, fill_value="aa")
            ) == list(string_array.take([1, 2, -1], allow_fill=True, fill_value="aa"))

            # test shift
            assert list(arrow_array.shift(2, fill_value="aa")) == list(
                string_array.shift(2, fill_value="aa")
            )

            # test value_counts
            assert list(arrow_array.value_counts()) == list(string_array.value_counts())
            assert list(has_na_arrow_array.value_counts(dropna=True)) == list(
                has_na_string_array.value_counts(dropna=True)
            )

            # test all any
            assert arrow_array.all() == string_array.all()
            assert arrow_array.any() == string_array.any()

            # test arithmetic
            assert list(arrow_array + "s") == list(string_array + "s")
            assert list((arrow_array + has_na_arrow_array).fillna("ss")) == list(
                (string_array + has_na_string_array).fillna("ss")
            )

            # test comparison
            np.testing.assert_array_equal(arrow_array < "s", string_array < "s")
            pd.testing.assert_series_equal(
                pd.Series(arrow_array < has_na_arrow_array),
                pd.Series(string_array < has_na_string_array),
            )

            # test repr
            assert "ArrowStringArray" in repr(arrow_array)

            # test concat empty
            arrow_array5 = ArrowStringArray(pa.chunked_array([], type=pa.string()))
            concatenated = ArrowStringArray._concat_same_type(
                [arrow_array5, arrow_array5]
            )
            if not pandas_only:
                assert len(concatenated._arrow_array.chunks) == 1
            pd.testing.assert_series_equal(
                pd.Series(arrow_array5), pd.Series(concatenated)
            )


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_arrow_list_functions():
    lst = np.array([["a, bc"], ["de"], ["e", "ee"], ["中文", "中文2"]], dtype=object)
    has_na_lst = lst.copy()
    has_na_lst[1] = None

    for pandas_only in [False, True]:
        with option_context({"dataframe.arrow_array.pandas_only": pandas_only}):
            arrow_array = ArrowListArray(lst)
            has_na_arrow_array = ArrowListArray(has_na_lst)

            # getitem, scalar
            assert arrow_array[1] == lst[1]
            assert list(arrow_array[-1]) == lst[-1]
            # getitem, slice
            np.testing.assert_array_equal(arrow_array[:2].to_numpy(), lst[:2])

            # setitem
            arrow_array2 = arrow_array.copy()
            lst2 = lst.copy()
            for s in [["ss"], pd.Series(["ss"])]:
                arrow_array2[0] = s
                lst2[0] = ["ss"]
                np.testing.assert_array_equal(arrow_array2.to_numpy(), lst2)
            arrow_array2[0] = None
            lst2[0] = None
            np.testing.assert_array_equal(arrow_array2.to_numpy(), lst2)
            with pytest.raises(ValueError):
                # must set list like object
                arrow_array2[0] = "ss"

            # test to_numpy
            np.testing.assert_array_equal(arrow_array.to_numpy(), lst)
            np.testing.assert_array_equal(arrow_array.to_numpy(copy=True), lst)
            np.testing.assert_array_equal(
                has_na_arrow_array.to_numpy(na_value=1),
                pd.Series(has_na_lst).fillna(1).to_numpy(),
            )

            # test fillna
            if not pandas_only:
                arrow_array3 = has_na_arrow_array.fillna(lst[1])
                np.testing.assert_array_equal(arrow_array3.to_numpy(), lst)

            # test astype
            with pytest.raises(TypeError):
                arrow_array.astype(np.int64)
            with pytest.raises(TypeError):
                arrow_array.astype(ArrowListDtype(np.int64))
            arrow_array4 = ArrowListArray([[1, 2], [3]])
            expected = np.array([["1", "2"], ["3"]], dtype=object)
            np.testing.assert_array_equal(
                arrow_array4.astype(ArrowListDtype(str)), expected
            )
            np.testing.assert_array_equal(
                arrow_array4.astype(ArrowListDtype(arrow_array4.dtype)), arrow_array4
            )
            np.testing.assert_array_equal(
                arrow_array4.astype(ArrowListDtype(arrow_array4.dtype), copy=False),
                arrow_array4,
            )

            # test nbytes
            assert arrow_array.nbytes < pd.Series(lst).memory_usage(deep=True)

            # test memory_usage
            if not pandas_only:
                assert arrow_array.memory_usage(deep=True) == arrow_array.nbytes

            # test isna
            np.testing.assert_array_equal(
                has_na_arrow_array.isna(), pd.Series(has_na_lst).isna()
            )

            # test take
            assert list(arrow_array.take([1, 2, -1])) == list(
                pd.Series(lst).take([1, 2, -1])
            )

            # test shift
            assert (
                list(arrow_array.shift(2, fill_value=["aa"]))
                == [["aa"]] * 2 + lst[:-2].tolist()
            )

            # test all any
            if _use_bool_any_all:
                assert arrow_array.all() == pd.array(lst).all()
                assert arrow_array.any() == pd.array(lst).any()
            else:
                assert arrow_array.all() == lst.all()
                assert arrow_array.any() == lst.any()

            # test repr
            assert "ArrowListArray" in repr(arrow_array)

            # test concat empty
            arrow_array5 = ArrowListArray(
                pa.chunked_array([], type=pa.list_(pa.string()))
            )
            concatenated = ArrowListArray._concat_same_type(
                [arrow_array5, arrow_array5]
            )
            if not pandas_only:
                assert len(concatenated._arrow_array.chunks) == 1
            pd.testing.assert_series_equal(
                pd.Series(arrow_array5), pd.Series(concatenated)
            )


@pytest.mark.skipif(pa is None, reason="pyarrow not installed")
def test_to_pandas():
    rs = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "a": rs.rand(100),
            "b": ["s" + str(i) for i in rs.randint(100, size=100)],
            "c": [["ss0" + str(i), "ss1" + str(i)] for i in rs.randint(100, size=100)],
        }
    )

    batch_size = 15
    n_batch = len(df) // 15 + 1
    batches = [
        pa.RecordBatch.from_pandas(df[i * batch_size : (i + 1) * batch_size])
        for i in range(n_batch)
    ]
    table = pa.Table.from_batches(batches)

    df1 = arrow_table_to_pandas_dataframe(table, use_arrow_dtype=False)
    assert df1.dtypes.iloc[1] == np.dtype("O")
    assert df1.dtypes.iloc[2] == np.dtype("O")

    df2 = arrow_table_to_pandas_dataframe(table)
    assert df2.dtypes.iloc[1] == ArrowStringDtype()
    assert df2.dtypes.iloc[2] == ArrowListDtype(str)
    assert df2.memory_usage(deep=True).sum() < df.memory_usage(deep=True).sum()

    # test df method
    df4 = df2.groupby("b").sum()
    df4.index = df4.index.astype(object)
    expected = df.groupby("b").sum()
    pd.testing.assert_frame_equal(df4, expected)

    s = ("s" + df2["b"]).astype("string")
    expected = ("s" + df["b"]).astype("string")
    pd.testing.assert_series_equal(s, expected)

    s2 = df2["b"].str[:2]
    expected = df["b"].astype("string").str[:2]
    pd.testing.assert_series_equal(s2, expected)
