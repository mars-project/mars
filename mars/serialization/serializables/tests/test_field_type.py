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

from collections import namedtuple
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ....core import EntityData
from .. import FieldTypes


class MyClass(EntityData):
    __slots__ = ()

    @staticmethod
    def my_func():
        """
        Test function
        """


my_named_tuple = namedtuple("my_named_tuple", "a b")


fields_values = [
    # field_type, valid values, invalid values
    [FieldTypes.bool, [True, np.bool_(False)], [1]],
    [FieldTypes.int8, [8, np.int8(8)], [8.0]],
    [FieldTypes.int16, [16, np.int16(16)], [16.0]],
    [FieldTypes.int32, [32, np.int32(32)], [64.0]],
    [FieldTypes.uint8, [8, np.uint8(8)], [8.0]],
    [FieldTypes.uint16, [16, np.uint16(16)], [16.0]],
    [FieldTypes.uint32, [32, np.uint32(32)], [32.0]],
    [FieldTypes.uint64, [64, np.uint64(64)], [64.0]],
    [FieldTypes.float16, [16.0, np.float16(16)], [16]],
    [FieldTypes.float32, [32.0, np.float32(32)], [32]],
    [FieldTypes.float64, [64.0, np.float64(64)], [64]],
    [FieldTypes.complex64, [1 + 2j, np.complex64(1 + 2j)], [64]],
    [FieldTypes.complex128, [1 + 2j, np.complex128(1 + 2j)], [128]],
    [FieldTypes.bytes, [b"abc", np.bytes_("abc")], ["abc"]],
    [FieldTypes.string, ["abc", np.str_("abc")], [b"abc"]],
    [FieldTypes.ndarray, [np.array([1, 2, 3])], [object()]],
    [FieldTypes.dtype, [np.dtype(np.int32), pd.StringDtype()], [object()]],
    [FieldTypes.key, [MyClass()], [object()]],
    [FieldTypes.slice, [slice(1, 10), slice("a", "b")], [object()]],
    [FieldTypes.datetime, [datetime.now(), pd.Timestamp(0)], [object()]],
    [FieldTypes.timedelta, [timedelta(days=1), pd.Timedelta(days=1)], [object()]],
    [FieldTypes.tzinfo, [timezone.utc], [object()]],
    [FieldTypes.index, [pd.RangeIndex(10), pd.Index([1, 2])], [object()]],
    [FieldTypes.series, [pd.Series([1, 2, 3])], [object()]],
    [FieldTypes.dataframe, [pd.DataFrame({"a": [1, 2]})], [object()]],
    [FieldTypes.interval_array, [pd.arrays.IntervalArray([])], [object()]],
    [FieldTypes.function, [MyClass.my_func], [object()]],
    [FieldTypes.namedtuple, [my_named_tuple(a=1, b=2)], [tuple()]],
    [FieldTypes.reference(MyClass), [MyClass()], [object()]],
    [
        FieldTypes.tuple(FieldTypes.int64, ...),
        [tuple(), tuple([1, 2])],
        [list(), tuple([1, 2.0])],
    ],
    [
        FieldTypes.list(FieldTypes.int64, FieldTypes.float64),
        [[1, 1.0]],
        [tuple(), [1, 1]],
    ],
    [
        FieldTypes.dict(FieldTypes.string, FieldTypes.int64),
        [{"a": 1}],
        [{1: "a"}, {"a": 1.0}],
    ],
    [FieldTypes.any, [object()], []],
]


@pytest.mark.parametrize("field_type, valid_values, invalid_values", fields_values)
def test_field_type(field_type, valid_values, invalid_values):
    assert isinstance(field_type.type_name, str)
    assert isinstance(field_type.name, str)

    for valid_value in valid_values:
        field_type.validate(valid_value)

    for invalid_value in invalid_values:
        with pytest.raises(TypeError):
            field_type.validate(invalid_value)


def test_collction_field_error():
    with pytest.raises(ValueError):
        FieldTypes.tuple(FieldTypes.int64, FieldTypes.float32).validate(
            tuple([1, 3.0, 3.0])
        )


def test_field_name():
    assert FieldTypes.list().name == "List"
    assert (
        FieldTypes.list(FieldTypes.int64, FieldTypes.float32).name
        == "List[Int64, Float32]"
    )
    assert FieldTypes.tuple(FieldTypes.int8, ...).name == "Tuple[Int8, ...]"
    assert FieldTypes.tuple(FieldTypes.int8).name == "Tuple[Int8, ...]"
    assert FieldTypes.dict().name == "Dict"
    assert (
        FieldTypes.dict(FieldTypes.int8, FieldTypes.float64).name
        == "Dict[Int8, Float64]"
    )
