# Copyright 2022 XProbe Inc.
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

from typing import Dict, Any, Set, List, Union

import pytest

from ..input_column_selector import InputColumnSelector
from ......core import TileableData, ENTITY_TYPE
from ......core.operand import Operand
from ......dataframe import DataFrame, Series
from ......tensor import tensor


class MockOperand(Operand):
    _mock_input: TileableData = DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6)}).data

    @property
    def inputs(self) -> List[Union[ENTITY_TYPE]]:
        return [self._mock_input]

    @classmethod
    def get_mock_input(cls) -> TileableData:
        return cls._mock_input


class MockEntityData(TileableData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._op = MockOperand()


def test_register():
    def _select_input_columns(
        tileable_data: TileableData, required_cols: Set[Any]
    ) -> Dict[TileableData, Set[Any]]:
        return {}

    InputColumnSelector.register(MockOperand, _select_input_columns)
    mock_data = MockEntityData()
    assert InputColumnSelector.select(mock_data, {"foo"}) == {}

    # unregister
    InputColumnSelector.unregister(MockOperand)
    assert InputColumnSelector.select(mock_data, {"foo"}) == {
        MockOperand.get_mock_input(): {"foo", "bar"}
    }


def test_df_groupby_agg():
    df: DataFrame = DataFrame(
        {
            "foo": (1, 1, 2, 2),
            "bar": (3, 4, 3, 4),
            "baz": (5, 6, 7, 8),
            "qux": (9, 10, 11, 12),
        }
    )

    s = df.groupby(by="foo")["baz"].sum()
    input_columns = InputColumnSelector.select(s.data, {"baz"})
    assert len(input_columns) == 1
    assert df.data in input_columns
    assert input_columns[df.data] == {"foo", "baz"}

    s = df.groupby(by=["foo", "bar"]).sum()
    input_columns = InputColumnSelector.select(s.data, {"baz"})
    assert len(input_columns) == 1
    assert df.data in input_columns
    assert input_columns[df.data] == {"foo", "bar", "baz"}

    s = df.groupby(by="foo").agg(["sum", "max"])
    input_columns = InputColumnSelector.select(s.data, {"baz"})
    assert len(input_columns) == 1
    assert df.data in input_columns
    assert input_columns[df.data] == {"foo", "baz"}

    s = df.groupby(by="foo")["bar", "baz"].agg(["sum", "max"])
    input_columns = InputColumnSelector.select(s.data, {"baz"})
    assert len(input_columns) == 1
    assert df.data in input_columns
    assert input_columns[df.data] == {"foo", "bar", "baz"}

    s = df.groupby(by="foo").agg(new_bar=("bar", "sum"), new_baz=("baz", "sum"))
    input_columns = InputColumnSelector.select(s.data, {"new_bar"})
    assert len(input_columns) == 1
    assert df.data in input_columns
    assert input_columns[df.data] == {"foo", "bar", "baz"}


@pytest.mark.skip(reason="group by index is not supported yet")
def test_df_groupby_index_agg():
    df: DataFrame = DataFrame({"foo": (1, 1, 3), "bar": (4, 5, 6)})
    df = df.set_index("foo")
    s = df.groupby(by="foo").sum()
    input_columns = InputColumnSelector.select(s.data, {"bar"})
    assert len(input_columns) == 1
    assert df.data in input_columns
    assert input_columns[df.data] == {"bar"}


def test_df_merge():
    left: DataFrame = DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6), 1: (7, 8, 9)})
    right = DataFrame({"foo": (1, 2), "bar": (4, 5), "baz": (5, 8), 1: (7, 8)})

    joined = left.merge(right, on=["foo"])

    input_columns = InputColumnSelector.select(joined.data, {"foo"})
    assert left.data in input_columns
    assert input_columns[left.data] == {"foo"}
    assert right.data in input_columns
    assert input_columns[right.data] == {"foo"}

    input_columns = InputColumnSelector.select(joined.data, {"foo", "baz"})
    assert left.data in input_columns
    assert input_columns[left.data] == {"foo"}
    assert right.data in input_columns
    assert input_columns[right.data] == {"foo", "baz"}

    input_columns = InputColumnSelector.select(joined.data, {"foo", "1_x"})
    assert left.data in input_columns
    assert input_columns[left.data] == {"foo", 1}
    assert right.data in input_columns
    assert input_columns[right.data] == {"foo", 1}

    joined = left.merge(right, on=["foo", "bar"])
    input_columns = InputColumnSelector.select(joined.data, {"baz"})
    assert left.data in input_columns
    assert input_columns[left.data] == {"foo", "bar"}
    assert right.data in input_columns
    assert input_columns[right.data] == {"foo", "bar", "baz"}

    joined = left.merge(right, on=["foo", "bar"])
    input_columns = InputColumnSelector.select(joined.data, {"1_x", "1_y"})
    assert left.data in input_columns
    assert input_columns[left.data] == {"foo", "bar", 1}
    assert right.data in input_columns
    assert input_columns[right.data] == {"foo", "bar", 1}


def test_df_merge_on_index():
    left: DataFrame = DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6), 1: (7, 8, 9)})
    left = left.set_index("foo")
    right = DataFrame({"foo": (1, 2), "bar": (4, 5), "baz": (5, 8), 1: (7, 8)})
    right = right.set_index("foo")

    # join on index
    joined = left.merge(right, on="foo")
    input_columns = InputColumnSelector.select(joined.data, {"baz"})
    assert left.data in input_columns
    assert input_columns[left.data] == set()
    assert right.data in input_columns
    assert input_columns[right.data] == {"baz"}

    # left_on is an index and right_on is a column
    joined = left.merge(right, left_on="foo", right_on="bar")
    input_columns = InputColumnSelector.select(joined.data, {"baz"})
    assert left.data in input_columns
    assert input_columns[left.data] == set()
    assert right.data in input_columns
    assert input_columns[right.data] == {"bar", "baz"}

    # left_on is a column and right_on is an index
    joined = left.merge(right, left_on="bar", right_on="foo")
    input_columns = InputColumnSelector.select(joined.data, {"baz"})
    assert left.data in input_columns
    assert input_columns[left.data] == {"bar"}
    assert right.data in input_columns
    assert input_columns[right.data] == {"baz"}


def test_df_arithmatic_ops():
    def add(x, y):
        return x + y

    def sub(x, y):
        return x - y

    def mul(x, y):
        return x * y

    def div(x, y):
        return x / y

    ops = (add, sub, mul, div)
    df1: DataFrame = DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6)})
    df2: DataFrame = DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6)})

    for op in ops:
        res: DataFrame = op(df1, 1)
        input_columns = InputColumnSelector.select(res.data, {"foo"})
        assert len(input_columns) == 1
        assert res.data.inputs[0] in input_columns
        assert input_columns[res.data.inputs[0]] == {"foo"}

    for op in ops:
        res: DataFrame = op(df1, df2)
        input_columns = InputColumnSelector.select(res.data, {"foo"})
        assert len(input_columns) == 2
        assert res.data.inputs[0] in input_columns
        assert input_columns[res.data.inputs[0]] == {"foo"}
        assert res.data.inputs[1] in input_columns
        assert input_columns[res.data.inputs[1]] == {"foo"}


def test_df_setitem():
    df: DataFrame = DataFrame(
        {
            "foo": (1, 1, 2, 2),
            "bar": (3, 4, 3, 4),
            "baz": (5, 6, 7, 8),
            "qux": (9, 10, 11, 12),
        }
    )

    # scaler
    df[4] = 13
    input_columns = InputColumnSelector.select(df.data, {"foo"})
    assert len(input_columns) == 1
    assert df.data.inputs[0] in input_columns
    assert input_columns[df.data.inputs[0]] == {"foo"}

    # scaler tensor
    df[5] = tensor()
    input_columns = InputColumnSelector.select(df.data, {"foo"})
    assert len(input_columns) == 1
    assert df.data.inputs[0] in input_columns
    assert input_columns[df.data.inputs[0]] == {"foo"}

    # tensor
    df[6] = tensor([13, 14, 15, 16])
    input_columns = InputColumnSelector.select(df.data, {"foo"})
    assert len(input_columns) == 2
    assert df.data.inputs[0] in input_columns
    assert input_columns[df.data.inputs[0]] == {"foo"}
    assert df.data.inputs[1] in input_columns
    assert input_columns[df.data.inputs[1]] == {None}

    # series
    df[7] = Series([13, 14, 15, 16])
    input_columns = InputColumnSelector.select(df.data, {"foo"})
    assert len(input_columns) == 2
    assert df.data.inputs[0] in input_columns
    assert input_columns[df.data.inputs[0]] == {"foo"}
    assert df.data.inputs[1] in input_columns
    assert input_columns[df.data.inputs[1]] == {None}

    # dataframe
    df[[8, 9]] = df[["foo", "bar"]]
    input_columns = InputColumnSelector.select(df.data, {8})
    assert len(input_columns) == 2
    assert df.data.inputs[0] in input_columns
    assert input_columns[df.data.inputs[0]] == set()
    assert df.data.inputs[1] in input_columns
    assert input_columns[df.data.inputs[1]] == {"foo", "bar"}


def test_select_all():
    df: DataFrame = DataFrame(
        {
            "foo": (1, 1, 2, 2),
            "bar": (3, 4, 3, 4),
            "baz": (5, 6, 7, 8),
            "qux": (9, 10, 11, 12),
        }
    )
    head = df.head()
    input_columns = InputColumnSelector.select(head.data, {"foo"})
    assert len(input_columns) == 1
    assert head.data.inputs[0] in input_columns
    assert input_columns[head.data.inputs[0]] == {"foo", "bar", "baz", "qux"}


def test_getitem():
    df: DataFrame = DataFrame(
        {
            "foo": (1, 1, 2, 2),
            "bar": (3, 4, 3, 4),
            "baz": (5, 6, 7, 8),
            "qux": (9, 10, 11, 12),
        }
    )

    getitem = df[df["foo"] == 1]
    input_columns = InputColumnSelector.select(getitem.data, {"foo"})
    assert input_columns[getitem.data.inputs[0]] == {"foo", "bar", "baz", "qux"}

    getitem = df["foo"]
    input_columns = InputColumnSelector.select(getitem.data, {"foo"})
    assert input_columns[getitem.data.inputs[0]] == {"foo"}
