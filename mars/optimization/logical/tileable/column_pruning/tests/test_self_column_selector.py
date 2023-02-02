# Copyright 2022-2023 XProbe Inc.
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

from ......dataframe import DataFrame
from ..self_column_selector import SelfColumnSelector


def test_df_setitem():
    df = DataFrame({"foo": (1, 1, 3)})

    df["bar"] = [1, 2, 3]
    required_columns = SelfColumnSelector.select(df.data)
    assert required_columns == {"bar"}

    df[["baz", "qux"]] = 1, 2
    required_columns = SelfColumnSelector.select(df.data)
    assert required_columns == {"baz", "qux"}


def test_df_getitem():
    df = DataFrame({"foo": (1, 1, 3), "bar": (4, 5, 6)})

    getitem = df["foo"]
    required_columns = SelfColumnSelector.select(getitem.data)
    assert required_columns == {"foo"}

    getitem = df[["foo", "bar"]]
    required_columns = SelfColumnSelector.select(getitem.data)
    assert required_columns == {"foo", "bar"}


def test_df_groupby_agg():
    df = DataFrame({"foo": (1, 1, 3), "bar": (4, 5, 6)})

    a = df.groupby(by="foo", as_index=False).sum()
    required_columns = SelfColumnSelector.select(a.data)
    assert required_columns == {"foo"}

    a = df.groupby(by="foo").sum()
    required_columns = SelfColumnSelector.select(a.data)
    assert required_columns == set()


def test_df_merge():
    left = DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6), 1: (7, 8, 9)})
    right = DataFrame({"foo": (1, 2), "bar": (4, 5), "baz": (5, 8), 1: (7, 8)})

    joined = left.merge(right, on="foo")
    required_columns = SelfColumnSelector.select(joined.data)
    assert required_columns == {"foo"}

    joined = left.merge(right, on=["foo", "bar"])
    required_columns = SelfColumnSelector.select(joined.data)
    assert required_columns == {"foo", "bar"}

    joined = left.merge(right, left_on=["foo", "bar"], right_on=["foo", "bar"])
    required_columns = SelfColumnSelector.select(joined.data)
    assert required_columns == {"foo", "bar"}

    joined = left.merge(right)
    required_columns = SelfColumnSelector.select(joined.data)
    assert required_columns == {"foo", "bar", 1}

    joined = left.merge(right, left_on=["foo"], right_on=["bar"])
    required_columns = SelfColumnSelector.select(joined.data)
    assert required_columns == {"foo_x", "bar_y"}

    joined = left.merge(right, left_index=True, right_index=True)
    required_columns = SelfColumnSelector.select(joined.data)
    assert required_columns == set()

    joined = left.merge(right, left_index=True, right_on="foo")
    required_columns = SelfColumnSelector.select(joined.data)
    assert required_columns == {"foo"}

    joined = left.merge(right, left_index=True, right_on=["foo"])
    required_columns = SelfColumnSelector.select(joined.data)
    assert required_columns == {"foo"}

    joined = left.merge(right, left_on="foo", right_index=True)
    required_columns = SelfColumnSelector.select(joined.data)
    assert required_columns == {"foo"}

    joined = left.merge(right, left_on=["foo"], right_index=True)
    required_columns = SelfColumnSelector.select(joined.data)
    assert required_columns == {"foo"}
