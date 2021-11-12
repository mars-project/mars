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

import pickle
import sys

import pandas as pd
import numpy as np

from ...tests.core import assert_groupby_equal
from ...utils import calc_data_size, estimate_pandas_size
from ..groupby_wrapper import wrapped_groupby


def test_groupby_wrapper():
    df = pd.DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.randn(8),
            "D": np.random.randn(8),
        },
        index=pd.MultiIndex.from_tuples([(i // 4, i) for i in range(8)]),
    )

    conv_func = lambda x: pickle.loads(pickle.dumps(x))

    grouped = conv_func(wrapped_groupby(df, level=0))
    assert_groupby_equal(grouped, df.groupby(level=0))
    assert grouped.shape == (8, 4)
    assert grouped.is_frame is True
    assert sys.getsizeof(grouped) > sys.getsizeof(grouped.groupby_obj)
    assert calc_data_size(grouped) > sys.getsizeof(grouped.groupby_obj)
    assert grouped.estimate_size() > estimate_pandas_size(grouped.groupby_obj)

    grouped = conv_func(wrapped_groupby(df, level=0).C)
    assert_groupby_equal(grouped, df.groupby(level=0).C)
    assert grouped.shape == (8,)
    assert grouped.is_frame is False

    grouped = conv_func(wrapped_groupby(df, "B"))
    assert_groupby_equal(grouped, df.groupby("B"))
    assert grouped.shape == (8, 4)
    assert grouped.is_frame is True

    grouped = conv_func(wrapped_groupby(df, "B").C)
    assert_groupby_equal(grouped, df.groupby("B").C, with_selection=True)
    assert grouped.shape == (8,)
    assert grouped.is_frame is False

    grouped = conv_func(wrapped_groupby(df, "B")[["C", "D"]])
    assert_groupby_equal(grouped, df.groupby("B")[["C", "D"]], with_selection=True)
    assert grouped.shape == (8, 2)
    assert grouped.is_frame is True

    grouped = conv_func(wrapped_groupby(df, ["B", "C"]))
    assert_groupby_equal(grouped, df.groupby(["B", "C"]))
    assert grouped.shape == (8, 4)
    assert grouped.is_frame is True

    grouped = conv_func(wrapped_groupby(df, ["B", "C"]).C)
    assert_groupby_equal(grouped, df.groupby(["B", "C"]).C, with_selection=True)
    assert grouped.shape == (8,)
    assert grouped.is_frame is False

    grouped = conv_func(wrapped_groupby(df, ["B", "C"])[["A", "D"]])
    assert_groupby_equal(
        grouped, df.groupby(["B", "C"])[["A", "D"]], with_selection=True
    )
    assert grouped.shape == (8, 2)
    assert grouped.is_frame is True

    grouped = conv_func(wrapped_groupby(df, ["B", "C"])[["C", "D"]])
    assert_groupby_equal(
        grouped, df.groupby(["B", "C"])[["C", "D"]], with_selection=True
    )
    assert grouped.shape == (8, 2)
    assert grouped.is_frame is True

    grouped = conv_func(wrapped_groupby(df, lambda x: x[-1] % 2))
    assert_groupby_equal(grouped, df.groupby(lambda x: x[-1] % 2), with_selection=True)
    assert grouped.shape == (8, 4)
    assert grouped.is_frame is True

    grouped = conv_func(wrapped_groupby(df, lambda x: x[-1] % 2).C)
    assert_groupby_equal(
        grouped, df.groupby(lambda x: x[-1] % 2).C, with_selection=True
    )
    assert grouped.shape == (8,)
    assert grouped.is_frame is False

    grouped = conv_func(wrapped_groupby(df, lambda x: x[-1] % 2)[["C", "D"]])
    assert_groupby_equal(
        grouped, df.groupby(lambda x: x[-1] % 2)[["C", "D"]], with_selection=True
    )
    assert grouped.shape == (8, 2)
    assert grouped.is_frame is True

    grouped = conv_func(wrapped_groupby(df.B, lambda x: x[-1] % 2))
    assert_groupby_equal(
        grouped, df.B.groupby(lambda x: x[-1] % 2), with_selection=True
    )
    assert grouped.shape == (8,)
    assert grouped.is_frame is False
