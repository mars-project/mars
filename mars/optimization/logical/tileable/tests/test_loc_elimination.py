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

from ..... import dataframe as md
from .....core import enter_mode, TileableGraph, TileableGraphBuilder
from .. import optimize


@enter_mode(build=True)
def test_loc_elimination_for_merge(setup):
    ns = np.random.RandomState(0)
    # small dataframe
    raw1 = pd.DataFrame(
        {
            "key": ns.randint(0, 10, size=10),
            "value": np.arange(10),
        },
        index=[f"a{i}" for i in range(10)],
    )
    # big dataframe
    raw2 = pd.DataFrame(
        {
            "key": ns.randint(0, 100, size=100),
            "value": np.arange(100, 200),
            "other": ns.randint(0, 100, size=100),
        },
        index=[f"a{i}" for i in range(100)],
    )

    raw3 = pd.DataFrame(
        {
            "key": ns.randint(0, 100, size=100),
            "value": np.arange(100, 200),
            "other": ns.randint(0, 100, size=100),
        },
        index=[f"a{i}" for i in range(100)],
    )

    df1 = md.DataFrame(raw2, chunk_size=30)
    df2 = md.DataFrame(raw3, chunk_size=30)
    indexed_df1 = df1[["key", "value"]]
    indexes_df2 = df2.loc[:, ["key", "value"]]
    merged = indexed_df1.merge(indexes_df2, on="key")
    graph = TileableGraph([merged.data])
    next(TileableGraphBuilder(graph).build())
    assert len(graph) == 5
    records = optimize(graph)
    opt_merged = records.get_optimization_result(merged.data)
    assert len(graph) == 3
    assert len(records._records) == 3
    assert opt_merged.op.index_functions is not None
    assert opt_merged.inputs == [df1.data, df2.data]

    pd.testing.assert_frame_equal(
        merged.execute()
        .fetch()
        .sort_values(by=["key", "value_x"])
        .reset_index(drop=True),
        raw2[["key", "value"]]
        .merge(raw3.loc[:, ["key", "value"]], on="key")
        .sort_values(by=["key", "value_x"])
        .reset_index(drop=True),
    )

    df1 = md.DataFrame(raw2, chunk_size=30)
    df2 = md.DataFrame(raw1)
    indexed_df1 = df1.iloc[:, :2][["key", "value"]]
    merged = indexed_df1.merge(df2, on="key")
    r = merged.sum()
    graph = TileableGraph([r.data])
    next(TileableGraphBuilder(graph).build())
    assert len(graph) == 6
    records = optimize(graph)
    opt_merged = records.get_optimization_result(merged.data)
    assert len(graph) == 4
    assert len(records._records) == 3
    assert opt_merged.op.index_functions is not None
    assert opt_merged.inputs == [df1.data, df2.data]

    pd.testing.assert_series_equal(
        r.execute().fetch().reset_index(drop=True),
        raw2.iloc[:, :2].merge(raw1, on="key").sum().reset_index(drop=True),
    )
