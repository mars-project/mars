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
def test_arithmetic_query(setup):
    raw = pd.DataFrame(np.random.rand(100, 10), columns=list("ABCDEFGHIJ"))
    df1 = md.DataFrame(raw, chunk_size=10)
    df2 = -df1["A"] + df1["B"] * 5
    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())
    records = optimize(graph)
    opt_df2 = records.get_optimization_result(df2.data)
    print(opt_df2.op.expr)
