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

from ....core import tile
from ... import DataFrame


def test_to_csv():
    raw = pd.DataFrame(np.random.rand(10, 5))
    df = DataFrame(raw, chunk_size=4)

    r = df.to_csv("*.csv")
    r = tile(r)

    assert r.chunk_shape[1] == 1
    for i, c in enumerate(r.chunks):
        assert type(c.op).__name__ == "DataFrameToCSV"
        assert c.inputs[0] is r.inputs[0].chunks[i].data

    # test one file
    r = df.to_csv("out.csv")
    r = tile(r)

    assert r.chunk_shape[1] == 1
    for i, c in enumerate(r.chunks):
        assert len(c.inputs) == 2
        assert c.inputs[0].inputs[0] is r.inputs[0].chunks[i].data
        assert type(c.inputs[1].op).__name__ == "DataFrameToCSVStat"
