# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import mars.dataframe as md
from mars.tests.core import TestBase
from mars.dataframe.core import DataFrameGroupBy, DataFrame
from mars.dataframe.groupby.core import DataFrameGroupByOperand,\
    DataFrameGroupByReduce, DataFrameGroupByMap, DataFrameShuffleProxy
from mars.dataframe.groupby.aggregation import DataFrameGroupByAgg



class Test(TestBase):
    def testGroupBy(self):
        df = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                           'b': [1, 3, 4, 5, 6, 5, 4, 4, 4]})
        mdf = md.DataFrame(df, chunk_size=2)
        grouped = mdf.groupby('c2')

        self.assertIsInstance(grouped, DataFrameGroupBy)
        self.assertIsInstance(grouped.op, DataFrameGroupByOperand)

        grouped.tiles()
        self.assertEqual(len(grouped.chunks), 5)
        for chunk in grouped.chunks:
            self.assertIsInstance(chunk.op, DataFrameGroupByOperand)

    def testGroupByAgg(self):
        df = pd.DataFrame({'c1': range(10),
                           'c2': np.random.choice(['a', 'b', 'c'], (10,)),
                           'c3': np.random.rand(10)})
        mdf = md.DataFrame(df, chunk_size=2)
        r = mdf.groupby('c2').sum()

        self.assertIsInstance(r.op, DataFrameGroupByAgg)
        self.assertIsInstance(r, DataFrame)

        r.tiles()
        self.assertEqual(len(r.chunks), 5)
        for chunk in r.chunks:
            self.assertIsInstance(chunk.op, DataFrameGroupByAgg)
            self.assertEqual(chunk.op.stage.value, 'combine')
            self.assertIsInstance(chunk.inputs[0].op, DataFrameGroupByReduce)
            self.assertIsInstance(chunk.inputs[0].inputs[0].op, DataFrameShuffleProxy)
            self.assertIsInstance(chunk.inputs[0].inputs[0].inputs[0].op, DataFrameGroupByMap)

            agg_chunk = chunk.inputs[0].inputs[0].inputs[0].inputs[0]
            self.assertEqual(agg_chunk.op.stage.value, 'agg')
