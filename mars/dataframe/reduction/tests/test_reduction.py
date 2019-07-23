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

import pandas as pd
import numpy as np

from .... import opcodes as OperandDef
from mars.tests.core import TestBase
from mars.dataframe.core import IndexValue
from mars.dataframe.reduction import sum, SeriesSum
from mars.dataframe.merge import DataFrameConcat
from mars.dataframe.datasource.series import from_pandas as from_pandas_series


class Test(TestBase):
    def testSeriesSumSerialize(self):
        data = pd.Series(np.random.rand(10), name='a')
        sum_df = sum(from_pandas_series(data), axis='index', skipna=False).tiles()

        # pb
        chunk = sum_df.chunks[0]
        serials = self._pb_serial(chunk)
        op, pb = serials[chunk.op, chunk.data]

        self.assertEqual(tuple(pb.index), chunk.index)
        self.assertEqual(pb.key, chunk.key)
        self.assertEqual(tuple(pb.shape), chunk.shape)
        self.assertEqual(int(op.type.split('.', 1)[1]), OperandDef.SUM)

        chunk2 = self._pb_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk.name, chunk2.name)
        self.assertEqual(chunk.op.skipna, chunk2.op.skipna)
        self.assertEqual(chunk.op.axis, chunk2.op.axis)
        pd.testing.assert_index_equal(chunk2.index_value.to_pandas(), chunk.index_value.to_pandas())

        # json
        chunk = sum_df.chunks[0]
        serials = self._json_serial(chunk)

        chunk2 = self._json_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk2.name, chunk.name)
        self.assertEqual(chunk.op.skipna, chunk2.op.skipna)
        self.assertEqual(chunk.op.axis, chunk2.op.axis)
        pd.testing.assert_index_equal(chunk2.index_value.to_pandas(), chunk.index_value.to_pandas())

    def testSeriesSum(self):
        data = pd.Series({'a': list(range(20))}, index=[str(i) for i in range(20)])
        series = sum(from_pandas_series(data, chunk_size=3))

        self.assertEqual(series.name, data.name)
        self.assertIsInstance(series.index_value._index_value, IndexValue.Index)
        self.assertEqual(series.shape, ())

        series.tiles()

        self.assertEqual(len(series.chunks), 1)
        self.assertIsInstance(series.chunks[0].op, SeriesSum)
        self.assertIsInstance(series.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(series.chunks[0].inputs[0].inputs), 2)

        data = pd.Series(np.random.rand(25), name='a')
        series = sum(from_pandas_series(data, chunk_size=7), axis='index', skipna=False)

        self.assertEqual(series.name, data.name)
        self.assertIsInstance(series.index_value._index_value, IndexValue.Index)
        self.assertEqual(series.shape, ())

        series.tiles()

        self.assertEqual(len(series.chunks), 1)
        self.assertIsInstance(series.chunks[0].op, SeriesSum)
        self.assertIsInstance(series.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(series.chunks[0].inputs[0].inputs), 4)
