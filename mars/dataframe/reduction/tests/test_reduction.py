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

from mars import opcodes as OperandDef
from mars.tests.core import TestBase
from mars.dataframe.core import IndexValue, Series
from mars.dataframe.reduction import SeriesSum, DataFrameSum
from mars.dataframe.merge import DataFrameConcat
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df


class Test(TestBase):
    def testSeriesSumSerialize(self):
        data = pd.Series(np.random.rand(10), name='a')
        sum_df = from_pandas_series(data).sum(axis='index', skipna=False).tiles()

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
        series = from_pandas_series(data, chunk_size=3).sum()

        self.assertIsInstance(series, Series)
        self.assertEqual(series.name, data.name)
        self.assertIsInstance(series.index_value._index_value, IndexValue.RangeIndex)
        self.assertEqual(series.shape, ())

        series.tiles()

        self.assertEqual(len(series.chunks), 1)
        self.assertIsInstance(series.chunks[0].op, SeriesSum)
        self.assertIsInstance(series.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(series.chunks[0].inputs[0].inputs), 2)

        data = pd.Series(np.random.rand(25), name='a')
        series = from_pandas_series(data, chunk_size=7).sum(axis='index', skipna=False)

        self.assertIsInstance(series, Series)
        self.assertEqual(series.name, data.name)
        self.assertIsInstance(series.index_value._index_value, IndexValue.RangeIndex)
        self.assertEqual(series.shape, ())

        series.tiles()

        self.assertEqual(len(series.chunks), 1)
        self.assertIsInstance(series.chunks[0].op, SeriesSum)
        self.assertIsInstance(series.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(series.chunks[0].inputs[0].inputs), 4)

    def testDataFrameSumSerialize(self):
        data = pd.DataFrame(np.random.rand(10, 8), columns=[np.random.bytes(10) for _ in range(8)])
        sum_df = from_pandas_df(data, chunk_size=3).sum(axis='index', skipna=False, numeric_only=True).tiles()

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
        self.assertEqual(chunk.op.skipna, chunk2.op.skipna)
        self.assertEqual(chunk.op.axis, chunk2.op.axis)
        self.assertEqual(chunk.op.numeric_only, chunk2.op.numeric_only)
        pd.testing.assert_index_equal(chunk2.index_value.to_pandas(), chunk.index_value.to_pandas())

        # json
        chunk = sum_df.chunks[0]
        serials = self._json_serial(chunk)

        chunk2 = self._json_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk.op.skipna, chunk2.op.skipna)
        self.assertEqual(chunk.op.axis, chunk2.op.axis)
        self.assertEqual(chunk.op.numeric_only, chunk2.op.numeric_only)
        pd.testing.assert_index_equal(chunk2.index_value.to_pandas(), chunk.index_value.to_pandas())

    def testDataFrameSum(self):
        data = pd.DataFrame({'a': list(range(20)), 'b': list(range(20, 0, -1))},
                            index=[str(i) for i in range(20)])
        sum_df = from_pandas_df(data, chunk_size=3).sum()

        self.assertIsInstance(sum_df, Series)
        self.assertIsInstance(sum_df.index_value._index_value, IndexValue.Index)
        self.assertEqual(sum_df.shape, (2,))

        sum_df.tiles()

        self.assertEqual(len(sum_df.chunks), 1)
        self.assertIsInstance(sum_df.chunks[0].op, DataFrameSum)
        self.assertIsInstance(sum_df.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(sum_df.chunks[0].inputs[0].inputs), 2)

        data = pd.DataFrame(np.random.rand(20, 10))
        sum_df = from_pandas_df(data, chunk_size=3).sum()

        self.assertIsInstance(sum_df, Series)
        self.assertIsInstance(sum_df.index_value._index_value, IndexValue.RangeIndex)
        self.assertEqual(sum_df.shape, (10,))

        sum_df.tiles()

        self.assertEqual(len(sum_df.chunks), 4)
        self.assertEqual(sum_df.nsplits, ((3, 3, 3, 1),))
        self.assertIsInstance(sum_df.chunks[0].op, DataFrameSum)
        self.assertIsInstance(sum_df.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(sum_df.chunks[0].inputs[0].inputs), 2)

        data = pd.DataFrame(np.random.rand(20, 20), index=[str(i) for i in range(20)])
        sum_df = from_pandas_df(data, chunk_size=4).sum(axis='columns')

        self.assertEqual(sum_df.shape, (20,))

        sum_df.tiles()

        self.assertEqual(len(sum_df.chunks), 5)
        self.assertEqual(sum_df.nsplits, ((np.nan,) * 5,))
        self.assertIsInstance(sum_df.chunks[0].op, DataFrameSum)
        self.assertIsInstance(sum_df.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(sum_df.chunks[0].inputs[0].inputs), 2)

