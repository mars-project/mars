# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
from mars.tests.core import TestBase, parameterized
from mars.dataframe.core import IndexValue, Series
from mars.dataframe.reduction import DataFrameSum, DataFrameProd, DataFrameMin, DataFrameMax, \
    DataFrameCount, DataFrameMean
from mars.dataframe.merge import DataFrameConcat
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df


reduction_functions = dict(
    sum=dict(func_name='sum', op=DataFrameSum, has_skipna=True),
    prod=dict(func_name='prod', op=DataFrameProd, has_skipna=True),
    min=dict(func_name='min', op=DataFrameMin, has_skipna=True),
    max=dict(func_name='max', op=DataFrameMax, has_skipna=True),
    count=dict(func_name='count', op=DataFrameCount, has_skipna=False),
    mean=dict(func_name='mean', op=DataFrameMean, has_skipna=True)
)


@parameterized(**reduction_functions)
class Test(TestBase):
    @property
    def op_num(self):
        return getattr(OperandDef, self.func_name.upper())

    def testSeriesReductionSerialize(self):
        data = pd.Series(np.random.rand(10), name='a')
        if self.has_skipna:
            kwargs = dict(axis='index', skipna=False)
        else:
            kwargs = dict()
        reduction_df = getattr(from_pandas_series(data), self.func_name)(**kwargs).tiles()

        # pb
        chunk = reduction_df.chunks[0]
        serials = self._pb_serial(chunk)
        op, pb = serials[chunk.op, chunk.data]

        self.assertEqual(tuple(pb.index), chunk.index)
        self.assertEqual(pb.key, chunk.key)
        self.assertEqual(tuple(pb.shape), chunk.shape)
        self.assertEqual(int(op.type.split('.', 1)[1]), self.op_num)

        chunk2 = self._pb_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk.name, chunk2.name)
        self.assertEqual(chunk.op.skipna, chunk2.op.skipna)
        self.assertEqual(chunk.op.axis, chunk2.op.axis)
        pd.testing.assert_index_equal(chunk2.index_value.to_pandas(), chunk.index_value.to_pandas())

        # json
        chunk = reduction_df.chunks[0]
        serials = self._json_serial(chunk)

        chunk2 = self._json_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk2.name, chunk.name)
        self.assertEqual(chunk.op.skipna, chunk2.op.skipna)
        self.assertEqual(chunk.op.axis, chunk2.op.axis)
        pd.testing.assert_index_equal(chunk2.index_value.to_pandas(), chunk.index_value.to_pandas())

    def testSeriesReduction(self):
        data = pd.Series({'a': list(range(20))}, index=[str(i) for i in range(20)])
        series = getattr(from_pandas_series(data, chunk_size=3), self.func_name)()

        self.assertIsInstance(series, Series)
        self.assertEqual(series.name, data.name)
        self.assertIsInstance(series.index_value._index_value, IndexValue.RangeIndex)
        self.assertEqual(series.shape, ())

        series = series.tiles()

        self.assertEqual(len(series.chunks), 1)
        self.assertIsInstance(series.chunks[0].op, self.op)
        self.assertIsInstance(series.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(series.chunks[0].inputs[0].inputs), 2)

        data = pd.Series(np.random.rand(25), name='a')
        if self.has_skipna:
            kwargs = dict(axis='index', skipna=False)
        else:
            kwargs = dict()
        series = getattr(from_pandas_series(data, chunk_size=7), self.func_name)(**kwargs)

        self.assertIsInstance(series, Series)
        self.assertEqual(series.name, data.name)
        self.assertIsInstance(series.index_value._index_value, IndexValue.RangeIndex)
        self.assertEqual(series.shape, ())

        series = series.tiles()

        self.assertEqual(len(series.chunks), 1)
        self.assertIsInstance(series.chunks[0].op, self.op)
        self.assertIsInstance(series.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(series.chunks[0].inputs[0].inputs), 4)

    def testDataFrameReductionSerialize(self):
        data = pd.DataFrame(np.random.rand(10, 8), columns=[np.random.bytes(10) for _ in range(8)])
        kwargs = dict(axis='index', numeric_only=True)
        if self.has_skipna:
            kwargs['skipna'] = False
        reduction_df = getattr(from_pandas_df(data, chunk_size=3), self.func_name)(**kwargs).tiles()

        # pb
        chunk = reduction_df.chunks[0]
        serials = self._pb_serial(chunk)
        op, pb = serials[chunk.op, chunk.data]

        self.assertEqual(tuple(pb.index), chunk.index)
        self.assertEqual(pb.key, chunk.key)
        self.assertEqual(tuple(pb.shape), chunk.shape)
        self.assertEqual(int(op.type.split('.', 1)[1]), self.op_num)

        chunk2 = self._pb_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk.op.skipna, chunk2.op.skipna)
        self.assertEqual(chunk.op.axis, chunk2.op.axis)
        self.assertEqual(chunk.op.numeric_only, chunk2.op.numeric_only)
        pd.testing.assert_index_equal(chunk2.index_value.to_pandas(), chunk.index_value.to_pandas())

        # json
        chunk = reduction_df.chunks[0]
        serials = self._json_serial(chunk)

        chunk2 = self._json_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk.op.skipna, chunk2.op.skipna)
        self.assertEqual(chunk.op.axis, chunk2.op.axis)
        self.assertEqual(chunk.op.numeric_only, chunk2.op.numeric_only)
        pd.testing.assert_index_equal(chunk2.index_value.to_pandas(), chunk.index_value.to_pandas())

    def testDataFrameReduction(self):
        data = pd.DataFrame({'a': list(range(20)), 'b': list(range(20, 0, -1))},
                            index=[str(i) for i in range(20)])
        reduction_df = getattr(from_pandas_df(data, chunk_size=3), self.func_name)()

        self.assertIsInstance(reduction_df, Series)
        self.assertIsInstance(reduction_df.index_value._index_value, IndexValue.Index)
        self.assertEqual(reduction_df.shape, (2,))

        reduction_df = reduction_df.tiles()

        self.assertEqual(len(reduction_df.chunks), 1)
        self.assertIsInstance(reduction_df.chunks[0].op, self.op)
        self.assertIsInstance(reduction_df.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(reduction_df.chunks[0].inputs[0].inputs), 2)

        data = pd.DataFrame(np.random.rand(20, 10))
        reduction_df = getattr(from_pandas_df(data, chunk_size=3), self.func_name)()

        self.assertIsInstance(reduction_df, Series)
        self.assertIsInstance(reduction_df.index_value._index_value, IndexValue.RangeIndex)
        self.assertEqual(reduction_df.shape, (10,))

        reduction_df = reduction_df.tiles()

        self.assertEqual(len(reduction_df.chunks), 4)
        self.assertEqual(reduction_df.nsplits, ((3, 3, 3, 1),))
        self.assertIsInstance(reduction_df.chunks[0].op, self.op)
        self.assertIsInstance(reduction_df.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(reduction_df.chunks[0].inputs[0].inputs), 2)

        data = pd.DataFrame(np.random.rand(20, 20), index=[str(i) for i in range(20)])
        reduction_df = getattr(from_pandas_df(data, chunk_size=4), self.func_name)(axis='columns')

        self.assertEqual(reduction_df.shape, (20,))

        reduction_df = reduction_df.tiles()

        self.assertEqual(len(reduction_df.chunks), 5)
        self.assertEqual(reduction_df.nsplits, ((np.nan,) * 5,))
        self.assertIsInstance(reduction_df.chunks[0].op, self.op)
        self.assertIsInstance(reduction_df.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(reduction_df.chunks[0].inputs[0].inputs), 2)

