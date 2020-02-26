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
import unittest

from mars import opcodes as OperandDef
from mars.operands import OperandStage
from mars.tests.core import TestBase, parameterized
from mars.dataframe.core import Series
from mars.dataframe.reduction import DataFrameCummax, DataFrameCummin, \
    DataFrameCumprod, DataFrameCumsum
from mars.dataframe.datasource.series import from_pandas as from_pandas_series

cum_reduction_functions = dict(
    cummin=dict(func_name='cummin', op=DataFrameCummin, has_skipna=True),
    cummax=dict(func_name='cummax', op=DataFrameCummax, has_skipna=True),
    cumprod=dict(func_name='cumprod', op=DataFrameCumprod, has_skipna=True),
    cumsum=dict(func_name='cumsum', op=DataFrameCumsum, has_skipna=True),
)


@parameterized(**cum_reduction_functions)
class TestCumReduction(TestBase):
    @property
    def op_name(self):
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
        self.assertEqual(int(op.type.split('.', 1)[1]), self.op_name)

        chunk2 = self._pb_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk.op.skipna, chunk2.op.skipna)
        self.assertEqual(chunk.op.axis, chunk2.op.axis)

        # json
        chunk = reduction_df.chunks[0]
        serials = self._json_serial(chunk)

        chunk2 = self._json_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk.op.skipna, chunk2.op.skipna)
        self.assertEqual(chunk.op.axis, chunk2.op.axis)

    def testSeriesReduction(self):
        data = pd.Series({'a': list(range(20))}, index=[str(i) for i in range(20)])
        series = getattr(from_pandas_series(data, chunk_size=3), self.func_name)()

        self.assertIsInstance(series, Series)
        self.assertEqual(series.shape, (20,))

        series = series.tiles()

        self.assertEqual(len(series.chunks), 7)
        self.assertIsInstance(series.chunks[0].op, self.op)
        self.assertEqual(series.chunks[0].op.stage, OperandStage.combine)
        self.assertIsInstance(series.chunks[-1].inputs[-1].op, self.op)
        self.assertEqual(series.chunks[-1].inputs[-1].op.stage, OperandStage.map)
        self.assertEqual(len(series.chunks[-1].inputs), 7)

        data = pd.Series(np.random.rand(25), name='a')
        if self.has_skipna:
            kwargs = dict(axis='index', skipna=False)
        else:
            kwargs = dict()
        series = getattr(from_pandas_series(data, chunk_size=7), self.func_name)(**kwargs)

        self.assertIsInstance(series, Series)
        self.assertEqual(series.shape, (25,))

        series = series.tiles()

        self.assertEqual(len(series.chunks), 4)
        self.assertIsInstance(series.chunks[0].op, self.op)
        self.assertEqual(series.chunks[0].op.stage, OperandStage.combine)
        self.assertIsInstance(series.chunks[-1].inputs[-1].op, self.op)
        self.assertEqual(series.chunks[-1].inputs[-1].op.stage, OperandStage.map)
        self.assertEqual(len(series.chunks[-1].inputs), 4)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
