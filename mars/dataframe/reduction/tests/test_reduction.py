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

import functools
import operator
import unittest
from functools import reduce

import pandas as pd
import numpy as np

from mars import opcodes as OperandDef, dataframe as md
from mars.operands import OperandStage
from mars.tests.core import TestBase, parameterized
from mars.tensor import Tensor
from mars.dataframe.core import DataFrame, IndexValue, Series, OutputType
from mars.dataframe.reduction import DataFrameSum, DataFrameProd, DataFrameMin, \
    DataFrameMax, DataFrameCount, DataFrameMean, DataFrameVar, DataFrameAll, \
    DataFrameAny, DataFrameSkew, DataFrameKurtosis, DataFrameSem, \
    DataFrameAggregate, DataFrameCummin, DataFrameCummax, DataFrameCumprod, \
    DataFrameCumsum, DataFrameNunique, CustomReduction
from mars.dataframe.reduction.aggregation import where_function
from mars.dataframe.reduction.core import ReductionCompiler
from mars.dataframe.merge import DataFrameConcat
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df


reduction_functions = dict(
    sum=dict(func_name='sum', op=DataFrameSum),
    prod=dict(func_name='prod', op=DataFrameProd),
    min=dict(func_name='min', op=DataFrameMin),
    max=dict(func_name='max', op=DataFrameMax),
    count=dict(func_name='count', op=DataFrameCount, has_skipna=False),
    mean=dict(func_name='mean', op=DataFrameMean),
    var=dict(func_name='var', op=DataFrameVar),
    skew=dict(func_name='skew', op=DataFrameSkew),
    kurt=dict(func_name='kurt', op=DataFrameKurtosis),
    sem=dict(func_name='sem', op=DataFrameSem),
    all=dict(func_name='all', op=DataFrameAll, has_numeric_only=False, has_bool_only=True),
    any=dict(func_name='any', op=DataFrameAny, has_numeric_only=False, has_bool_only=True),
)


@parameterized(defaults=dict(has_skipna=True, has_numeric_only=True, has_bool_only=False),
               **reduction_functions)
class TestReduction(TestBase):
    @property
    def op_name(self):
        return getattr(OperandDef, self.func_name.upper())

    def testSeriesReductionSerialize(self):
        data = pd.Series(np.random.rand(10), name='a')
        if self.has_skipna:
            kwargs = dict(axis='index', skipna=False)
        else:
            kwargs = dict()
        reduction_df = getattr(from_pandas_series(data, chunk_size=3), self.func_name)(**kwargs).tiles()

        # pb
        chunk = reduction_df.chunks[0]
        serials = self._pb_serial(chunk)
        op, pb = serials[chunk.op, chunk.data]

        self.assertEqual(tuple(pb.index), chunk.index)
        self.assertEqual(pb.key, chunk.key)
        self.assertEqual(tuple(pb.shape), chunk.shape)
        self.assertEqual(int(op.type.split('.', 1)[1]), DataFrameAggregate._op_type_)

        chunk2 = self._pb_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk.op.agg_funcs[0].kwds.get('skipna'),
                         chunk2.op.agg_funcs[0].kwds.get('skipna'))
        self.assertEqual(chunk.op.axis, chunk2.op.axis)

        # json
        chunk = reduction_df.chunks[0]
        serials = self._json_serial(chunk)

        chunk2 = self._json_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk.op.agg_funcs[0].kwds.get('skipna'),
                         chunk2.op.agg_funcs[0].kwds.get('skipna'))
        self.assertEqual(chunk.op.axis, chunk2.op.axis)

    def testSeriesReduction(self):
        data = pd.Series(range(20), index=[str(i) for i in range(20)])
        series = getattr(from_pandas_series(data, chunk_size=3), self.func_name)()

        self.assertIsInstance(series, Tensor)
        self.assertIsInstance(series.op, self.op)
        self.assertEqual(series.shape, ())

        series = series.tiles()

        self.assertEqual(len(series.chunks), 1)
        self.assertIsInstance(series.chunks[0].op, DataFrameAggregate)
        self.assertIsInstance(series.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(series.chunks[0].inputs[0].inputs), 2)

        data = pd.Series(np.random.rand(25), name='a')
        if self.has_skipna:
            kwargs = dict(axis='index', skipna=False)
        else:
            kwargs = dict()
        series = getattr(from_pandas_series(data, chunk_size=7), self.func_name)(**kwargs)

        self.assertIsInstance(series, Tensor)
        self.assertEqual(series.shape, ())

        series = series.tiles()

        self.assertEqual(len(series.chunks), 1)
        self.assertIsInstance(series.chunks[0].op, DataFrameAggregate)
        self.assertIsInstance(series.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(series.chunks[0].inputs[0].inputs), 4)

    def testDataFrameReductionSerialize(self):
        data = pd.DataFrame(np.random.rand(10, 8), columns=[np.random.bytes(10) for _ in range(8)])
        kwargs = dict(axis='index')
        if self.has_skipna:
            kwargs['skipna'] = False
        if self.has_numeric_only:
            kwargs['numeric_only'] = True
        if self.has_bool_only:
            kwargs['bool_only'] = True
        reduction_df = getattr(from_pandas_df(data, chunk_size=3), self.func_name)(**kwargs).tiles()

        # pb
        chunk = reduction_df.chunks[0]
        serials = self._pb_serial(chunk)
        chunk_op, chunk_pb = serials[chunk.op, chunk.data]

        self.assertEqual(tuple(chunk_pb.index), chunk.index)
        self.assertEqual(chunk_pb.key, chunk.key)
        self.assertEqual(tuple(chunk_pb.shape), chunk.shape)
        self.assertEqual(int(chunk_op.type.split('.', 1)[1]), DataFrameAggregate._op_type_)

        chunk2 = self._pb_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk.op.agg_funcs[0].kwds.get('skipna'),
                         chunk2.op.agg_funcs[0].kwds.get('skipna'))
        self.assertEqual(chunk.op.axis, chunk2.op.axis)
        self.assertEqual(chunk.op.numeric_only, chunk2.op.numeric_only)
        self.assertEqual(chunk.op.bool_only, chunk2.op.bool_only)
        pd.testing.assert_index_equal(chunk2.index_value.to_pandas(), chunk.index_value.to_pandas())

        # json
        chunk = reduction_df.chunks[0]
        serials = self._json_serial(chunk)

        chunk2 = self._json_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk.op.agg_funcs[0].kwds.get('skipna'),
                         chunk2.op.agg_funcs[0].kwds.get('skipna'))
        self.assertEqual(chunk.op.axis, chunk2.op.axis)
        self.assertEqual(chunk.op.numeric_only, chunk2.op.numeric_only)
        self.assertEqual(chunk.op.bool_only, chunk2.op.bool_only)
        pd.testing.assert_index_equal(chunk2.index_value.to_pandas(), chunk.index_value.to_pandas())

    def testDataFrameReduction(self):
        data = pd.DataFrame({'a': list(range(20)), 'b': list(range(20, 0, -1))},
                            index=[str(i) for i in range(20)])
        reduction_df = getattr(from_pandas_df(data, chunk_size=3), self.func_name)()

        self.assertIsInstance(reduction_df, Series)
        self.assertIsInstance(reduction_df.op, self.op)
        self.assertIsInstance(reduction_df.index_value._index_value, IndexValue.Index)
        self.assertEqual(reduction_df.shape, (2,))

        reduction_df = reduction_df.tiles()

        self.assertEqual(len(reduction_df.chunks), 1)
        self.assertIsInstance(reduction_df.chunks[0].op, DataFrameAggregate)
        self.assertIsInstance(reduction_df.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(reduction_df.chunks[0].inputs[0].inputs), 2)

        data = pd.DataFrame(np.random.rand(20, 10))
        reduction_df = getattr(from_pandas_df(data, chunk_size=3), self.func_name)()

        self.assertIsInstance(reduction_df, Series)
        self.assertIsInstance(reduction_df.index_value._index_value,
                              (IndexValue.RangeIndex, IndexValue.Int64Index))
        self.assertEqual(reduction_df.shape, (10,))

        reduction_df = reduction_df.tiles()

        self.assertEqual(len(reduction_df.chunks), 4)
        self.assertEqual(reduction_df.nsplits, ((3, 3, 3, 1),))
        self.assertIsInstance(reduction_df.chunks[0].op, DataFrameAggregate)
        self.assertIsInstance(reduction_df.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(reduction_df.chunks[0].inputs[0].inputs), 2)

        data = pd.DataFrame(np.random.rand(20, 20), index=[str(i) for i in range(20)])
        reduction_df = getattr(from_pandas_df(data, chunk_size=4), self.func_name)(axis='columns')

        self.assertEqual(reduction_df.shape, (20,))

        reduction_df = reduction_df.tiles()

        self.assertEqual(len(reduction_df.chunks), 5)
        self.assertEqual(reduction_df.nsplits, ((4,) * 5,))
        self.assertIsInstance(reduction_df.chunks[0].op, DataFrameAggregate)
        self.assertIsInstance(reduction_df.chunks[0].inputs[0].op, DataFrameConcat)
        self.assertEqual(len(reduction_df.chunks[0].inputs[0].inputs), 2)

        with self.assertRaises(NotImplementedError):
            getattr(from_pandas_df(data, chunk_size=3), self.func_name)(level=0, axis=1)


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
        pd.testing.assert_index_equal(chunk.index_value.to_pandas(), chunk2.index_value.to_pandas())

        # json
        chunk = reduction_df.chunks[0]
        serials = self._json_serial(chunk)

        chunk2 = self._json_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk.op.skipna, chunk2.op.skipna)
        self.assertEqual(chunk.op.axis, chunk2.op.axis)
        pd.testing.assert_index_equal(chunk.index_value.to_pandas(), chunk2.index_value.to_pandas())

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

    def testDataFrameReductionSerialize(self):
        data = pd.DataFrame(np.random.rand(10, 8), columns=[np.random.bytes(10) for _ in range(8)])
        kwargs = dict(axis='index')
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
        self.assertEqual(int(op.type.split('.', 1)[1]), self.op_name)

        chunk2 = self._pb_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk.op.skipna, chunk2.op.skipna)
        self.assertEqual(chunk.op.axis, chunk2.op.axis)
        pd.testing.assert_index_equal(chunk2.columns_value.to_pandas(), chunk.columns_value.to_pandas())
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
        pd.testing.assert_index_equal(chunk2.columns_value.to_pandas(), chunk.columns_value.to_pandas())
        pd.testing.assert_index_equal(chunk2.index_value.to_pandas(), chunk.index_value.to_pandas())

    def testDataFrameReduction(self):
        data = pd.DataFrame({'a': list(range(20)), 'b': list(range(20, 0, -1))},
                            index=[str(i) for i in range(20)])
        reduction_df = getattr(from_pandas_df(data, chunk_size=3), self.func_name)()

        self.assertIsInstance(reduction_df, DataFrame)
        self.assertIsInstance(reduction_df.index_value._index_value, IndexValue.Index)
        self.assertEqual(reduction_df.shape, (20, 2))

        reduction_df = reduction_df.tiles()

        self.assertEqual(len(reduction_df.chunks), 7)
        self.assertIsInstance(reduction_df.chunks[0].op, self.op)
        self.assertEqual(reduction_df.chunks[0].op.stage, OperandStage.combine)
        self.assertIsInstance(reduction_df.chunks[-1].inputs[-1].op, self.op)
        self.assertEqual(reduction_df.chunks[-1].inputs[-1].op.stage, OperandStage.map)
        self.assertEqual(len(reduction_df.chunks[-1].inputs), 7)

        data = pd.DataFrame(np.random.rand(20, 10))
        reduction_df = getattr(from_pandas_df(data, chunk_size=3), self.func_name)()

        self.assertIsInstance(reduction_df, DataFrame)
        self.assertIsInstance(reduction_df.index_value._index_value, IndexValue.RangeIndex)
        self.assertEqual(reduction_df.shape, (20, 10))

        reduction_df = reduction_df.tiles()

        self.assertEqual(len(reduction_df.chunks), 28)
        self.assertEqual(reduction_df.nsplits, ((3, 3, 3, 3, 3, 3, 2), (3, 3, 3, 1)))
        self.assertEqual(reduction_df.chunks[0].op.stage, OperandStage.combine)
        self.assertIsInstance(reduction_df.chunks[-1].inputs[-1].op, self.op)
        self.assertEqual(reduction_df.chunks[-1].inputs[-1].op.stage, OperandStage.map)
        self.assertEqual(len(reduction_df.chunks[-1].inputs), 7)


class TestNUnique(TestBase):
    def testNunique(self):
        data = pd.DataFrame(np.random.randint(0, 6, size=(20, 10)),
                            columns=['c' + str(i) for i in range(10)])
        df = from_pandas_df(data, chunk_size=3)
        result = df.nunique()

        self.assertEqual(result.shape, (10,))
        self.assertEqual(result.op.output_types[0], OutputType.series)
        self.assertIsInstance(result.op, DataFrameNunique)

        tiled = result.tiles()
        self.assertEqual(tiled.shape, (10,))
        self.assertEqual(len(tiled.chunks), 4)
        self.assertEqual(tiled.nsplits, ((3, 3, 3, 1,),))
        self.assertEqual(tiled.chunks[0].op.stage, OperandStage.agg)
        self.assertIsInstance(tiled.chunks[0].op, DataFrameAggregate)

        data2 = data.copy()
        df2 = from_pandas_df(data2, chunk_size=3)
        result2 = df2.nunique(axis=1)

        self.assertEqual(result2.shape, (20,))
        self.assertEqual(result2.op.output_types[0], OutputType.series)
        self.assertIsInstance(result2.op, DataFrameNunique)

        tiled = result2.tiles()
        self.assertEqual(tiled.shape, (20,))
        self.assertEqual(len(tiled.chunks), 7)
        self.assertEqual(tiled.nsplits, ((3, 3, 3, 3, 3, 3, 2,),))
        self.assertEqual(tiled.chunks[0].op.stage, OperandStage.agg)
        self.assertIsInstance(tiled.chunks[0].op, DataFrameAggregate)


class TestAggregate(TestBase):
    def testDataFrameAggregate(self):
        data = pd.DataFrame(np.random.rand(20, 19))
        agg_funcs = ['sum', 'min', 'max', 'mean', 'var', 'std', 'all', 'any',
                     'skew', 'kurt', 'sem']

        df = from_pandas_df(data)
        result = df.agg(agg_funcs).tiles()
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.shape, (len(agg_funcs), data.shape[1]))
        self.assertListEqual(list(result.columns_value.to_pandas()), list(range(19)))
        self.assertListEqual(list(result.index_value.to_pandas()), agg_funcs)
        self.assertEqual(result.op.output_types[0], OutputType.dataframe)
        self.assertListEqual(result.op.func, agg_funcs)

        df = from_pandas_df(data, chunk_size=(3, 4))

        result = df.agg('sum').tiles()
        self.assertEqual(len(result.chunks), 5)
        self.assertEqual(result.shape, (data.shape[1],))
        self.assertListEqual(list(result.index_value.to_pandas()), list(range(data.shape[1])))
        self.assertEqual(result.op.output_types[0], OutputType.series)
        self.assertListEqual(result.op.func, ['sum'])
        agg_chunk = result.chunks[0]
        self.assertEqual(agg_chunk.shape, (4,))
        self.assertListEqual(list(agg_chunk.index_value.to_pandas()), list(range(4)))
        self.assertEqual(agg_chunk.op.stage, OperandStage.agg)

        result = df.agg('sum', axis=1).tiles()
        self.assertEqual(len(result.chunks), 7)
        self.assertEqual(result.shape, (data.shape[0],))
        self.assertListEqual(list(result.index_value.to_pandas()), list(range(data.shape[0])))
        self.assertEqual(result.op.output_types[0], OutputType.series)
        agg_chunk = result.chunks[0]
        self.assertEqual(agg_chunk.shape, (3,))
        self.assertListEqual(list(agg_chunk.index_value.to_pandas()), list(range(3)))
        self.assertEqual(agg_chunk.op.stage, OperandStage.agg)

        result = df.agg('var', axis=1).tiles()
        self.assertEqual(len(result.chunks), 7)
        self.assertEqual(result.shape, (data.shape[0],))
        self.assertListEqual(list(result.index_value.to_pandas()), list(range(data.shape[0])))
        self.assertEqual(result.op.output_types[0], OutputType.series)
        self.assertListEqual(result.op.func, ['var'])
        agg_chunk = result.chunks[0]
        self.assertEqual(agg_chunk.shape, (3,))
        self.assertListEqual(list(agg_chunk.index_value.to_pandas()), list(range(3)))
        self.assertEqual(agg_chunk.op.stage, OperandStage.agg)

        result = df.agg(agg_funcs).tiles()
        self.assertEqual(len(result.chunks), 5)
        self.assertEqual(result.shape, (len(agg_funcs), data.shape[1]))
        self.assertListEqual(list(result.columns_value.to_pandas()), list(range(data.shape[1])))
        self.assertListEqual(list(result.index_value.to_pandas()), agg_funcs)
        self.assertEqual(result.op.output_types[0], OutputType.dataframe)
        self.assertListEqual(result.op.func, agg_funcs)
        agg_chunk = result.chunks[0]
        self.assertEqual(agg_chunk.shape, (len(agg_funcs), 4))
        self.assertListEqual(list(agg_chunk.columns_value.to_pandas()), list(range(4)))
        self.assertListEqual(list(agg_chunk.index_value.to_pandas()), agg_funcs)
        self.assertEqual(agg_chunk.op.stage, OperandStage.agg)

        result = df.agg(agg_funcs, axis=1).tiles()
        self.assertEqual(len(result.chunks), 7)
        self.assertEqual(result.shape, (data.shape[0], len(agg_funcs)))
        self.assertListEqual(list(result.columns_value.to_pandas()), agg_funcs)
        self.assertListEqual(list(result.index_value.to_pandas()), list(range(data.shape[0])))
        self.assertEqual(result.op.output_types[0], OutputType.dataframe)
        self.assertListEqual(result.op.func, agg_funcs)
        agg_chunk = result.chunks[0]
        self.assertEqual(agg_chunk.shape, (3, len(agg_funcs)))
        self.assertListEqual(list(agg_chunk.columns_value.to_pandas()), agg_funcs)
        self.assertListEqual(list(agg_chunk.index_value.to_pandas()), list(range(3)))
        self.assertEqual(agg_chunk.op.stage, OperandStage.agg)

        dict_fun = {0: 'sum', 2: ['var', 'max'], 9: ['mean', 'var', 'std']}
        all_cols = set(reduce(operator.add, [[v] if isinstance(v, str) else v for v in dict_fun.values()]))
        result = df.agg(dict_fun).tiles()
        self.assertEqual(len(result.chunks), 2)
        self.assertEqual(result.shape, (len(all_cols), len(dict_fun)))
        self.assertSetEqual(set(result.columns_value.to_pandas()), set(dict_fun.keys()))
        self.assertSetEqual(set(result.index_value.to_pandas()), all_cols)
        self.assertEqual(result.op.output_types[0], OutputType.dataframe)
        self.assertListEqual(result.op.func[0], [dict_fun[0]])
        self.assertListEqual(result.op.func[2], dict_fun[2])
        agg_chunk = result.chunks[0]
        self.assertEqual(agg_chunk.shape, (len(all_cols), 2))
        self.assertListEqual(list(agg_chunk.columns_value.to_pandas()), [0, 2])
        self.assertSetEqual(set(agg_chunk.index_value.to_pandas()), all_cols)
        self.assertEqual(agg_chunk.op.stage, OperandStage.agg)

        with self.assertRaises(TypeError):
            df.agg(sum_0='sum', mean_0='mean')
        with self.assertRaises(NotImplementedError):
            df.agg({0: ['sum', 'min', 'var'], 9: ['mean', 'var', 'std']}, axis=1)

    def testSeriesAggregate(self):
        data = pd.Series(np.random.rand(20), index=[str(i) for i in range(20)], name='a')
        agg_funcs = ['sum', 'min', 'max', 'mean', 'var', 'std', 'all', 'any',
                     'skew', 'kurt', 'sem']

        series = from_pandas_series(data)

        result = series.agg(agg_funcs).tiles()
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.shape, (len(agg_funcs),))
        self.assertListEqual(list(result.index_value.to_pandas()), agg_funcs)
        self.assertEqual(result.op.output_types[0], OutputType.series)
        self.assertListEqual(result.op.func, agg_funcs)

        series = from_pandas_series(data, chunk_size=3)

        result = series.agg('sum').tiles()
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.shape, ())
        self.assertEqual(result.op.output_types[0], OutputType.scalar)
        agg_chunk = result.chunks[0]
        self.assertEqual(agg_chunk.shape, ())
        self.assertEqual(agg_chunk.op.stage, OperandStage.agg)

        result = series.agg(agg_funcs).tiles()
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.shape, (len(agg_funcs),))
        self.assertListEqual(list(result.index_value.to_pandas()), agg_funcs)
        self.assertEqual(result.op.output_types[0], OutputType.series)
        self.assertListEqual(result.op.func, agg_funcs)
        agg_chunk = result.chunks[0]
        self.assertEqual(agg_chunk.shape, (len(agg_funcs),))
        self.assertListEqual(list(agg_chunk.index_value.to_pandas()), agg_funcs)
        self.assertEqual(agg_chunk.op.stage, OperandStage.agg)

        with self.assertRaises(TypeError):
            series.agg(sum_0=(0, 'sum'), mean_0=(0, 'mean'))


class TestReductionCompiler(TestBase):
    def testCompileFunction(self):
        compiler = ReductionCompiler()
        ms = md.Series([1, 2, 3])
        # no Mars objects inside closures
        with self.assertRaises(ValueError):
            compiler.add_function(functools.partial(lambda x: (x + ms).sum()), ndim=2)
        # function should return a Mars object
        with self.assertRaises(ValueError):
            compiler.add_function(lambda x: x is not None, ndim=2)
        # function should perform some sort of reduction in dimensionality
        with self.assertRaises(ValueError):
            compiler.add_function(lambda x: x, ndim=2)
        # function should only contain acceptable operands
        with self.assertRaises(ValueError):
            compiler.add_function(lambda x: x.sort_values().max(), ndim=1)
        with self.assertRaises(ValueError):
            compiler.add_function(lambda x: x.max().shift(1), ndim=2)

        # test agg for all data
        for ndim in [1, 2]:
            compiler = ReductionCompiler(store_source=True)
            compiler.add_function(lambda x: (x ** 2).count() + 1, ndim=ndim)
            result = compiler.compile()
            # check pre_funcs
            self.assertEqual(len(result.pre_funcs), 1)
            self.assertIn('pow', result.pre_funcs[0].func.__source__)
            # check agg_funcs
            self.assertEqual(len(result.agg_funcs), 1)
            self.assertEqual(result.agg_funcs[0].map_func_name, 'count')
            self.assertEqual(result.agg_funcs[0].agg_func_name, 'sum')
            # check post_funcs
            self.assertEqual(len(result.post_funcs), 1)
            self.assertEqual(result.post_funcs[0].func_name, '<lambda_0>')
            self.assertIn('add', result.post_funcs[0].func.__source__)

            compiler.add_function(lambda x: -x.prod() ** 2 + (1 + (x ** 2).count()), ndim=ndim)
            result = compiler.compile()
            # check pre_funcs
            self.assertEqual(len(result.pre_funcs), 2)
            self.assertTrue('pow' in result.pre_funcs[0].func.__source__
                            or 'pow' in result.pre_funcs[1].func.__source__)
            self.assertTrue('pow' not in result.pre_funcs[0].func.__source__
                            or 'pow' not in result.pre_funcs[1].func.__source__)
            # check agg_funcs
            self.assertEqual(len(result.agg_funcs), 2)
            self.assertSetEqual(set(result.agg_funcs[i].map_func_name for i in range(2)),
                                {'count', 'prod'})
            self.assertSetEqual(set(result.agg_funcs[i].agg_func_name for i in range(2)),
                                {'sum', 'prod'})
            # check post_funcs
            self.assertEqual(len(result.post_funcs), 2)
            self.assertEqual(result.post_funcs[0].func_name, '<lambda_0>')
            self.assertIn('add', result.post_funcs[0].func.__source__)
            self.assertIn('add', result.post_funcs[1].func.__source__)

            compiler = ReductionCompiler(store_source=True)
            compiler.add_function(lambda x: where_function(x.all(), x.count(), 0), ndim=ndim)
            result = compiler.compile()
            # check pre_funcs
            self.assertEqual(len(result.pre_funcs), 1)
            self.assertEqual(result.pre_funcs[0].input_key, result.pre_funcs[0].output_key)
            # check agg_funcs
            self.assertEqual(len(result.agg_funcs), 2)
            self.assertSetEqual(set(result.agg_funcs[i].map_func_name for i in range(2)),
                                {'all', 'count'})
            self.assertSetEqual(set(result.agg_funcs[i].agg_func_name for i in range(2)),
                                {'sum', 'all'})
            # check post_funcs
            self.assertEqual(len(result.post_funcs), 1)
            if ndim == 1:
                self.assertIn('np.where', result.post_funcs[0].func.__source__)
            else:
                self.assertNotIn('np.where', result.post_funcs[0].func.__source__)
                self.assertIn('.where', result.post_funcs[0].func.__source__)

        # test agg for specific columns
        compiler = ReductionCompiler(store_source=True)
        compiler.add_function(lambda x: 1 + x.sum(), ndim=2, cols=['a', 'b'])
        compiler.add_function(lambda x: -1 + x.sum(), ndim=2, cols=['b', 'c'])
        result = compiler.compile()
        # check pre_funcs
        self.assertEqual(len(result.pre_funcs), 1)
        self.assertSetEqual(set(result.pre_funcs[0].columns), set('abc'))
        # check agg_funcs
        self.assertEqual(len(result.agg_funcs), 1)
        self.assertEqual(result.agg_funcs[0].map_func_name, 'sum')
        self.assertEqual(result.agg_funcs[0].agg_func_name, 'sum')
        # check post_funcs
        self.assertEqual(len(result.post_funcs), 2)
        self.assertSetEqual(set(''.join(sorted(result.post_funcs[i].columns)) for i in range(2)),
                            {'ab', 'bc'})

    def testCustomAggregation(self):
        class MockReduction1(CustomReduction):
            def agg(self, v1):
                return v1.sum()

        class MockReduction2(CustomReduction):
            def pre(self, value):
                return value + 1, value ** 2

            def agg(self, v1, v2):
                return v1.sum(), v2.prod()

            def post(self, v1, v2):
                return v1 + v2

        for ndim in [1, 2]:
            compiler = ReductionCompiler()
            compiler.add_function(MockReduction1(), ndim=ndim)
            result = compiler.compile()
            # check agg_funcs
            self.assertEqual(len(result.agg_funcs), 1)
            self.assertEqual(result.agg_funcs[0].map_func_name, 'custom_reduction')
            self.assertEqual(result.agg_funcs[0].agg_func_name, 'custom_reduction')
            self.assertIsInstance(result.agg_funcs[0].custom_reduction, MockReduction1)
            self.assertEqual(result.agg_funcs[0].output_limit, 1)

            compiler = ReductionCompiler()
            compiler.add_function(MockReduction2(), ndim=ndim)
            result = compiler.compile()
            # check agg_funcs
            self.assertEqual(len(result.agg_funcs), 1)
            self.assertEqual(result.agg_funcs[0].map_func_name, 'custom_reduction')
            self.assertEqual(result.agg_funcs[0].agg_func_name, 'custom_reduction')
            self.assertIsInstance(result.agg_funcs[0].custom_reduction, MockReduction2)
            self.assertEqual(result.agg_funcs[0].output_limit, 2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
