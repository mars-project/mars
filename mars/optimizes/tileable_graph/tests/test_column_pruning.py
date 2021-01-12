#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import os
import tempfile

import pandas as pd

import mars.dataframe as md
from mars.core import ExecutableTuple
from mars.config import option_context
from mars.dataframe.datasource.read_csv import DataFrameReadCSV
from mars.executor import register, Executor
from mars.tests.core import TestBase, ExecutorForTest
from mars.optimizes.tileable_graph.core import tileable_optimized


class Test(TestBase):
    def setUp(self):
        self.executor = ExecutorForTest()

    def testGroupByPruneReadCSV(self):
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.csv')

            df = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                               'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                               'c': list('aabaaddce'),
                               'd': list('abaaaddce')})
            df.to_csv(file_path, index=False)

            # Use test executor
            mdf = md.read_csv(file_path).groupby('c').agg({'a': 'sum'})
            result = self.executor.execute_dataframe(mdf)[0]
            expected = df.groupby('c').agg({'a': 'sum'})
            pd.testing.assert_frame_equal(result, expected)

            mdf = md.read_csv(file_path).groupby('c').agg({'a': 'sum'})
            expected = df.groupby('c').agg({'a': 'sum'})
            pd.testing.assert_frame_equal(mdf.to_pandas(), expected)
            pd.testing.assert_frame_equal(mdf.fetch(), expected)

            optimized_df = tileable_optimized[mdf.data]
            self.assertEqual(optimized_df.inputs[0].op.usecols, ['a', 'c'])

            mdf = md.read_csv(file_path).groupby('c').agg({'b': 'sum'})
            expected = df.groupby('c').agg({'b': 'sum'})
            pd.testing.assert_frame_equal(mdf.to_pandas(), expected)
            pd.testing.assert_frame_equal(mdf.fetch(), expected)

            optimized_df = tileable_optimized[mdf.data]
            self.assertEqual(optimized_df.inputs[0].op.usecols, ['b', 'c'])

            mdf = md.read_csv(file_path).groupby('c').agg({'b': 'sum'}) + 1
            expected = df.groupby('c').agg({'b': 'sum'}) + 1
            pd.testing.assert_frame_equal(mdf.to_pandas(), expected)
            pd.testing.assert_frame_equal(mdf.fetch(), expected)

            mdf = md.read_csv(file_path, usecols=['a', 'b', 'c']).groupby('c').agg({'b': 'sum'})
            expected = df.groupby('c').agg({'b': 'sum'})
            pd.testing.assert_frame_equal(mdf.to_pandas(), expected)
            pd.testing.assert_frame_equal(mdf.fetch(), expected)
            optimized_df = tileable_optimized[mdf.data]
            self.assertEqual(optimized_df.inputs[0].op.usecols, ['b', 'c'])

            in_df = md.read_csv(file_path)
            df1 = in_df.groupby('c').agg({'b': 'sum'})
            df2 = in_df.groupby('b').agg({'a': 'sum'})

            dfs = ExecutableTuple((df1, df2))
            results = dfs.execute().fetch()
            expected1 = df.groupby('c').agg({'b': 'sum'})
            expected2 = df.groupby('b').agg({'a': 'sum'})
            pd.testing.assert_frame_equal(results[0], expected1)
            pd.testing.assert_frame_equal(results[1], expected2)

            in_df = md.read_csv(file_path)
            df1 = in_df.groupby('c').agg({'b': 'sum'})

            dfs = ExecutableTuple((in_df, df1))
            results = dfs.execute().fetch()
            expected1 = df.groupby('c').agg({'b': 'sum'})
            pd.testing.assert_frame_equal(results[0], df)
            pd.testing.assert_frame_equal(results[1], expected1)

            with option_context({'optimize_tileable_graph': False}):
                mdf = md.read_csv(file_path).groupby('c').agg({'b': 'sum'})
                expected = df.groupby('c').agg({'b': 'sum'})
                pd.testing.assert_frame_equal(mdf.to_pandas(), expected)
                pd.testing.assert_frame_equal(mdf.fetch(), expected)

                tileable_graph = mdf.build_graph()
                self.assertIsNone(list(tileable_graph.topological_iter())[0].op.usecols)

    def testGroupbyPruneReadParquet(self):
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.parquet')

            df = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                               'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                               'c': list('aabaaddce'),
                               'd': list('abaaaddce')})
            df.to_parquet(file_path, index=False)

            # Use test executor
            mdf = md.read_parquet(file_path).groupby('c').agg({'a': 'sum'})
            result = self.executor.execute_dataframes([mdf])[0]
            mdf._shape = result.shape
            expected = df.groupby('c').agg({'a': 'sum'})
            pd.testing.assert_frame_equal(result, expected)

            optimized_df = tileable_optimized[mdf.data]
            self.assertEqual(optimized_df.inputs[0].op.columns, ['a', 'c'])

            mdf = md.read_parquet(file_path).groupby('c', as_index=False).c.agg({'cnt': 'count'})
            result = self.executor.execute_dataframes([mdf])[0]
            mdf._shape = result.shape
            expected = df.groupby('c', as_index=False).c.agg({'cnt': 'count'})
            pd.testing.assert_frame_equal(result, expected)

            optimized_df = tileable_optimized[mdf.data]
            self.assertEqual(optimized_df.inputs[0].op.columns, ['c'])

    def testExecutedPruning(self):
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.csv')

            pd_df = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                                  'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                                  'c': list('aabaaddce'),
                                  'd': list('abaaaddce')})
            pd_df.to_csv(file_path, index=False)

            in_df = md.read_csv(file_path)
            mdf = in_df.groupby('c').agg({'a': 'sum'})

            expected = pd_df.groupby('c').agg({'a': 'sum'})
            pd.testing.assert_frame_equal(mdf.to_pandas(), expected)
            optimized_df = tileable_optimized[mdf.data]
            self.assertEqual(optimized_df.inputs[0].op.usecols, ['a', 'c'])

            # make sure in_df has correct columns
            pd.testing.assert_frame_equal(in_df.to_pandas(), pd_df)

            # skip pruning
            in_df = md.read_csv(file_path)
            df1 = in_df.groupby('d').agg({'b': 'min'})
            df2 = in_df[in_df.d.isin(df1.index)]

            expected1 = pd_df.groupby('d').agg({'b': 'min'})
            expected2 = pd_df[pd_df.d.isin(expected1.index)]

            pd.testing.assert_frame_equal(df2.to_pandas(), expected2)

    def testFetch(self):
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, 'test_fetch.csv')
            pd_df = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                                  'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                                  'c': list('aabaaddce'),
                                  'd': list('abaaaddce')})
            pd_df.to_csv(filename, index=False)

            df = md.read_csv(filename)
            df2 = df.groupby('d').agg({'b': 'min'})
            expected = pd_df.groupby('d').agg({'b': 'min'})
            _ = df2.execute()

            def _execute_read_csv(*_):  # pragma: no cover
                raise ValueError('cannot run read_csv again')

            try:
                register(DataFrameReadCSV, _execute_read_csv)

                pd.testing.assert_frame_equal(df2.fetch(), expected)
                pd.testing.assert_frame_equal(df2.iloc[:3].fetch(), expected.iloc[:3])
            finally:
                del Executor._op_runners[DataFrameReadCSV]
