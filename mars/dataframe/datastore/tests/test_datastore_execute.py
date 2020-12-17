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
import unittest

import numpy as np
import pandas as pd

import mars.dataframe as md
from mars.config import option_context
from mars.dataframe import DataFrame
from mars.deploy.local.core import new_cluster
from mars.session import new_session
from mars.tests.core import mock, TestBase, ExecutorForTest

try:
    import vineyard
except ImportError:
    vineyard = None
try:
    import sqlalchemy
except ImportError:
    sqlalchemy = None
try:
    import pyarrow as pa
except ImportError:
    pa = None
try:
    import fastparquet
except ImportError:
    fastparquet = None


class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.executor = ExecutorForTest()

    def testToCSVExecution(self):
        index = pd.RangeIndex(100, 0, -1, name='index')
        raw = pd.DataFrame({
            'col1': np.random.rand(100),
            'col2': np.random.choice(['a', 'b', 'c'], (100,)),
            'col3': np.arange(100)
        }, index=index)
        df = DataFrame(raw, chunk_size=33)

        with tempfile.TemporaryDirectory() as base_path:
            # DATAFRAME TESTS
            # test one file with dataframe
            path = os.path.join(base_path, 'out.csv')

            r = df.to_csv(path)
            self.executor.execute_dataframe(r)

            result = pd.read_csv(path, dtype=raw.dtypes.to_dict())
            result.set_index('index', inplace=True)
            pd.testing.assert_frame_equal(result, raw)

            # test multi files with dataframe
            path = os.path.join(base_path, 'out-*.csv')
            r = df.to_csv(path)
            self.executor.execute_dataframe(r)

            dfs = [pd.read_csv(os.path.join(base_path, f'out-{i}.csv'),
                               dtype=raw.dtypes.to_dict())
                   for i in range(4)]
            result = pd.concat(dfs, axis=0)
            result.set_index('index', inplace=True)
            pd.testing.assert_frame_equal(result, raw)
            pd.testing.assert_frame_equal(dfs[1].set_index('index'), raw.iloc[33: 66])

            # SERIES TESTS
            series = md.Series(raw.col1, chunk_size=33)

            # test one file with series
            path = os.path.join(base_path, 'out.csv')
            r = series.to_csv(path)
            self.executor.execute_dataframe(r)

            result = pd.read_csv(path, dtype=raw.dtypes.to_dict())
            result.set_index('index', inplace=True)
            pd.testing.assert_frame_equal(result, raw.col1.to_frame())

            # test multi files with series
            path = os.path.join(base_path, 'out-*.csv')
            r = series.to_csv(path)
            self.executor.execute_dataframe(r)

            dfs = [pd.read_csv(os.path.join(base_path, f'out-{i}.csv'),
                               dtype=raw.dtypes.to_dict())
                   for i in range(4)]
            result = pd.concat(dfs, axis=0)
            result.set_index('index', inplace=True)
            pd.testing.assert_frame_equal(result, raw.col1.to_frame())
            pd.testing.assert_frame_equal(dfs[1].set_index('index'), raw.col1.to_frame().iloc[33: 66])

    @unittest.skipIf(sqlalchemy is None, 'sqlalchemy not installed')
    def testToSQL(self):
        index = pd.RangeIndex(100, 0, -1, name='index')
        raw = pd.DataFrame({
            'col1': np.random.rand(100),
            'col2': np.random.choice(['a', 'b', 'c'], (100,)),
            'col3': np.arange(100).astype('int64'),
        }, index=index)

        with tempfile.TemporaryDirectory() as d:
            table_name1 = 'test_table'
            table_name2 = 'test_table2'
            uri = 'sqlite:///' + os.path.join(d, 'test.db')

            engine = sqlalchemy.create_engine(uri)

            # test write dataframe
            df = DataFrame(raw, chunk_size=33)
            r = df.to_sql(table_name1, con=engine)
            self.executor.execute_dataframe(r)

            written = pd.read_sql(table_name1, con=engine, index_col='index') \
                .sort_index(ascending=False)
            pd.testing.assert_frame_equal(raw, written)

            # test write with existing table
            with self.assertRaises(ValueError):
                df.to_sql(table_name1, con=uri).execute()

            # test write series
            series = md.Series(raw.col1, chunk_size=33)
            with engine.connect() as conn:
                r = series.to_sql(table_name2, con=conn)
                self.executor.execute_dataframe(r)

            written = pd.read_sql(table_name2, con=engine, index_col='index') \
                .sort_index(ascending=False)
            pd.testing.assert_frame_equal(raw.col1.to_frame(), written)

    @unittest.skipIf(vineyard is None, 'vineyard not installed')
    @mock.patch('webbrowser.open_new_tab', new=lambda *_, **__: True)
    def testToVineyard(self):
        def testWithGivenSession(session):
            with option_context({'vineyard.socket': '/tmp/vineyard.sock'}):
                df1 = DataFrame(pd.DataFrame(np.arange(12).reshape(3, 4), columns=['a', 'b', 'c', 'd']),
                                chunk_size=2)
                object_id = df1.to_vineyard().execute(session=session)
                df2 = md.from_vineyard(object_id)

                df1_value = df1.execute(session=session)
                df2_value = df2.execute(session=session)
                pd.testing.assert_frame_equal(df1_value.reset_index(drop=True), df2_value.reset_index(drop=True))

        with new_session().as_default() as session:
            testWithGivenSession(session)

        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M', web=True) as cluster:
            with new_session(cluster.endpoint).as_default() as session:
                testWithGivenSession(session)

            with new_session('http://' + cluster._web_endpoint).as_default() as web_session:
                testWithGivenSession(web_session)

    @unittest.skipIf(pa is None, 'pyarrow not installed')
    def testToParquetArrowExecution(self):
        raw = pd.DataFrame({
            'col1': np.random.rand(100),
            'col2': np.arange(100),
            'col3': np.random.choice(['a', 'b', 'c'], (100,)),
        })
        df = DataFrame(raw, chunk_size=33)

        with tempfile.TemporaryDirectory() as base_path:
            # DATAFRAME TESTS
            path = os.path.join(base_path, 'out-*.parquet')
            r = df.to_parquet(path)
            self.executor.execute_dataframe(r)

            read_df = md.read_parquet(path)
            result = self.executor.execute_dataframe(read_df, concat=True)[0]
            result = result.sort_index()
            pd.testing.assert_frame_equal(result, raw)

            read_df = md.read_parquet(path)
            result = self.executor.execute_dataframe(read_df, concat=True)[0]
            result = result.sort_index()
            pd.testing.assert_frame_equal(result, raw)

            # test read_parquet then to_parquet
            read_df = md.read_parquet(path)
            r = read_df.to_parquet(path)
            self.executor.execute_dataframes([r])

            # test partition_cols
            path = os.path.join(base_path, 'out-partitioned')
            r = df.to_parquet(path, partition_cols=['col3'])
            self.executor.execute_dataframe(r)

            read_df = md.read_parquet(path)
            result = self.executor.execute_dataframe(read_df, concat=True)[0]
            result['col3'] = result['col3'].astype('object')
            pd.testing.assert_frame_equal(result.sort_values('col1').reset_index(drop=True),
                                          raw.sort_values('col1').reset_index(drop=True))

    @unittest.skipIf(fastparquet is None, 'fastparquet not installed')
    def testToParquetFastParquetExecution(self):
        raw = pd.DataFrame({
            'col1': np.random.rand(100),
            'col2': np.arange(100),
            'col3': np.random.choice(['a', 'b', 'c'], (100,)),
        })
        df = DataFrame(raw, chunk_size=33)

        with tempfile.TemporaryDirectory() as base_path:
            # test fastparquet
            path = os.path.join(base_path, 'out-fastparquet-*.parquet')
            r = df.to_parquet(path, engine='fastparquet', compression='gzip')
            self.executor.execute_dataframe(r)
