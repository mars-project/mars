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
import shutil
from collections import OrderedDict
from string import printable

import numpy as np
import pandas as pd

import mars.tensor as mt
import mars.dataframe as md
from mars.session import new_session
from mars.tests.core import TestBase, require_cudf, ExecutorForTest
from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.dataframe.datasource.from_tensor import dataframe_from_tensor, dataframe_from_1d_tensors
from mars.dataframe.datasource.from_records import from_records


class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.executor = ExecutorForTest()

    def testFromPandasDataFrameExecution(self):
        pdf = pd.DataFrame(np.random.rand(20, 30), index=[np.arange(20), np.arange(20, 0, -1)])
        df = from_pandas_df(pdf, chunk_size=(13, 21))

        result = self.executor.execute_dataframe(df, concat=True)[0]
        pd.testing.assert_frame_equal(pdf, result)

    def testFromPandasSeriesExecution(self):
        ps = pd.Series(np.random.rand(20), index=[np.arange(20), np.arange(20, 0, -1)], name='a')
        series = from_pandas_series(ps, chunk_size=13)

        result = self.executor.execute_dataframe(series, concat=True)[0]
        pd.testing.assert_series_equal(ps, result)

    def testInitializerExecution(self):
        pdf = pd.DataFrame(np.random.rand(20, 30), index=[np.arange(20), np.arange(20, 0, -1)])
        df = md.DataFrame(pdf, chunk_size=(15, 10))
        result = self.executor.execute_dataframe(df, concat=True)[0]
        pd.testing.assert_frame_equal(pdf, result)

        ps = pd.Series(np.random.rand(20), index=[np.arange(20), np.arange(20, 0, -1)], name='a')
        series = md.Series(ps, chunk_size=7)
        result = self.executor.execute_dataframe(series, concat=True)[0]
        pd.testing.assert_series_equal(ps, result)

    def testSeriesFromTensor(self):
        data = np.random.rand(10)
        series = md.Series(mt.tensor(data), name='a')
        pd.testing.assert_series_equal(self.executor.execute_dataframe(series, concat=True)[0],
                                       pd.Series(data, name='a'))

        series = md.Series(mt.tensor(data, chunk_size=3))
        pd.testing.assert_series_equal(self.executor.execute_dataframe(series, concat=True)[0],
                                       pd.Series(data))

        series = md.Series(mt.ones((10,), chunk_size=4))
        pd.testing.assert_series_equal(self.executor.execute_dataframe(series, concat=True)[0],
                                       pd.Series(np.ones(10,)))

        index_data = np.random.rand(10)
        series = md.Series(mt.tensor(data, chunk_size=3), name='a',
                           index=mt.tensor(index_data, chunk_size=4))
        pd.testing.assert_series_equal(self.executor.execute_dataframe(series, concat=True)[0],
                                       pd.Series(data, name='a', index=index_data))

    def testFromTensorExecution(self):
        tensor = mt.random.rand(10, 10, chunk_size=5)
        df = dataframe_from_tensor(tensor)
        tensor_res = self.executor.execute_tensor(tensor, concat=True)[0]
        pdf_expected = pd.DataFrame(tensor_res)
        df_result = self.executor.execute_dataframe(df, concat=True)[0]
        pd.testing.assert_index_equal(df_result.index, pd.RangeIndex(0, 10))
        pd.testing.assert_index_equal(df_result.columns, pd.RangeIndex(0, 10))
        pd.testing.assert_frame_equal(df_result, pdf_expected)

        # test converted with specified index_value and columns
        tensor2 = mt.random.rand(2, 2, chunk_size=1)
        df2 = dataframe_from_tensor(tensor2, index=pd.Index(['a', 'b']), columns=pd.Index([3, 4]))
        df_result = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_index_equal(df_result.index, pd.Index(['a', 'b']))
        pd.testing.assert_index_equal(df_result.columns, pd.Index([3, 4]))

        # test converted from 1-d tensor
        tensor3 = mt.array([1, 2, 3])
        df3 = dataframe_from_tensor(tensor3)
        result3 = self.executor.execute_dataframe(df3, concat=True)[0]
        pdf_expected = pd.DataFrame(np.array([1, 2, 3]))
        pd.testing.assert_frame_equal(pdf_expected, result3)

        # test converted from identical chunks
        tensor4 = mt.ones((10, 10), chunk_size=3)
        df4 = dataframe_from_tensor(tensor4)
        result4 = self.executor.execute_dataframe(df4, concat=True)[0]
        pdf_expected = pd.DataFrame(self.executor.execute_tensor(tensor4, concat=True)[0])
        pd.testing.assert_frame_equal(pdf_expected, result4)

        # from tensor with given index
        tensor5 = mt.ones((10, 10), chunk_size=3)
        df5 = dataframe_from_tensor(tensor5, index=np.arange(0, 20, 2))
        result5 = self.executor.execute_dataframe(df5, concat=True)[0]
        pdf_expected = pd.DataFrame(self.executor.execute_tensor(tensor5, concat=True)[0],
                                    index=np.arange(0, 20, 2))
        pd.testing.assert_frame_equal(pdf_expected, result5)

        # from tensor with given index that is a tensor
        raw7 = np.random.rand(10, 10)
        tensor7 = mt.tensor(raw7, chunk_size=3)
        index_raw7 = np.random.rand(10)
        index7 = mt.tensor(index_raw7, chunk_size=4)
        df7 = dataframe_from_tensor(tensor7, index=index7)
        result7 = self.executor.execute_dataframe(df7, concat=True)[0]
        pdf_expected = pd.DataFrame(raw7, index=index_raw7)
        pd.testing.assert_frame_equal(pdf_expected, result7)

        # from tensor with given columns
        tensor6 = mt.ones((10, 10), chunk_size=3)
        df6 = dataframe_from_tensor(tensor6, columns=list('abcdefghij'))
        result6 = self.executor.execute_dataframe(df6, concat=True)[0]
        pdf_expected = pd.DataFrame(self.executor.execute_tensor(tensor6, concat=True)[0],
                                    columns=list('abcdefghij'))
        pd.testing.assert_frame_equal(pdf_expected, result6)

        # from 1d tensors
        raws8 = [('a', np.random.rand(8)), ('b', np.random.randint(10, size=8)),
                 ('c', [''.join(np.random.choice(list(printable), size=6)) for _ in range(8)])]
        tensors8 = [mt.tensor(r[1], chunk_size=3) for r in raws8]
        df8 = dataframe_from_1d_tensors(tensors8, columns=[r[0] for r in raws8])
        result = self.executor.execute_dataframe(df8, concat=True)[0]
        pdf_expected = pd.DataFrame(OrderedDict(raws8))
        pd.testing.assert_frame_equal(result, pdf_expected)

        # from 1d tensors and specify index with a tensor
        index_raw9 = np.random.rand(8)
        index9 = mt.tensor(index_raw9, chunk_size=4)
        df9 = dataframe_from_1d_tensors(tensors8, columns=[r[0] for r in raws8],
                                        index=index9)
        result = self.executor.execute_dataframe(df9, concat=True)[0]
        pdf_expected = pd.DataFrame(OrderedDict(raws8), index=index_raw9)
        pd.testing.assert_frame_equal(result, pdf_expected)

    def testFromRecordsExecution(self):
        dtype = np.dtype([('x', 'int'), ('y', 'double'), ('z', '<U16')])

        ndarr = np.ones((10,), dtype=dtype)
        pdf_expected = pd.DataFrame.from_records(ndarr, index=pd.RangeIndex(10))

        # from structured array of mars
        tensor = mt.ones((10,), dtype=dtype, chunk_size=3)
        df1 = from_records(tensor)
        df1_result = self.executor.execute_dataframe(df1, concat=True)[0]
        pd.testing.assert_frame_equal(df1_result, pdf_expected)

        # from structured array of numpy
        df2 = from_records(ndarr)
        df2_result = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_frame_equal(df2_result, pdf_expected)

    def testReadCSVExecution(self):
        tempdir = tempfile.mkdtemp()
        file_path = os.path.join(tempdir, 'test.csv')
        try:
            df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
            df.to_csv(file_path)

            pdf = pd.read_csv(file_path, index_col=0)
            mdf = self.executor.execute_dataframe(md.read_csv(file_path, index_col=0), concat=True)[0]
            pd.testing.assert_frame_equal(pdf, mdf)

            mdf2 = self.executor.execute_dataframe(md.read_csv(file_path, index_col=0, chunk_bytes=10),
                                                   concat=True)[0]
            pd.testing.assert_frame_equal(pdf, mdf2)

        finally:
            shutil.rmtree(tempdir)

        # test sep
        tempdir = tempfile.mkdtemp()
        file_path = os.path.join(tempdir, 'test.csv')
        try:
            df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
            df.to_csv(file_path, sep=';')

            pdf = pd.read_csv(file_path, sep=';', index_col=0)
            mdf = self.executor.execute_dataframe(md.read_csv(file_path, sep=';', index_col=0), concat=True)[0]
            pd.testing.assert_frame_equal(pdf, mdf)

            mdf2 = self.executor.execute_dataframe(md.read_csv(file_path, sep=';', index_col=0, chunk_bytes=10),
                                                   concat=True)[0]
            pd.testing.assert_frame_equal(pdf, mdf2)

        finally:
            shutil.rmtree(tempdir)

        # test missing value
        tempdir = tempfile.mkdtemp()
        file_path = os.path.join(tempdir, 'test.csv')
        try:
            df = pd.DataFrame({'c1': [np.nan, 'a', 'b', 'c'], 'c2': [1, 2, 3, np.nan],
                               'c3': [np.nan, np.nan, 3.4, 2.2]})
            df.to_csv(file_path)

            pdf = pd.read_csv(file_path, index_col=0)
            mdf = self.executor.execute_dataframe(md.read_csv(file_path, index_col=0), concat=True)[0]
            pd.testing.assert_frame_equal(pdf, mdf)

            mdf2 = self.executor.execute_dataframe(md.read_csv(file_path, index_col=0, chunk_bytes=12),
                                                   concat=True)[0]
            pd.testing.assert_frame_equal(pdf, mdf2)

        finally:
            shutil.rmtree(tempdir)

        tempdir = tempfile.mkdtemp()
        file_path = os.path.join(tempdir, 'test.csv')
        try:
            index = pd.date_range(start='1/1/2018', periods=100)
            df = pd.DataFrame({
                'col1': np.random.rand(100),
                'col2': np.random.choice(['a', 'b', 'c'], (100,)),
                'col3': np.arange(100)
            }, index=index)
            df.to_csv(file_path)

            pdf = pd.read_csv(file_path, index_col=0)
            mdf = self.executor.execute_dataframe(md.read_csv(file_path, index_col=0), concat=True)[0]
            pd.testing.assert_frame_equal(pdf, mdf)

            mdf2 = self.executor.execute_dataframe(md.read_csv(file_path, index_col=0, chunk_bytes=100),
                                                   concat=True)[0]
            pd.testing.assert_frame_equal(pdf, mdf2)

        finally:
            shutil.rmtree(tempdir)

        # test compression
        tempdir = tempfile.mkdtemp()
        file_path = os.path.join(tempdir, 'test.gzip')
        try:
            index = pd.date_range(start='1/1/2018', periods=100)
            df = pd.DataFrame({
                'col1': np.random.rand(100),
                'col2': np.random.choice(['a', 'b', 'c'], (100,)),
                'col3': np.arange(100)
            }, index=index)
            df.to_csv(file_path, compression='gzip')

            pdf = pd.read_csv(file_path, compression='gzip', index_col=0)
            mdf = self.executor.execute_dataframe(md.read_csv(file_path, compression='gzip', index_col=0),
                                                  concat=True)[0]
            pd.testing.assert_frame_equal(pdf, mdf)

            mdf2 = self.executor.execute_dataframe(md.read_csv(file_path, compression='gzip', index_col=0,
                                                               chunk_bytes='1k'), concat=True)[0]
            pd.testing.assert_frame_equal(pdf, mdf2)

        finally:
            shutil.rmtree(tempdir)

        # test multiply files
        tempdir = tempfile.mkdtemp()
        try:
            df = pd.DataFrame(np.random.rand(300, 3), columns=['a', 'b', 'c'])

            file_paths = [os.path.join(tempdir, 'test{}.csv'.format(i)) for i in range(3)]
            df[:100].to_csv(file_paths[0])
            df[100:200].to_csv(file_paths[1])
            df[200:].to_csv(file_paths[2])

            mdf = self.executor.execute_dataframe(md.read_csv(file_paths, index_col=0), concat=True)[0]
            pd.testing.assert_frame_equal(df, mdf)

            mdf2 = self.executor.execute_dataframe(md.read_csv(file_paths, index_col=0, chunk_bytes=50),
                                                   concat=True)[0]
            pd.testing.assert_frame_equal(df, mdf2)

        finally:
            shutil.rmtree(tempdir)

        # test wildcards in path
        tempdir = tempfile.mkdtemp()
        try:
            df = pd.DataFrame(np.random.rand(300, 3), columns=['a', 'b', 'c'])

            file_paths = [os.path.join(tempdir, 'test{}.csv'.format(i)) for i in range(3)]
            df[:100].to_csv(file_paths[0])
            df[100:200].to_csv(file_paths[1])
            df[200:].to_csv(file_paths[2])

            # As we can not guarantee the order in which these files are processed,
            # the result may not keep the original order.
            mdf = self.executor.execute_dataframe(
                md.read_csv('{}/*.csv'.format(tempdir), index_col=0), concat=True)[0]
            pd.testing.assert_frame_equal(df, mdf.sort_index())

            mdf2 = self.executor.execute_dataframe(
                md.read_csv('{}/*.csv'.format(tempdir), index_col=0, chunk_bytes=50), concat=True)[0]
            pd.testing.assert_frame_equal(df, mdf2.sort_index())

        finally:
            shutil.rmtree(tempdir)

    @require_cudf
    def testReadCSVGPUExecution(self):
        tempdir = tempfile.mkdtemp()
        file_path = os.path.join(tempdir, 'test.csv')
        try:
            df = pd.DataFrame({
                'col1': np.random.rand(100),
                'col2': np.random.choice(['a', 'b', 'c'], (100,)),
                'col3': np.arange(100)
            })
            df.to_csv(file_path, index=False)

            pdf = pd.read_csv(file_path)
            mdf = self.executor.execute_dataframe(md.read_csv(file_path, gpu=True), concat=True)[0]
            pd.testing.assert_frame_equal(pdf.reset_index(drop=True), mdf.to_pandas().reset_index(drop=True))

            mdf2 = self.executor.execute_dataframe(md.read_csv(file_path, gpu=True, chunk_bytes=200),
                                                   concat=True)[0]
            pd.testing.assert_frame_equal(pdf.reset_index(drop=True), mdf2.to_pandas().reset_index(drop=True))

        finally:
            shutil.rmtree(tempdir)

    def testReadCSVWithoutIndex(self):
        sess = new_session()

        # test csv file without storing index
        tempdir = tempfile.mkdtemp()
        file_path = os.path.join(tempdir, 'test.csv')
        try:
            df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
            df.to_csv(file_path, index=False)

            pdf = pd.read_csv(file_path)
            mdf = sess.run(md.read_csv(file_path))
            pd.testing.assert_frame_equal(pdf, mdf)

            mdf2 = sess.run(md.read_csv(file_path, chunk_bytes=10))
            pd.testing.assert_frame_equal(pdf, mdf2)
        finally:
            shutil.rmtree(tempdir)

    def testReadSQLTableExecution(self):
        import sqlalchemy as sa

        test_df = pd.DataFrame({'a': np.arange(10),
                                'b': ['s%d' % i for i in range(10)],
                                'c': np.random.rand(10)})

        with tempfile.NamedTemporaryFile(suffix='.db') as f:
            table_name = 'test'
            uri = 'sqlite:///' + f.name

            test_df.to_sql(table_name, uri, index=False)

            r = md.read_sql_table('test', uri, chunk_size=4)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            pd.testing.assert_frame_equal(result, test_df)

            engine = sa.create_engine(uri)

            try:
                # test index_col and columns
                r = md.read_sql_table('test', engine.connect(), chunk_size=4,
                                      index_col='a', columns=['b'])
                result = self.executor.execute_dataframe(r, concat=True)[0]
                expected = test_df.copy(deep=True)
                expected.set_index('a', inplace=True)
                del expected['c']
                pd.testing.assert_frame_equal(result, expected)

                table = sa.Table(table_name, sa.MetaData(), autoload=True,
                                 autoload_with=engine)
                r = md.read_sql_table(table, engine, chunk_size=4,
                                      index_col=[table.columns['a'], table.columns['b']],
                                      columns=[table.columns['c']])
                result = self.executor.execute_dataframe(r, concat=True)[0]
                expected = test_df.copy(deep=True)
                expected.set_index(['a', 'b'], inplace=True)
                pd.testing.assert_frame_equal(result, expected)
            finally:
                engine.dispose()
