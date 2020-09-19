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
import time
import unittest
from collections import OrderedDict
from datetime import datetime
from string import printable

import numpy as np
import pandas as pd
try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None
try:
    import fastparquet
except ImportError:  # pragma: no cover
    fastparquet = None

import mars.tensor as mt
import mars.dataframe as md
from mars.config import option_context
from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.dataframe.datasource.index import from_pandas as from_pandas_index, from_tileable
from mars.dataframe.datasource.from_tensor import dataframe_from_tensor, dataframe_from_1d_tileables
from mars.dataframe.datasource.from_records import from_records
from mars.session import new_session
from mars.tests.core import TestBase, require_cudf
from mars.utils import arrow_array_to_objects


class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.ctx, self.executor = self._create_test_context()

    def testFromPandasDataFrameExecution(self):
        # test empty DataFrame
        pdf = pd.DataFrame()
        df = from_pandas_df(pdf)

        result = self.executor.execute_dataframe(df, concat=True)[0]
        pd.testing.assert_frame_equal(pdf, result)

        pdf = pd.DataFrame(columns=list('ab'))
        df = from_pandas_df(pdf)

        result = self.executor.execute_dataframe(df, concat=True)[0]
        pd.testing.assert_frame_equal(pdf, result)

        pdf = pd.DataFrame(np.random.rand(20, 30), index=[np.arange(20), np.arange(20, 0, -1)])
        df = from_pandas_df(pdf, chunk_size=(13, 21))

        result = self.executor.execute_dataframe(df, concat=True)[0]
        pd.testing.assert_frame_equal(pdf, result)

    def testFromPandasSeriesExecution(self):
        # test empty Series
        ps = pd.Series(name='a')
        series = from_pandas_series(ps, chunk_size=13)

        result = self.executor.execute_dataframe(series, concat=True)[0]
        pd.testing.assert_series_equal(ps, result)

        ps = pd.Series(np.random.rand(20), index=[np.arange(20), np.arange(20, 0, -1)], name='a')
        series = from_pandas_series(ps, chunk_size=13)

        result = self.executor.execute_dataframe(series, concat=True)[0]
        pd.testing.assert_series_equal(ps, result)

    def testFromPandasIndexExecution(self):
        pd_index = pd.timedelta_range('1 days', periods=10)
        index = from_pandas_index(pd_index, chunk_size=7)

        result = self.executor.execute_dataframe(index, concat=True)[0]
        pd.testing.assert_index_equal(pd_index, result)

    def testIndexExecution(self):
        rs = np.random.RandomState(0)
        pdf = pd.DataFrame(rs.rand(20, 10), index=np.arange(20, 0, -1),
                           columns=['a' + str(i) for i in range(10)])
        df = from_pandas_df(pdf, chunk_size=13)

        # test df.index
        result = self.executor.execute_dataframe(df.index, concat=True)[0]
        pd.testing.assert_index_equal(result, pdf.index)

        result = self.executor.execute_dataframe(df.columns, concat=True)[0]
        pd.testing.assert_index_equal(result, pdf.columns)

        # df has unknown chunk shape on axis 0
        df = df[df.a1 < 0.5]

        # test df.index
        result = self.executor.execute_dataframe(df.index, concat=True)[0]
        pd.testing.assert_index_equal(result, pdf[pdf.a1 < 0.5].index)

        s = pd.Series(pdf['a1'], index=pd.RangeIndex(20))
        series = from_pandas_series(s, chunk_size=13)

        # test series.index which has value
        result = self.executor.execute_dataframe(series.index, concat=True)[0]
        pd.testing.assert_index_equal(result, s.index)

        s = pdf['a2']
        series = from_pandas_series(s, chunk_size=13)

        # test series.index
        result = self.executor.execute_dataframe(series.index, concat=True)[0]
        pd.testing.assert_index_equal(result, s.index)

        # test tensor
        raw = rs.random(20)
        t = mt.tensor(raw, chunk_size=13)

        result = self.executor.execute_dataframe(from_tileable(t), concat=True)[0]
        pd.testing.assert_index_equal(result, pd.Index(raw))

    def testInitializerExecution(self):
        arr = np.random.rand(20, 30)

        pdf = pd.DataFrame(arr, index=[np.arange(20), np.arange(20, 0, -1)])
        df = md.DataFrame(pdf, chunk_size=(15, 10))
        result = self.executor.execute_dataframe(df, concat=True)[0]
        pd.testing.assert_frame_equal(pdf, result)

        df = md.DataFrame(arr, index=md.date_range('2020-1-1', periods=20))
        result = self.executor.execute_dataframe(df, concat=True)[0]
        pd.testing.assert_frame_equal(
            result, pd.DataFrame(arr, index=pd.date_range('2020-1-1', periods=20)))

        df = md.DataFrame({"prices": [100, 101, np.nan, 100, 89, 88]},
                          index=md.date_range('1/1/2010', periods=6, freq='D'))
        result = self.executor.execute_dataframe(df, concat=True)[0]
        pd.testing.assert_frame_equal(
            result, pd.DataFrame({"prices": [100, 101, np.nan, 100, 89, 88]},
                                 index=pd.date_range('1/1/2010', periods=6, freq='D')))

        s = np.random.rand(20)

        ps = pd.Series(s, index=[np.arange(20), np.arange(20, 0, -1)], name='a')
        series = md.Series(ps, chunk_size=7)
        result = self.executor.execute_dataframe(series, concat=True)[0]
        pd.testing.assert_series_equal(ps, result)

        series = md.Series(s, index=md.date_range('2020-1-1', periods=20))
        result = self.executor.execute_dataframe(series, concat=True)[0]
        pd.testing.assert_series_equal(
            result, pd.Series(s, index=pd.date_range('2020-1-1', periods=20)))

        pi = pd.IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)])
        index = md.Index(md.Index(pi))
        result = self.executor.execute_dataframe(index, concat=True)[0]
        pd.testing.assert_index_equal(pi, result)

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

        series = md.Series(mt.tensor(data, chunk_size=3), name='a',
                           index=md.date_range('2020-1-1', periods=10))
        pd.testing.assert_series_equal(self.executor.execute_dataframe(series, concat=True)[0],
                                       pd.Series(data, name='a', index=pd.date_range('2020-1-1', periods=10)))

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

        # from tensor with given index is a md.Index
        raw10 = np.random.rand(10, 10)
        tensor10 = mt.tensor(raw10, chunk_size=3)
        index10 = md.date_range('2020-1-1', periods=10, chunk_size=3)
        df10 = dataframe_from_tensor(tensor10, index=index10)
        result10 = self.executor.execute_dataframe(df10, concat=True)[0]
        pdf_expected = pd.DataFrame(raw10, index=pd.date_range('2020-1-1', periods=10))
        pd.testing.assert_frame_equal(pdf_expected, result10)

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
        tensors8 = OrderedDict((r[0], mt.tensor(r[1], chunk_size=3)) for r in raws8)
        raws8.append(('d', 1))
        raws8.append(('e', pd.date_range('2020-1-1', periods=8)))
        tensors8['d'] = 1
        tensors8['e'] = raws8[-1][1]
        df8 = dataframe_from_1d_tileables(tensors8, columns=[r[0] for r in raws8])
        result = self.executor.execute_dataframe(df8, concat=True)[0]
        pdf_expected = pd.DataFrame(OrderedDict(raws8))
        pd.testing.assert_frame_equal(result, pdf_expected)

        # from 1d tensors and specify index with a tensor
        index_raw9 = np.random.rand(8)
        index9 = mt.tensor(index_raw9, chunk_size=4)
        df9 = dataframe_from_1d_tileables(tensors8, columns=[r[0] for r in raws8],
                                          index=index9)
        result = self.executor.execute_dataframe(df9, concat=True)[0]
        pdf_expected = pd.DataFrame(OrderedDict(raws8), index=index_raw9)
        pd.testing.assert_frame_equal(result, pdf_expected)

        # from 1d tensors and specify index
        df11 = dataframe_from_1d_tileables(tensors8, columns=[r[0] for r in raws8],
                                           index=md.date_range('2020-1-1', periods=8))
        result = self.executor.execute_dataframe(df11, concat=True)[0]
        pdf_expected = pd.DataFrame(OrderedDict(raws8),
                                    index=pd.date_range('2020-1-1', periods=8))
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
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.csv')

            df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64), columns=['a', 'b', 'c'])
            df.to_csv(file_path)

            pdf = pd.read_csv(file_path, index_col=0)
            mdf = self.executor.execute_dataframe(md.read_csv(file_path, index_col=0), concat=True)[0]
            pd.testing.assert_frame_equal(pdf, mdf)

            mdf2 = self.executor.execute_dataframe(md.read_csv(file_path, index_col=0, chunk_bytes=10),
                                                   concat=True)[0]
            pd.testing.assert_frame_equal(pdf, mdf2)

            mdf = self.executor.execute_dataframe(md.read_csv(file_path, index_col=0, nrows=1), concat=True)[0]
            pd.testing.assert_frame_equal(df[:1], mdf)

        # test sep
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.csv')

            df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
            df.to_csv(file_path, sep=';')

            pdf = pd.read_csv(file_path, sep=';', index_col=0)
            mdf = self.executor.execute_dataframe(md.read_csv(file_path, sep=';', index_col=0), concat=True)[0]
            pd.testing.assert_frame_equal(pdf, mdf)

            mdf2 = self.executor.execute_dataframe(md.read_csv(file_path, sep=';', index_col=0, chunk_bytes=10),
                                                   concat=True)[0]
            pd.testing.assert_frame_equal(pdf, mdf2)

        # test missing value
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.csv')

            df = pd.DataFrame({'c1': [np.nan, 'a', 'b', 'c'], 'c2': [1, 2, 3, np.nan],
                               'c3': [np.nan, np.nan, 3.4, 2.2]})
            df.to_csv(file_path)

            pdf = pd.read_csv(file_path, index_col=0)
            mdf = self.executor.execute_dataframe(md.read_csv(file_path, index_col=0), concat=True)[0]
            pd.testing.assert_frame_equal(pdf, mdf)

            mdf2 = self.executor.execute_dataframe(md.read_csv(file_path, index_col=0, chunk_bytes=12),
                                                   concat=True)[0]
            pd.testing.assert_frame_equal(pdf, mdf2)

        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.csv')

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

        # test compression
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.gzip')

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

        # test multiply files
        with tempfile.TemporaryDirectory() as tempdir:
            df = pd.DataFrame(np.random.rand(300, 3), columns=['a', 'b', 'c'])

            file_paths = [os.path.join(tempdir, f'test{i}.csv') for i in range(3)]
            df[:100].to_csv(file_paths[0])
            df[100:200].to_csv(file_paths[1])
            df[200:].to_csv(file_paths[2])

            mdf = self.executor.execute_dataframe(md.read_csv(file_paths, index_col=0), concat=True)[0]
            pd.testing.assert_frame_equal(df, mdf)

            mdf2 = self.executor.execute_dataframe(md.read_csv(file_paths, index_col=0, chunk_bytes=50),
                                                   concat=True)[0]
            pd.testing.assert_frame_equal(df, mdf2)

        # test wildcards in path
        with tempfile.TemporaryDirectory() as tempdir:
            df = pd.DataFrame(np.random.rand(300, 3), columns=['a', 'b', 'c'])

            file_paths = [os.path.join(tempdir, f'test{i}.csv') for i in range(3)]
            df[:100].to_csv(file_paths[0])
            df[100:200].to_csv(file_paths[1])
            df[200:].to_csv(file_paths[2])

            # As we can not guarantee the order in which these files are processed,
            # the result may not keep the original order.
            mdf = self.executor.execute_dataframe(
                md.read_csv(f'{tempdir}/*.csv', index_col=0), concat=True)[0]
            pd.testing.assert_frame_equal(df, mdf.sort_index())

            mdf2 = self.executor.execute_dataframe(
                md.read_csv(f'{tempdir}/*.csv', index_col=0, chunk_bytes=50), concat=True)[0]
            pd.testing.assert_frame_equal(df, mdf2.sort_index())

    @unittest.skipIf(pa is None, 'pyarrow not installed')
    def testReadCSVUseArrowDtype(self):
        df = pd.DataFrame({
            'col1': np.random.rand(100),
            'col2': np.random.choice(['a' * 2, 'b' * 3, 'c' * 4], (100,)),
            'col3': np.arange(100)
        })
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.csv')
            df.to_csv(file_path, index=False)

            pdf = pd.read_csv(file_path)
            mdf = md.read_csv(file_path, use_arrow_dtype=True)
            result = self.executor.execute_dataframe(mdf, concat=True)[0]
            self.assertIsInstance(mdf.dtypes.iloc[1], md.ArrowStringDtype)
            self.assertIsInstance(result.dtypes.iloc[1], md.ArrowStringDtype)
            pd.testing.assert_frame_equal(arrow_array_to_objects(result), pdf)

        with tempfile.TemporaryDirectory() as tempdir:
            with option_context({'dataframe.use_arrow_dtype': True}):
                file_path = os.path.join(tempdir, 'test.csv')
                df.to_csv(file_path, index=False)

                pdf = pd.read_csv(file_path)
                mdf = md.read_csv(file_path)
                result = self.executor.execute_dataframe(mdf, concat=True)[0]
                self.assertIsInstance(mdf.dtypes.iloc[1], md.ArrowStringDtype)
                self.assertIsInstance(result.dtypes.iloc[1], md.ArrowStringDtype)
                pd.testing.assert_frame_equal(arrow_array_to_objects(result), pdf)

        # test compression
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.gzip')
            df.to_csv(file_path, compression='gzip', index=False)

            pdf = pd.read_csv(file_path, compression='gzip')
            mdf = md.read_csv(file_path, compression='gzip', use_arrow_dtype=True)
            result = self.executor.execute_dataframe(mdf, concat=True)[0]
            self.assertIsInstance(mdf.dtypes.iloc[1], md.ArrowStringDtype)
            self.assertIsInstance(result.dtypes.iloc[1], md.ArrowStringDtype)
            pd.testing.assert_frame_equal(arrow_array_to_objects(result), pdf)

    @require_cudf
    def testReadCSVGPUExecution(self):
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.csv')

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

    def testReadCSVWithoutIndex(self):
        sess = new_session()

        # test csv file without storing index
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.csv')

            df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
            df.to_csv(file_path, index=False)

            pdf = pd.read_csv(file_path)
            mdf = sess.run(md.read_csv(file_path, incremental_index=True))
            pd.testing.assert_frame_equal(pdf, mdf)

            mdf2 = sess.run(md.read_csv(file_path, incremental_index=True, chunk_bytes=10))
            pd.testing.assert_frame_equal(pdf, mdf2)

    def testReadSQLExecution(self):
        import sqlalchemy as sa

        test_df = pd.DataFrame({'a': np.arange(10).astype(np.int64, copy=False),
                                'b': [f's{i}' for i in range(10)],
                                'c': np.random.rand(10),
                                'd': [datetime.fromtimestamp(time.time() + 3600 * (i - 5))
                                      for i in range(10)]})

        with tempfile.TemporaryDirectory() as d:
            table_name = 'test'
            table_name2 = 'test2'
            uri = 'sqlite:///' + os.path.join(d, 'test.db')

            test_df.to_sql(table_name, uri, index=False)

            # test read with table name
            r = md.read_sql_table('test', uri, chunk_size=4)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            pd.testing.assert_frame_equal(result, test_df)

            # test read with sql string and offset method
            r = md.read_sql_query('select * from test where c > 0.5', uri,
                                  parse_dates=['d'], chunk_size=4,
                                  incremental_index=True)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            pd.testing.assert_frame_equal(result, test_df[test_df.c > 0.5].reset_index(drop=True))

            # test read with sql string and partition method with integer cols
            r = md.read_sql('select * from test where b > \'s5\'', uri,
                            parse_dates=['d'], partition_col='a', num_partitions=3,
                            incremental_index=True)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            pd.testing.assert_frame_equal(result, test_df[test_df.b > 's5'].reset_index(drop=True))

            # test read with sql string and partition method with datetime cols
            r = md.read_sql_query('select * from test where b > \'s5\'', uri,
                                  parse_dates={'d': '%Y-%m-%d %H:%M:%S'},
                                  partition_col='d', num_partitions=3,
                                  incremental_index=True)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            pd.testing.assert_frame_equal(result, test_df[test_df.b > 's5'].reset_index(drop=True))

            # test read with sql string and partition method with datetime cols
            r = md.read_sql_query('select * from test where b > \'s5\'', uri,
                                  parse_dates=['d'], partition_col='d', num_partitions=3,
                                  index_col='d')
            result = self.executor.execute_dataframe(r, concat=True)[0]
            pd.testing.assert_frame_equal(result, test_df[test_df.b > 's5'].set_index('d'))

            # test SQL that return no result
            r = md.read_sql_query('select * from test where a > 1000', uri)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            pd.testing.assert_frame_equal(result, pd.DataFrame(columns=test_df.columns))

            engine = sa.create_engine(uri)
            m = sa.MetaData()
            try:
                # test index_col and columns
                r = md.read_sql_table('test', engine.connect(), chunk_size=4,
                                      index_col='a', columns=['b', 'd'])
                result = self.executor.execute_dataframe(r, concat=True)[0]
                expected = test_df.copy(deep=True)
                expected.set_index('a', inplace=True)
                del expected['c']
                pd.testing.assert_frame_equal(result, expected)

                # do not specify chunk_size
                r = md.read_sql_table('test', engine.connect(),
                                      index_col='a', columns=['b', 'd'])
                result = self.executor.execute_dataframe(r, concat=True)[0]
                pd.testing.assert_frame_equal(result, expected)

                table = sa.Table(table_name, m, autoload=True,
                                 autoload_with=engine)
                r = md.read_sql_table(table, engine, chunk_size=4,
                                      index_col=[table.columns['a'], table.columns['b']],
                                      columns=[table.columns['c'], 'd'])
                result = self.executor.execute_dataframe(r, concat=True)[0]
                expected = test_df.copy(deep=True)
                expected.set_index(['a', 'b'], inplace=True)
                pd.testing.assert_frame_equal(result, expected)

                # test table with primary key
                sa.Table(table_name2, m,
                         sa.Column('id', sa.Integer, primary_key=True),
                         sa.Column('a', sa.Integer),
                         sa.Column('b', sa.String),
                         sa.Column('c', sa.Float),
                         sa.Column('d', sa.DateTime))
                m.create_all(engine)
                test_df = test_df.copy(deep=True)
                test_df.index.name = 'id'
                test_df.to_sql(table_name2, uri, if_exists='append')

                r = md.read_sql_table(table_name2, engine, chunk_size=4, index_col='id')
                result = self.executor.execute_dataframe(r, concat=True)[0]
                pd.testing.assert_frame_equal(result, test_df)
            finally:
                engine.dispose()

    @unittest.skipIf(pa is None, 'pyarrow not installed')
    def testReadSQLUseArrowDtype(self):
        test_df = pd.DataFrame({'a': np.arange(10).astype(np.int64, copy=False),
                                'b': [f's{i}' for i in range(10)],
                                'c': np.random.rand(10),
                                'd': [datetime.fromtimestamp(time.time() + 3600 * (i - 5))
                                      for i in range(10)]})

        with tempfile.TemporaryDirectory() as d:
            table_name = 'test'
            uri = 'sqlite:///' + os.path.join(d, 'test.db')

            test_df.to_sql(table_name, uri, index=False)

            r = md.read_sql_table('test', uri, chunk_size=4, use_arrow_dtype=True)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            self.assertIsInstance(r.dtypes.iloc[1], md.ArrowStringDtype)
            self.assertIsInstance(result.dtypes.iloc[1], md.ArrowStringDtype)
            pd.testing.assert_frame_equal(arrow_array_to_objects(result), test_df)

            # test read with sql string and offset method
            r = md.read_sql_query('select * from test where c > 0.5', uri,
                                  parse_dates=['d'], chunk_size=4,
                                  incremental_index=True,
                                  use_arrow_dtype=True)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            self.assertIsInstance(r.dtypes.iloc[1], md.ArrowStringDtype)
            self.assertIsInstance(result.dtypes.iloc[1], md.ArrowStringDtype)
            pd.testing.assert_frame_equal(arrow_array_to_objects(result),
                                          test_df[test_df.c > 0.5].reset_index(drop=True))

    def testDateRangeExecution(self):
        for closed in [None, 'left', 'right']:
            # start, periods, freq
            dr = md.date_range('2020-1-1', periods=10, chunk_size=3, closed=closed)

            result = self.executor.execute_dataframe(dr, concat=True)[0]
            expected = pd.date_range('2020-1-1', periods=10, closed=closed)
            pd.testing.assert_index_equal(result, expected)

            # end, periods, freq
            dr = md.date_range(end='2020-1-10', periods=10, chunk_size=3, closed=closed)

            result = self.executor.execute_dataframe(dr, concat=True)[0]
            expected = pd.date_range(end='2020-1-10', periods=10, closed=closed)
            pd.testing.assert_index_equal(result, expected)

            # start, end, freq
            dr = md.date_range('2020-1-1', '2020-1-10', chunk_size=3, closed=closed)

            result = self.executor.execute_dataframe(dr, concat=True)[0]
            expected = pd.date_range('2020-1-1', '2020-1-10', closed=closed)
            pd.testing.assert_index_equal(result, expected)

            # start, end and periods
            dr = md.date_range('2020-1-1', '2020-1-10', periods=19,
                               chunk_size=3, closed=closed)

            result = self.executor.execute_dataframe(dr, concat=True)[0]
            expected = pd.date_range('2020-1-1', '2020-1-10', periods=19,
                                     closed=closed)
            pd.testing.assert_index_equal(result, expected)

            # start, end and freq
            dr = md.date_range('2020-1-1', '2020-1-10', freq='12H',
                               chunk_size=3, closed=closed)

            result = self.executor.execute_dataframe(dr, concat=True)[0]
            expected = pd.date_range('2020-1-1', '2020-1-10', freq='12H',
                                     closed=closed)
            pd.testing.assert_index_equal(result, expected)

        # test timezone
        dr = md.date_range('2020-1-1', periods=10, tz='Asia/Shanghai', chunk_size=7)

        result = self.executor.execute_dataframe(dr, concat=True)[0]
        expected = pd.date_range('2020-1-1', periods=10, tz='Asia/Shanghai')
        pd.testing.assert_index_equal(result, expected)

        # test periods=0
        dr = md.date_range('2020-1-1', periods=0)

        result = self.executor.execute_dataframe(dr, concat=True)[0]
        expected = pd.date_range('2020-1-1', periods=0)
        pd.testing.assert_index_equal(result, expected)

        # test start == end
        dr = md.date_range('2020-1-1', '2020-1-1', periods=1)

        result = self.executor.execute_dataframe(dr, concat=True)[0]
        expected = pd.date_range('2020-1-1', '2020-1-1', periods=1)
        pd.testing.assert_index_equal(result, expected)

        # test normalize=True
        dr = md.date_range('2020-1-1', periods=10, normalize=True, chunk_size=4)

        result = self.executor.execute_dataframe(dr, concat=True)[0]
        expected = pd.date_range('2020-1-1', periods=10, normalize=True)
        pd.testing.assert_index_equal(result, expected)

        # test freq
        dr = md.date_range(start='1/1/2018', periods=5, freq='M', chunk_size=3)

        result = self.executor.execute_dataframe(dr, concat=True)[0]
        expected = pd.date_range(start='1/1/2018', periods=5, freq='M')
        pd.testing.assert_index_equal(result, expected)

    @unittest.skipIf(pa is None or fastparquet is None, 'pyarrow or fastparquet not installed')
    def testReadParquet(self):
        test_df = pd.DataFrame({'a': np.arange(10).astype(np.int64, copy=False),
                                'b': [f's{i}' for i in range(10)],
                                'c': np.random.rand(10),})

        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.csv')
            test_df.to_parquet(file_path)

            df = md.read_parquet(file_path)
            result = self.executor.execute_dataframe(df, concat=True)[0]
            pd.testing.assert_frame_equal(result, test_df)

        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.csv')
            test_df.to_parquet(file_path, row_group_size=3)

            df = md.read_parquet(file_path, groups_as_chunks=True, columns=['a', 'b'])
            result = self.executor.execute_dataframe(df, concat=True)[0]
            pd.testing.assert_frame_equal(result.reset_index(drop=True), test_df[['a', 'b']])

        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.csv')
            test_df.to_parquet(file_path, row_group_size=5)

            df = md.read_parquet(file_path, groups_as_chunks=True,
                                 use_arrow_dtype=True,
                                 incremental_index=True)
            result = self.executor.execute_dataframe(df, concat=True)[0]
            self.assertIsInstance(df.dtypes.iloc[1], md.ArrowStringDtype)
            self.assertIsInstance(result.dtypes.iloc[1], md.ArrowStringDtype)
            pd.testing.assert_frame_equal(arrow_array_to_objects(result), test_df)

        # test fastparquet engine
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.csv')
            test_df.to_parquet(file_path, compression=None)

            df = md.read_parquet(file_path, engine='fastparquet')
            result = self.executor.execute_dataframe(df, concat=True)[0]
            pd.testing.assert_frame_equal(result, test_df)

        # test wildcards in path
        with tempfile.TemporaryDirectory() as tempdir:
            df = pd.DataFrame({'a': np.arange(300).astype(np.int64, copy=False),
                               'b': [f's{i}' for i in range(300)],
                               'c': np.random.rand(300), })

            file_paths = [os.path.join(tempdir, f'test{i}.parquet') for i in range(3)]
            df[:100].to_parquet(file_paths[0], row_group_size=50)
            df[100:200].to_parquet(file_paths[1], row_group_size=30)
            df[200:].to_parquet(file_paths[2])

            mdf = md.read_parquet(f'{tempdir}/*.parquet')
            r = self.executor.execute_dataframe(mdf, concat=True)[0]
            pd.testing.assert_frame_equal(df, r.sort_values('a').reset_index(drop=True))

            mdf = md.read_parquet(f'{tempdir}/*.parquet', groups_as_chunks=True)
            r = self.executor.execute_dataframe(mdf, concat=True)[0]
            pd.testing.assert_frame_equal(df, r.sort_values('a').reset_index(drop=True))
