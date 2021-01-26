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

import contextlib
import os
import tempfile

import numpy as np
import pandas as pd

import mars.dataframe as md
from mars.config import option_context
from mars.dataframe.indexing.iloc import DataFrameIlocGetItem, SeriesIlocGetItem
from mars.executor import register, Executor
from mars.tests.core import TestBase


class Test(TestBase):
    def setUp(self):
        self.ctx, self.executor = self._create_test_context()
        rs = np.random.RandomState(0)
        self.df = pd.DataFrame({'a': rs.randint(10, size=100),
                                'b': rs.rand(100),
                                'c': rs.choice(list('abc'), size=100)})

    @contextlib.contextmanager
    def _raise_iloc(self):
        def _execute_iloc(*_):  # pragma: no cover
            raise ValueError('cannot run iloc')

        self.ctx.__enter__()
        try:
            register(DataFrameIlocGetItem, _execute_iloc)
            register(SeriesIlocGetItem, _execute_iloc)

            yield
        finally:
            del Executor._op_runners[DataFrameIlocGetItem]
            del Executor._op_runners[SeriesIlocGetItem]
            self.ctx.__exit__(None, None, None)

    def testReadCSVHead(self):
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.csv')

            df = self.df
            df.to_csv(file_path, index=False)

            size = os.stat(file_path).st_size / 2
            mdf = md.read_csv(file_path, chunk_bytes=size)

            with self._raise_iloc():
                hdf = mdf.head(5)
                expected = df.head(5)
                pd.testing.assert_frame_equal(hdf.execute().fetch(), expected)

                with self.assertRaises(ValueError) as cm:
                    # need iloc
                    mdf.head(99).execute()

                self.assertIn('cannot run iloc', str(cm.exception))

            with self._raise_iloc():
                s = mdf.head(5).sum()
                expected = df.head(5).sum()
                pd.testing.assert_series_equal(s.execute().fetch(), expected)

            pd.testing.assert_frame_equal(
                mdf.head(99).execute().fetch().reset_index(drop=True), df.head(99))

    def testReadParquetHead(self):
        with tempfile.TemporaryDirectory() as tempdir:
            df = self.df
            dirname = os.path.join(tempdir, 'test_parquet')
            os.makedirs(dirname)
            for i in range(3):
                file_path = os.path.join(dirname , f'test{i}.parquet')
                df[i * 40: (i + 1) * 40].to_parquet(file_path, index=False)

            mdf = md.read_parquet(dirname)

            with self._raise_iloc():
                hdf = mdf.head(5)
                expected = df.head(5)
                pd.testing.assert_frame_equal(hdf.execute().fetch(), expected)

                with self.assertRaises(ValueError) as cm:
                    # need iloc
                    mdf.head(99).execute()

                self.assertIn('cannot run iloc', str(cm.exception))

            pd.testing.assert_frame_equal(
                mdf.head(99).execute().fetch().reset_index(drop=True), df.head(99))

    def testSortHead(self):
        mdf = md.DataFrame(self.df, chunk_size=20)
        df2 = self.df.copy()
        df2.set_index('b', inplace=True)
        mdf2 = md.DataFrame(df2, chunk_size=20)

        with self._raise_iloc():
            hdf = mdf.sort_values(by='b').head(10)
            expected = self.df.sort_values(by='b').head(10)
            pd.testing.assert_frame_equal(hdf.execute().fetch(), expected)

            hdf = mdf2.sort_index().head(10)
            expected = df2.sort_index().head(10)
            pd.testing.assert_frame_equal(hdf.execute().fetch(), expected)

            with option_context({'optimize.head_optimize_threshold': 9}):
                with self.assertRaises(ValueError) as cm:
                    mdf.sort_values(by='b').head(10).execute()
                self.assertIn('cannot run iloc', str(cm.exception))

        with option_context({'optimize.head_optimize_threshold': 9}):
            hdf = mdf.sort_values(by='b').head(11)
            expected = self.df.sort_values(by='b').head(11)
            pd.testing.assert_frame_equal(hdf.execute().fetch(), expected)

    def testValueCountsHead(self):
        for chunk_size in (100, 20):
            mdf = md.DataFrame(self.df, chunk_size=chunk_size)

            with self._raise_iloc():
                hdf = mdf['a'].value_counts().head(3)
                expected = self.df['a'].value_counts().head(3)
                pd.testing.assert_series_equal(hdf.execute().fetch(), expected)

                if chunk_size == 20:
                    with self.assertRaises(ValueError) as cm:
                        mdf['a'].value_counts(sort=False).head(3).execute()
                    self.assertIn('cannot run iloc', str(cm.exception))
