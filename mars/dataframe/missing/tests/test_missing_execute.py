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

import random
import re
import string

import numpy as np
import pandas as pd
try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

import mars.dataframe as md
from mars.tests.core import TestBase


class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.ctx, self.executor = self._create_test_context()

    def testCheckNAExecution(self):
        df_raw = pd.DataFrame(np.nan, index=range(0, 20), columns=list('ABCDEFGHIJ'))
        for _ in range(20):
            df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)

        df = md.DataFrame(df_raw, chunk_size=4)

        pd.testing.assert_frame_equal(self.executor.execute_dataframe(df.isna(), concat=True)[0],
                                      df_raw.isna())
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(df.notna(), concat=True)[0],
                                      df_raw.notna())

        series_raw = pd.Series(np.nan, index=range(20))
        for _ in range(3):
            series_raw.iloc[random.randint(0, 19)] = random.randint(0, 99)

        series = md.Series(series_raw, chunk_size=4)

        pd.testing.assert_series_equal(self.executor.execute_dataframe(series.isna(), concat=True)[0],
                                       series_raw.isna())
        pd.testing.assert_series_equal(self.executor.execute_dataframe(series.notna(), concat=True)[0],
                                       series_raw.notna())

    def testDataFrameFillNAExecution(self):
        df_raw = pd.DataFrame(np.nan, index=range(0, 20), columns=list('ABCDEFGHIJ'))
        for _ in range(20):
            df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)
        value_df_raw = pd.DataFrame(np.random.randint(0, 100, (10, 7)).astype(np.float32),
                                    columns=list('ABCDEFG'))
        df = md.DataFrame(df_raw)

        # test DataFrame single chunk with numeric fill
        r = df.fillna(1)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.fillna(1))

        # test DataFrame single chunk with value as single chunk
        value_df = md.DataFrame(value_df_raw)
        r = df.fillna(value_df)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.fillna(value_df_raw))

        df = md.DataFrame(df_raw, chunk_size=3)

        # test chunked with numeric fill
        r = df.fillna(1)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.fillna(1))

        # test forward fill in axis=0 without limit
        r = df.fillna(method='pad')
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.fillna(method='pad'))

        # test backward fill in axis=0 without limit
        r = df.fillna(method='backfill')
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.fillna(method='backfill'))

        # test forward fill in axis=1 without limit
        r = df.ffill(axis=1)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.ffill(axis=1))

        # test backward fill in axis=1 without limit
        r = df.bfill(axis=1)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.bfill(axis=1))

        # test fill with dataframe
        value_df = md.DataFrame(value_df_raw, chunk_size=4)
        r = df.fillna(value_df)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.fillna(value_df_raw))

        # test fill with series
        value_series_raw = pd.Series(np.random.randint(0, 100, (10,)).astype(np.float32),
                                     index=list('ABCDEFGHIJ'))
        value_series = md.Series(value_series_raw, chunk_size=4)
        r = df.fillna(value_series)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.fillna(value_series_raw))

        # test inplace tile
        df.fillna(1, inplace=True)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(df, concat=True)[0],
                                      df_raw.fillna(1))

    def testSeriesFillNAExecution(self):
        series_raw = pd.Series(np.nan, index=range(20))
        for _ in range(3):
            series_raw.iloc[random.randint(0, 19)] = random.randint(0, 99)
        value_series_raw = pd.Series(np.random.randint(0, 100, (10,)).astype(np.float32))

        # test single chunk
        series = md.Series(series_raw)

        r = series.fillna(1)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       series_raw.fillna(1))

        # test single chunk with value as single chunk
        value_series = md.Series(value_series_raw)
        r = series.fillna(value_series)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       series_raw.fillna(value_series_raw))

        series = md.Series(series_raw, chunk_size=3)

        # test chunked with numeric fill
        r = series.fillna(1)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       series_raw.fillna(1))

        # test forward fill in axis=0 without limit
        r = series.fillna(method='pad')
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       series_raw.fillna(method='pad'))

        # test backward fill in axis=0 without limit
        r = series.fillna(method='backfill')
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       series_raw.fillna(method='backfill'))

        # test fill with series
        value_df = md.Series(value_series_raw, chunk_size=4)
        r = series.fillna(value_df)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       series_raw.fillna(value_series_raw))

        # test inplace tile
        series.fillna(1, inplace=True)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(series, concat=True)[0],
                                       series_raw.fillna(1))

    def testDropNAExecution(self):
        # dataframe cases
        df_raw = pd.DataFrame(np.nan, index=range(0, 20), columns=list('ABCDEFGHIJ'))
        for _ in range(30):
            df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)
        for rowid in range(random.randint(1, 5)):
            row = random.randint(0, 19)
            for idx in range(0, 10):
                df_raw.iloc[row, idx] = random.randint(0, 99)

        # only one chunk in columns, can run dropna directly
        r = md.DataFrame(df_raw, chunk_size=(4, 10)).dropna()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.dropna())

        # multiple chunks in columns, count() will be called first
        r = md.DataFrame(df_raw, chunk_size=4).dropna()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.dropna())

        r = md.DataFrame(df_raw, chunk_size=4).dropna(how='all')
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.dropna(how='all'))

        r = md.DataFrame(df_raw, chunk_size=4).dropna(subset=list('ABFI'))
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.dropna(subset=list('ABFI')))

        r = md.DataFrame(df_raw, chunk_size=4).dropna(how='all', subset=list('BDHJ'))
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.dropna(how='all', subset=list('BDHJ')))

        r = md.DataFrame(df_raw, chunk_size=4)
        r.dropna(how='all', inplace=True)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.dropna(how='all'))

        # series cases
        series_raw = pd.Series(np.nan, index=range(20))
        for _ in range(10):
            series_raw.iloc[random.randint(0, 19)] = random.randint(0, 99)

        r = md.Series(series_raw, chunk_size=4).dropna()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       series_raw.dropna())

        r = md.Series(series_raw, chunk_size=4)
        r.dropna(inplace=True)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       series_raw.dropna())

    def testReplaceExecution(self):
        # dataframe cases
        df_raw = pd.DataFrame(-1, index=range(0, 20), columns=list('ABCDEFGHIJ'))
        for _ in range(30):
            df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)
        for rowid in range(random.randint(1, 5)):
            row = random.randint(0, 19)
            for idx in range(0, 10):
                df_raw.iloc[row, idx] = random.randint(0, 99)
        df = md.DataFrame(df_raw, chunk_size=4)

        r = df.replace(-1, method='ffill')
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.replace(-1, method='ffill'))

        r = df.replace(-1, method='bfill')
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.replace(-1, method='bfill'))

        r = df.replace(-1, 999)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.replace(-1, 999))

        raw_to_replace = pd.Series([-1, 1, 2])
        to_replace_series = md.Series(raw_to_replace)
        raw_value = pd.Series([2, 3, -1])
        value_series = md.Series(raw_value)
        r = df.replace(to_replace_series, value_series)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.replace(raw_to_replace, raw_value))

        df.replace({'A': -1}, {'A': 9}, inplace=True)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(df, concat=True)[0],
                                      df_raw.replace({'A': -1}, {'A': 9}))

        # series cases
        series_raw = pd.Series(-1, index=range(20))
        for _ in range(10):
            series_raw.iloc[random.randint(0, 19)] = random.randint(0, 99)
        series = md.Series(series_raw, chunk_size=4)

        r = series.replace(-1, method='ffill')
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       series_raw.replace(-1, method='ffill'))

        r = series.replace(-1, method='bfill')
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       series_raw.replace(-1, method='bfill'))

        r = series.replace(-1, 999)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       series_raw.replace(-1, 999))

        # str series cases
        tmpl_chars = list(string.ascii_letters + string.digits)
        random.shuffle(tmpl_chars)

        def _rand_slice():
            lb = random.randint(0, len(tmpl_chars) - 1)
            rb = random.randint(lb, len(tmpl_chars) - 1)
            return ''.join(tmpl_chars[lb:rb + 1])

        series_raw = pd.Series([_rand_slice() for _ in range(20)])
        series = md.Series(series_raw, chunk_size=4)

        regs = [re.compile(r'.A.', flags=re.IGNORECASE),
                re.compile(r'.B.', flags=re.IGNORECASE),
                re.compile(r'.C.', flags=re.IGNORECASE),
                re.compile(r'.D.', flags=re.IGNORECASE), ]
        r = series.replace(regex=regs, value='new')
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       series_raw.replace(regex=regs, value='new'))
