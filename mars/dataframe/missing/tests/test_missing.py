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

import numpy as np
import pandas as pd

import mars.dataframe as md
from mars.core.operand import OperandStage
from mars.tests.core import TestBase


class Test(TestBase):
    def testFillNA(self):
        df_raw = pd.DataFrame(np.nan, index=range(0, 20), columns=list('ABCDEFGHIJ'))
        for _ in range(20):
            df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)
        value_df_raw = pd.DataFrame(np.random.randint(0, 100, (10, 7)).astype(np.float32),
                                    columns=list('ABCDEFG'))
        series_raw = pd.Series(np.nan, index=range(20))
        for _ in range(3):
            series_raw.iloc[random.randint(0, 19)] = random.randint(0, 99)
        value_series_raw = pd.Series(np.random.randint(0, 100, (10,)).astype(np.float32),
                                     index=list('ABCDEFGHIJ'))

        df = md.DataFrame(df_raw)
        series = md.Series(series_raw)

        # when nothing supplied, raise
        with self.assertRaises(ValueError):
            df.fillna()
        # when both values and methods supplied, raises
        with self.assertRaises(ValueError):
            df.fillna(value=1, method='ffill')
        # when call on series, cannot supply DataFrames
        with self.assertRaises(ValueError):
            series.fillna(value=df)
        with self.assertRaises(ValueError):
            series.fillna(value=df_raw)
        with self.assertRaises(NotImplementedError):
            series.fillna(value=series_raw, downcast='infer')
        with self.assertRaises(NotImplementedError):
            series.ffill(limit=1)

        df2 = df.fillna(value_series_raw).tiles()
        self.assertEqual(len(df2.chunks), 1)
        self.assertEqual(df2.chunks[0].shape, df2.shape)
        self.assertIsNone(df2.chunks[0].op.stage)

        series2 = series.fillna(value_series_raw).tiles()
        self.assertEqual(len(series2.chunks), 1)
        self.assertEqual(series2.chunks[0].shape, series2.shape)
        self.assertIsNone(series2.chunks[0].op.stage)

        df = md.DataFrame(df_raw, chunk_size=5)
        df2 = df.fillna(value_series_raw).tiles()
        self.assertEqual(len(df2.chunks), 8)
        self.assertEqual(df2.chunks[0].shape, (5, 5))
        self.assertIsNone(df2.chunks[0].op.stage)

        series = md.Series(series_raw, chunk_size=5)
        series2 = series.fillna(value_series_raw).tiles()
        self.assertEqual(len(series2.chunks), 4)
        self.assertEqual(series2.chunks[0].shape, (5,))
        self.assertIsNone(series2.chunks[0].op.stage)

        df2 = df.ffill(axis='columns').tiles()
        self.assertEqual(len(df2.chunks), 8)
        self.assertEqual(df2.chunks[0].shape, (5, 5))
        self.assertEqual(df2.chunks[0].op.axis, 1)
        self.assertEqual(df2.chunks[0].op.stage, OperandStage.combine)
        self.assertEqual(df2.chunks[0].op.method, 'ffill')
        self.assertIsNone(df2.chunks[0].op.limit)

        series2 = series.bfill().tiles()
        self.assertEqual(len(series2.chunks), 4)
        self.assertEqual(series2.chunks[0].shape, (5,))
        self.assertEqual(series2.chunks[0].op.stage, OperandStage.combine)
        self.assertEqual(series2.chunks[0].op.method, 'bfill')
        self.assertIsNone(series2.chunks[0].op.limit)

        value_df = md.DataFrame(value_df_raw, chunk_size=7)
        value_series = md.Series(value_series_raw, chunk_size=7)

        df2 = df.fillna(value_df).tiles()
        self.assertEqual(df2.shape, df.shape)
        self.assertIsNone(df2.chunks[0].op.stage)

        df2 = df.fillna(value_series).tiles()
        self.assertEqual(df2.shape, df.shape)
        self.assertIsNone(df2.chunks[0].op.stage)

        value_series_raw.index = list(range(10))
        value_series = md.Series(value_series_raw)
        series2 = series.fillna(value_series).tiles()
        self.assertEqual(series2.shape, series.shape)
        self.assertIsNone(series2.chunks[0].op.stage)

    def testDropNA(self):
        # dataframe cases
        df_raw = pd.DataFrame(np.nan, index=range(0, 20), columns=list('ABCDEFGHIJ'))
        for _ in range(30):
            df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)
        for rowid in range(random.randint(1, 5)):
            row = random.randint(0, 19)
            for idx in range(0, 10):
                df_raw.iloc[row, idx] = random.randint(0, 99)

        # not supporting drop with axis=1
        with self.assertRaises(NotImplementedError):
            md.DataFrame(df_raw).dropna(axis=1)

        # only one chunk in columns, can run dropna directly
        r = md.DataFrame(df_raw, chunk_size=(4, 10)).dropna().tiles()
        self.assertEqual(r.shape, (np.nan, 10))
        self.assertEqual(r.nsplits, ((np.nan,) * 5, (10,)))
        for c in r.chunks:
            self.assertIsInstance(c.op, type(r.op))
            self.assertEqual(len(c.inputs), 1)
            self.assertEqual(len(c.inputs[0].inputs), 0)
            self.assertEqual(c.shape, (np.nan, 10))

        # multiple chunks in columns, count() will be called first
        r = md.DataFrame(df_raw, chunk_size=4).dropna().tiles()
        self.assertEqual(r.shape, (np.nan, 10))
        self.assertEqual(r.nsplits, ((np.nan,) * 5, (4, 4, 2)))
        for c in r.chunks:
            self.assertIsInstance(c.op, type(r.op))
            self.assertEqual(len(c.inputs), 2)
            self.assertEqual(len(c.inputs[0].inputs), 0)
            self.assertEqual(c.inputs[1].op.stage, OperandStage.agg)
            self.assertTrue(np.isnan(c.shape[0]))

        # series cases
        series_raw = pd.Series(np.nan, index=range(20))
        for _ in range(10):
            series_raw.iloc[random.randint(0, 19)] = random.randint(0, 99)

        r = md.Series(series_raw, chunk_size=4).dropna().tiles()
        self.assertEqual(r.shape, (np.nan,))
        self.assertEqual(r.nsplits, ((np.nan,) * 5,))
        for c in r.chunks:
            self.assertIsInstance(c.op, type(r.op))
            self.assertEqual(len(c.inputs), 1)
            self.assertEqual(len(c.inputs[0].inputs), 0)
            self.assertEqual(c.shape, (np.nan,))

    def testReplace(self):
        # dataframe cases
        df_raw = pd.DataFrame(-1, index=range(0, 20), columns=list('ABCDEFGHIJ'))
        for _ in range(30):
            df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)
        for rowid in range(random.randint(1, 5)):
            row = random.randint(0, 19)
            for idx in range(0, 10):
                df_raw.iloc[row, idx] = random.randint(0, 99)

        # not supporting fill with limit
        df = md.DataFrame(df_raw, chunk_size=4)
        with self.assertRaises(NotImplementedError):
            df.replace(-1, method='ffill', limit=5)

        r = df.replace(-1, method='ffill').tiles()
        self.assertEqual(len(r.chunks), 15)
        self.assertEqual(r.chunks[0].shape, (4, 4))
        self.assertEqual(r.chunks[0].op.stage, OperandStage.combine)
        self.assertEqual(r.chunks[0].op.method, 'ffill')
        self.assertIsNone(r.chunks[0].op.limit)
        self.assertEqual(r.chunks[-1].inputs[-1].shape, (1, 2))
        self.assertEqual(r.chunks[-1].inputs[-1].op.stage, OperandStage.map)
        self.assertEqual(r.chunks[-1].inputs[-1].op.method, 'ffill')
        self.assertIsNone(r.chunks[-1].inputs[-1].op.limit)

        r = df.replace(-1, 99).tiles()
        self.assertEqual(len(r.chunks), 15)
        self.assertEqual(r.chunks[0].shape, (4, 4))
        self.assertIsNone(r.chunks[0].op.stage)
        self.assertIsNone(r.chunks[0].op.limit)

        # series cases
        series_raw = pd.Series(-1, index=range(20))
        for _ in range(10):
            series_raw.iloc[random.randint(0, 19)] = random.randint(0, 99)
        series = md.Series(series_raw, chunk_size=4)

        r = series.replace(-1, method='ffill').tiles()
        self.assertEqual(len(r.chunks), 5)
        self.assertEqual(r.chunks[0].shape, (4,))
        self.assertEqual(r.chunks[0].op.stage, OperandStage.combine)
        self.assertEqual(r.chunks[0].op.method, 'ffill')
        self.assertIsNone(r.chunks[0].op.limit)
        self.assertEqual(r.chunks[-1].inputs[-1].shape, (1,))
        self.assertEqual(r.chunks[-1].inputs[-1].op.stage, OperandStage.map)
        self.assertEqual(r.chunks[-1].inputs[-1].op.method, 'ffill')
        self.assertIsNone(r.chunks[-1].inputs[-1].op.limit)

        r = series.replace(-1, 99).tiles()
        self.assertEqual(len(r.chunks), 5)
        self.assertEqual(r.chunks[0].shape, (4,))
        self.assertIsNone(r.chunks[0].op.stage)
        self.assertIsNone(r.chunks[0].op.limit)
