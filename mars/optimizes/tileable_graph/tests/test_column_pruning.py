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
import shutil
import tempfile

import pandas as pd

import mars.dataframe as md
from mars.core import ExecutableTuple
from mars.config import option_context
from mars.tests.core import TestBase
from mars.optimizes.tileable_graph.core import optimized_result_tileable


class Test(TestBase):
    def testGroupByPruneReadCSV(self):
        tempdir = tempfile.mkdtemp()
        file_path = os.path.join(tempdir, 'test.csv')
        try:
            df = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                               'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                               'c': list('aabaaddce'),
                               'd': list('abaaaddce')})
            df.to_csv(file_path, index=False)

            mdf = md.read_csv(file_path).groupby('c').agg({'a': 'sum'})
            expected = df.groupby('c').agg({'a': 'sum'})
            pd.testing.assert_frame_equal(mdf.execute(), expected)
            pd.testing.assert_frame_equal(mdf.fetch(), expected)

            optimized_df = optimized_result_tileable[mdf.data]
            self.assertEqual(optimized_df.inputs[0].op.usecols, ['a', 'c'])

            mdf = md.read_csv(file_path).groupby('c').agg({'b': 'sum'})
            expected = df.groupby('c').agg({'b': 'sum'})
            pd.testing.assert_frame_equal(mdf.execute(), expected)
            pd.testing.assert_frame_equal(mdf.fetch(), expected)

            optimized_df = optimized_result_tileable[mdf.data]
            self.assertEqual(optimized_df.inputs[0].op.usecols, ['b', 'c'])

            mdf = md.read_csv(file_path).groupby('c').agg({'b': 'sum'}) + 1
            expected = df.groupby('c').agg({'b': 'sum'}) + 1
            pd.testing.assert_frame_equal(mdf.execute(), expected)
            pd.testing.assert_frame_equal(mdf.fetch(), expected)

            mdf = md.read_csv(file_path, usecols=['a', 'b', 'c']).groupby('c').agg({'b': 'sum'})
            expected = df.groupby('c').agg({'b': 'sum'})
            pd.testing.assert_frame_equal(mdf.execute(), expected)
            pd.testing.assert_frame_equal(mdf.fetch(), expected)
            optimized_df = optimized_result_tileable[mdf.data]
            self.assertEqual(optimized_df.inputs[0].op.usecols, ['b', 'c'])

            in_df = md.read_csv(file_path)
            df1 = in_df.groupby('c').agg({'b': 'sum'})
            df2 = in_df.groupby('b').agg({'a': 'sum'})

            dfs = ExecutableTuple((df1, df2))
            results = dfs.execute()
            expected1 = df.groupby('c').agg({'b': 'sum'})
            expected2 = df.groupby('b').agg({'a': 'sum'})
            pd.testing.assert_frame_equal(results[0], expected1)
            pd.testing.assert_frame_equal(results[1], expected2)

            with option_context({'optimize_tileable_graph': False}):
                mdf = md.read_csv(file_path).groupby('c').agg({'b': 'sum'})
                expected = df.groupby('c').agg({'b': 'sum'})
                pd.testing.assert_frame_equal(mdf.execute(), expected)
                pd.testing.assert_frame_equal(mdf.fetch(), expected)

                tileable_graph = mdf.build_graph()
                self.assertIsNone(list(tileable_graph)[0].inputs[0].op.usecols)

        finally:
            shutil.rmtree(tempdir)
