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

import numpy as np
import pandas as pd

from mars.operands import OperandStage
from mars.executor import Executor
from mars.tests.core import TestBase
from mars.dataframe.datasource.dataframe import from_pandas
from mars.dataframe.merge import DataFrameMergeAlign, DataFrameShuffleMerge


class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.executor = Executor()

    def testMerge(self):
        df1 = pd.DataFrame(np.arange(20).reshape((4, 5)) + 1, columns=['a', 'b', 'c', 'd', 'e'])
        df2 = pd.DataFrame(np.arange(20).reshape((5, 4)) + 1, columns=['a', 'b', 'x', 'y'])

        mdf1 = from_pandas(df1, chunk_size=2)
        mdf2 = from_pandas(df2, chunk_size=3)

        parameters = [
            {},
            {'how': 'left', 'right_on': 'x', 'left_index': True},
            {'how': 'right', 'left_on': 'a', 'right_index': True},
            {'how': 'left', 'left_on': 'a', 'right_on': 'x'},
            {'how': 'right', 'left_on': 'a', 'right_index': True},
            {'how': 'right', 'on': 'a'},
            {'how': 'inner', 'on': ['a', 'b']},
        ]

        for kw in parameters:
            df = mdf1.merge(mdf2, **kw)
            df = df.tiles()

            self.assertEqual(df.chunk_shape, (2, 1))
            for chunk in df.chunks:
                self.assertIsInstance(chunk.op, DataFrameShuffleMerge)
                self.assertEqual(chunk.op.how, kw.get('how', 'inner'))
                left, right = chunk.op.inputs
                self.assertIsInstance(left.op, DataFrameMergeAlign)
                self.assertEqual(left.op.stage, OperandStage.reduce)
                self.assertIsInstance(right.op, DataFrameMergeAlign)
                self.assertEqual(right.op.stage, OperandStage.reduce)
                self.assertEqual(len(left.inputs[0].inputs), 2)
                self.assertEqual(len(right.inputs[0].inputs), 2)
                for lchunk in left.inputs[0].inputs:
                    self.assertIsInstance(lchunk.op, DataFrameMergeAlign)
                    self.assertEqual(lchunk.op.stage, OperandStage.map)
                    self.assertEqual(lchunk.op.index_shuffle_size, 2)
                    self.assertEqual(lchunk.op.shuffle_on, kw.get('on', None) or kw.get('left_on', None))
                for rchunk in right.inputs[0].inputs:
                    self.assertIsInstance(rchunk.op, DataFrameMergeAlign)
                    self.assertEqual(rchunk.op.stage, OperandStage.map)
                    self.assertEqual(rchunk.op.index_shuffle_size, 2)
                    self.assertEqual(rchunk.op.shuffle_on, kw.get('on', None) or kw.get('right_on', None))
                pd.testing.assert_index_equal(chunk.columns_value.to_pandas(), df.columns_value.to_pandas())

    def testJoin(self):
        df1 = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]], index=['a1', 'a2', 'a3'])
        df2 = pd.DataFrame([[1, 2, 3], [1, 5, 6], [7, 8, 9]], index=['a1', 'b2', 'b3']) + 1
        df2 = pd.concat([df2, df2 + 1])

        mdf1 = from_pandas(df1, chunk_size=2)
        mdf2 = from_pandas(df2, chunk_size=2)

        parameters = [
            {'lsuffix': 'l_', 'rsuffix': 'r_'},
            {'lsuffix': 'l_', 'rsuffix': 'r_', 'how': 'left'},
            {'lsuffix': 'l_', 'rsuffix': 'r_', 'how': 'right'},
            {'lsuffix': 'l_', 'rsuffix': 'r_', 'how': 'inner'},
            {'lsuffix': 'l_', 'rsuffix': 'r_', 'how': 'left'},
        ]

        for kw in parameters:
            df = mdf1.join(mdf2, **kw)
            df = df.tiles()

            self.assertEqual(df.chunk_shape, (3, 1))
            for chunk in df.chunks:
                self.assertIsInstance(chunk.op, DataFrameShuffleMerge)
                self.assertEqual(chunk.op.how, kw.get('how', 'left'))
                left, right = chunk.op.inputs
                self.assertIsInstance(left.op, DataFrameMergeAlign)
                self.assertEqual(left.op.stage, OperandStage.reduce)
                self.assertIsInstance(right.op, DataFrameMergeAlign)
                self.assertEqual(right.op.stage, OperandStage.reduce)
                self.assertEqual(len(left.inputs[0].inputs), 2)
                self.assertEqual(len(right.inputs[0].inputs), 3)
                for lchunk in left.inputs[0].inputs:
                    self.assertIsInstance(lchunk.op, DataFrameMergeAlign)
                    self.assertEqual(lchunk.op.stage, OperandStage.map)
                    self.assertEqual(lchunk.op.index_shuffle_size, 3)
                    self.assertEqual(lchunk.op.shuffle_on, None)
                for rchunk in right.inputs[0].inputs:
                    self.assertIsInstance(rchunk.op, DataFrameMergeAlign)
                    self.assertEqual(rchunk.op.stage, OperandStage.map)
                    self.assertEqual(rchunk.op.index_shuffle_size, 3)
                    self.assertEqual(rchunk.op.shuffle_on, None)
                pd.testing.assert_index_equal(chunk.columns_value.to_pandas(), df.columns_value.to_pandas())

    def testJoinOn(self):
        df1 = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]], columns=['a1', 'a2', 'a3'])
        df2 = pd.DataFrame([[1, 2, 3], [1, 5, 6], [7, 8, 9]], columns=['a1', 'b2', 'b3']) + 1
        df2 = pd.concat([df2, df2 + 1])

        mdf1 = from_pandas(df1, chunk_size=2)
        mdf2 = from_pandas(df2, chunk_size=2)

        parameters = [
            {'lsuffix': 'l_', 'rsuffix': 'r_'},
            {'lsuffix': 'l_', 'rsuffix': 'r_', 'how': 'left', 'on': 'a1'},
            {'lsuffix': 'l_', 'rsuffix': 'r_', 'how': 'right', 'on': 'a2'},
            {'lsuffix': 'l_', 'rsuffix': 'r_', 'how': 'inner', 'on': 'a2'},
            {'lsuffix': 'l_', 'rsuffix': 'r_', 'how': 'outer', 'on': 'a2'},
        ]

        for kw in parameters:
            df = mdf1.join(mdf2, **kw)
            df = df.tiles()

            self.assertEqual(df.chunk_shape, (3, 1))
            for chunk in df.chunks:
                self.assertIsInstance(chunk.op, DataFrameShuffleMerge)
                self.assertEqual(chunk.op.how, kw.get('how', 'left'))
                left, right = chunk.op.inputs
                self.assertIsInstance(left.op, DataFrameMergeAlign)
                self.assertEqual(left.op.stage, OperandStage.reduce)
                self.assertIsInstance(right.op, DataFrameMergeAlign)
                self.assertEqual(right.op.stage, OperandStage.reduce)
                self.assertEqual(len(left.inputs[0].inputs), 2)
                self.assertEqual(len(right.inputs[0].inputs), 3)
                for lchunk in left.inputs[0].inputs:
                    self.assertIsInstance(lchunk.op, DataFrameMergeAlign)
                    self.assertEqual(lchunk.op.stage, OperandStage.map)
                    self.assertEqual(lchunk.op.index_shuffle_size, 3)
                    self.assertEqual(lchunk.op.shuffle_on, kw.get('on', None))
                for rchunk in right.inputs[0].inputs:
                    self.assertIsInstance(rchunk.op, DataFrameMergeAlign)
                    self.assertEqual(rchunk.op.stage, OperandStage.map)
                    self.assertEqual(rchunk.op.index_shuffle_size, 3)
                    self.assertEqual(rchunk.op.shuffle_on, None)
                pd.testing.assert_index_equal(chunk.columns_value.to_pandas(), df.columns_value.to_pandas())
