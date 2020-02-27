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

from mars.tests.core import TestBase, ExecutorForTest
from mars.dataframe.datasource.dataframe import from_pandas
from mars.dataframe.utils import sort_dataframe_inplace


class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.executor = ExecutorForTest()

    def testMerge(self):
        df1 = pd.DataFrame(np.arange(20).reshape((4, 5)) + 1, columns=['a', 'b', 'c', 'd', 'e'])
        df2 = pd.DataFrame(np.arange(20).reshape((5, 4)) + 1, columns=['a', 'b', 'x', 'y'])

        mdf1 = from_pandas(df1, chunk_size=2)
        mdf2 = from_pandas(df2, chunk_size=2)

        # Note [Index of Merge]
        #
        # When `left_index` and `right_index` of `merge` is both false, pandas will generate an RangeIndex to
        # the final result dataframe.
        #
        # We chunked the `left` and `right` dataframe, thus every result chunk will have its own RangeIndex.
        # When they are contenated we don't generate a new RangeIndex for the result, thus we cannot obtain the
        # same index value with pandas. But we guarantee that the content of dataframe is correct.

        # merge on index
        expected0 = df1.merge(df2)
        jdf0 = mdf1.merge(mdf2)
        result0 = self.executor.execute_dataframe(jdf0, concat=True)[0]
        pd.testing.assert_frame_equal(sort_dataframe_inplace(expected0, 0), sort_dataframe_inplace(result0, 0))

        # merge on left index and `right_on`
        expected1 = df1.merge(df2, how='left', right_on='x', left_index=True)
        jdf1 = mdf1.merge(mdf2, how='left', right_on='x', left_index=True)
        result1 = self.executor.execute_dataframe(jdf1, concat=True)[0]
        expected1.set_index('a_x', inplace=True)
        result1.set_index('a_x', inplace=True)
        pd.testing.assert_frame_equal(sort_dataframe_inplace(expected1, 0), sort_dataframe_inplace(result1, 0))

        # merge on `left_on` and right index
        expected2 = df1.merge(df2, how='right', left_on='a', right_index=True)
        jdf2 = mdf1.merge(mdf2, how='right', left_on='a', right_index=True)
        result2 = self.executor.execute_dataframe(jdf2, concat=True)[0]
        expected2.set_index('a', inplace=True)
        result2.set_index('a', inplace=True)
        pd.testing.assert_frame_equal(sort_dataframe_inplace(expected2, 0), sort_dataframe_inplace(result2, 0))

        # merge on `left_on` and `right_on`
        expected3 = df1.merge(df2, how='left', left_on='a', right_on='x')
        jdf3 = mdf1.merge(mdf2, how='left', left_on='a', right_on='x')
        result3 = self.executor.execute_dataframe(jdf3, concat=True)[0]
        expected3.set_index('a_x', inplace=True)
        result3.set_index('a_x', inplace=True)
        pd.testing.assert_frame_equal(sort_dataframe_inplace(expected3, 0), sort_dataframe_inplace(result3, 0))

        # merge on `on`
        expected4 = df1.merge(df2, how='right', on='a')
        jdf4 = mdf1.merge(mdf2, how='right', on='a')
        result4 = self.executor.execute_dataframe(jdf4, concat=True)[0]
        expected4.set_index('a', inplace=True)
        result4.set_index('a', inplace=True)
        pd.testing.assert_frame_equal(sort_dataframe_inplace(expected4, 0), sort_dataframe_inplace(result4, 0))

        # merge on multiple columns
        expected5 = df1.merge(df2, how='inner', on=['a', 'b'])
        jdf5 = mdf1.merge(mdf2, how='inner', on=['a', 'b'])
        result5 = self.executor.execute_dataframe(jdf5, concat=True)[0]
        pd.testing.assert_frame_equal(sort_dataframe_inplace(expected5, 0), sort_dataframe_inplace(result5, 0))

    def testJoin(self):
        df1 = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]], index=['a1', 'a2', 'a3'])
        df2 = pd.DataFrame([[1, 2, 3], [1, 5, 6], [7, 8, 9]], index=['a1', 'b2', 'b3']) + 1
        df2 = pd.concat([df2, df2 + 1])

        mdf1 = from_pandas(df1, chunk_size=2)
        mdf2 = from_pandas(df2, chunk_size=2)

        # default `how`
        expected0 = df1.join(df2, lsuffix='l_', rsuffix='r_')
        jdf0 = mdf1.join(mdf2, lsuffix='l_', rsuffix='r_')
        result0 = self.executor.execute_dataframe(jdf0, concat=True)[0]
        pd.testing.assert_frame_equal(expected0.sort_index(), result0.sort_index())

        # how = 'left'
        expected1 = df1.join(df2, how='left', lsuffix='l_', rsuffix='r_')
        jdf1 = mdf1.join(mdf2, how='left', lsuffix='l_', rsuffix='r_')
        result1 = self.executor.execute_dataframe(jdf1, concat=True)[0]
        pd.testing.assert_frame_equal(expected1.sort_index(), result1.sort_index())

        # how = 'right'
        expected2 = df1.join(df2, how='right', lsuffix='l_', rsuffix='r_')
        jdf2 = mdf1.join(mdf2, how='right', lsuffix='l_', rsuffix='r_')
        result2 = self.executor.execute_dataframe(jdf2, concat=True)[0]
        pd.testing.assert_frame_equal(expected2.sort_index(), result2.sort_index())

        # how = 'inner'
        expected3 = df1.join(df2, how='inner', lsuffix='l_', rsuffix='r_')
        jdf3 = mdf1.join(mdf2, how='inner', lsuffix='l_', rsuffix='r_')
        result3 = self.executor.execute_dataframe(jdf3, concat=True)[0]
        pd.testing.assert_frame_equal(expected3.sort_index(), result3.sort_index())

        # how = 'outer'
        expected4 = df1.join(df2, how='outer', lsuffix='l_', rsuffix='r_')
        jdf4 = mdf1.join(mdf2, how='outer', lsuffix='l_', rsuffix='r_')
        result4 = self.executor.execute_dataframe(jdf4, concat=True)[0]
        pd.testing.assert_frame_equal(expected4.sort_index(), result4.sort_index())

    def testJoinOn(self):
        df1 = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]], columns=['a1', 'a2', 'a3'])
        df2 = pd.DataFrame([[1, 2, 3], [1, 5, 6], [7, 8, 9]], columns=['a1', 'b2', 'b3']) + 1
        df2 = pd.concat([df2, df2 + 1])

        mdf1 = from_pandas(df1, chunk_size=2)
        mdf2 = from_pandas(df2, chunk_size=2)

        expected0 = df1.join(df2, on=None, lsuffix='_l', rsuffix='_r')
        jdf0 = mdf1.join(mdf2, on=None, lsuffix='_l', rsuffix='_r')
        result0 = self.executor.execute_dataframe(jdf0, concat=True)[0]
        pd.testing.assert_frame_equal(sort_dataframe_inplace(expected0, 0), sort_dataframe_inplace(result0, 0))

        expected1 = df1.join(df2, how='left', on='a1', lsuffix='_l', rsuffix='_r')
        jdf1 = mdf1.join(mdf2, how='left', on='a1', lsuffix='_l', rsuffix='_r')
        result1 = self.executor.execute_dataframe(jdf1, concat=True)[0]

        # Note [Columns of Left Join]
        #
        # I believe we have no chance to obtain the entirely same result with pandas here:
        #
        # Look at the following example:
        #
        # >>> df1
        #     a1  a2  a3
        # 0   1   3   3
        # >>> df2
        #     a1  b2  b3
        # 1   2   6   7
        # >>> df3
        #     a1  b2  b3
        # 1   2   6   7
        # 1   2   6   7
        #
        # >>> df1.merge(df2, how='left', left_on='a1', left_index=False, right_index=True)
        #     a1_x  a2  a3  a1_y  b2  b3
        # 0   1   3   3     2   6   7
        # >>> df1.merge(df3, how='left', left_on='a1', left_index=False, right_index=True)
        #     a1  a1_x  a2  a3  a1_y  b2  b3
        # 0   1     1   3   3     2   6   7
        # 0   1     1   3   3     2   6   7
        #
        # Note that the result of `df1.merge(df3)` has an extra column `a` compared to `df1.merge(df2)`.
        # The value of column `a` is the same of `a1_x`, just because `1` occurs twice in index of `df3`.
        # I haven't invistagated why pandas has such behaviour...
        #
        # We cannot yeild the same result with pandas, because, the `df3` is chunked, then some of the
        # result chunk has 6 columns, others may have 7 columns, when concatenated into one DataFrame
        # some cells of column `a` will have value `NaN`, which is different from the result of pandas.
        #
        # But we can guarantee that other effective columns have absolutely same value with pandas.

        columns_to_compare = jdf1.columns_value.to_pandas()

        pd.testing.assert_frame_equal(sort_dataframe_inplace(expected1[columns_to_compare], 0, 1),
                                      sort_dataframe_inplace(result1[columns_to_compare], 0, 1))

        # Note [Index of Join on EmptyDataFrame]
        #
        # It is tricky that it is non-trivial to get the same `index` result with pandas.
        #
        # Look at the following example:
        #
        # >>> df1
        #    a1  a2  a3
        # 1   4   2   6
        # >>> df2
        #    a1  b2  b3
        # 1   2   6   7
        # 2   8   9  10
        # >>> df3
        # Empty DataFrame
        # Columns: [a1, a2, a3]
        # Index: []
        # >>> df1.join(df2, how='right', on='a2', lsuffix='_l', rsuffix='_r')
        #       a1_l  a2   a3  a1_r  b2  b3
        # 1.0   4.0   2  6.0     8   9  10
        # NaN   NaN   1  NaN     2   6   7
        # >>> df3.join(df2, how='right', on='a2', lsuffix='_l', rsuffix='_r')
        #     a1_l  a2  a3  a1_r  b2  b3
        # 1   NaN   1 NaN     2   6   7
        # 2   NaN   2 NaN     8   9  10
        #
        # When the `left` dataframe is not empty, the mismatched rows in `right` will have index value `NaN`,
        # and the matched rows have index value from `right`. When the `left` dataframe is empty, the mismatched
        # rows have index value from `right`.
        #
        # Since we chunked the `left` dataframe, it is uneasy to obtain the same index value with pandas in the
        # final result dataframe, but we guaranteed that the dataframe content is correctly.

        expected2 = df1.join(df2, how='right', on='a2', lsuffix='_l', rsuffix='_r')
        jdf2 = mdf1.join(mdf2, how='right', on='a2', lsuffix='_l', rsuffix='_r')
        result2 = self.executor.execute_dataframe(jdf2, concat=True)[0]

        expected2.set_index('a2', inplace=True)
        result2.set_index('a2', inplace=True)
        pd.testing.assert_frame_equal(sort_dataframe_inplace(expected2, 0), sort_dataframe_inplace(result2, 0))

        expected3 = df1.join(df2, how='inner', on='a2', lsuffix='_l', rsuffix='_r')
        jdf3 = mdf1.join(mdf2, how='inner', on='a2', lsuffix='_l', rsuffix='_r')
        result3 = self.executor.execute_dataframe(jdf3, concat=True)[0]
        pd.testing.assert_frame_equal(sort_dataframe_inplace(expected3, 0), sort_dataframe_inplace(result3, 0))

        expected4 = df1.join(df2, how='outer', on='a2', lsuffix='_l', rsuffix='_r')
        jdf4 = mdf1.join(mdf2, how='outer', on='a2', lsuffix='_l', rsuffix='_r')
        result4 = self.executor.execute_dataframe(jdf4, concat=True)[0]

        expected4.set_index('a2', inplace=True)
        result4.set_index('a2', inplace=True)
        pd.testing.assert_frame_equal(sort_dataframe_inplace(expected4, 0), sort_dataframe_inplace(result4, 0))

    def testMergeOneChunk(self):
        df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
                            'value': [1, 2, 3, 5]}, index=['a1', 'a2', 'a3', 'a4'])
        df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],
                            'value': [5, 6, 7, 8]}, index=['a1', 'a2', 'a3', 'a4'])

        # all have one chunk
        mdf1 = from_pandas(df1)
        mdf2 = from_pandas(df2)

        expected = df1.merge(df2, left_on='lkey', right_on='rkey')
        jdf = mdf1.merge(mdf2, left_on='lkey', right_on='rkey')
        result = self.executor.execute_dataframe(jdf, concat=True)[0]

        pd.testing.assert_frame_equal(expected.sort_values(by=expected.columns[1]).reset_index(drop=True),
                                      result.sort_values(by=result.columns[1]).reset_index(drop=True))

        # left have one chunk
        mdf1 = from_pandas(df1)
        mdf2 = from_pandas(df2, chunk_size=2)

        expected = df1.merge(df2, left_on='lkey', right_on='rkey')
        jdf = mdf1.merge(mdf2, left_on='lkey', right_on='rkey')
        result = self.executor.execute_dataframe(jdf, concat=True)[0]

        pd.testing.assert_frame_equal(expected.sort_values(by=expected.columns[1]).reset_index(drop=True),
                                      result.sort_values(by=result.columns[1]).reset_index(drop=True))

        # right have one chunk
        mdf1 = from_pandas(df1, chunk_size=3)
        mdf2 = from_pandas(df2)

        expected = df1.merge(df2, left_on='lkey', right_on='rkey')
        jdf = mdf1.merge(mdf2, left_on='lkey', right_on='rkey')
        result = self.executor.execute_dataframe(jdf, concat=True)[0]

        pd.testing.assert_frame_equal(expected.sort_values(by=expected.columns[1]).reset_index(drop=True),
                                      result.sort_values(by=result.columns[1]).reset_index(drop=True))
