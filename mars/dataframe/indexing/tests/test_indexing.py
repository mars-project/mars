# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

from mars.executor import Executor
from mars.tests.core import TestBase
from mars.dataframe.core import DataFrame, Series
from mars.dataframe.datasource.dataframe import from_pandas


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.executor = Executor()

    def testILocGetItem(self):
        df1 = pd.DataFrame([[1,3,3], [4,2,6], [7, 8, 9]],
                           index=['a1', 'a2', 'a3'], columns=['x', 'y', 'z'])
        df2 = from_pandas(df1, chunk_size=2)

        # plain index
        df3 = df2.iloc[1]
        df3.tiles()
        self.assertIsInstance(df3, Series)
        self.assertEqual(df3.chunk_shape, (2,))
        self.assertEqual(df3.chunks[0].shape, (2,))
        self.assertEqual(df3.chunks[1].shape, (1,))
        self.assertEqual(df3.chunks[0].op.indexes, (1, slice(None, None, None)))
        self.assertEqual(df3.chunks[1].op.indexes, (1, slice(None, None, None)))
        self.assertEqual(df3.chunks[0].inputs[0].index, (0, 0))
        self.assertEqual(df3.chunks[0].inputs[0].shape, (2, 2))
        self.assertEqual(df3.chunks[1].inputs[0].index, (0, 1))
        self.assertEqual(df3.chunks[1].inputs[0].shape, (2, 1))

        # slice index
        df4 = df2.iloc[:, 2:4]
        df4.tiles()
        self.assertIsInstance(df4, DataFrame)
        self.assertEqual(df4.chunk_shape, (2, 1))
        self.assertEqual(df4.chunks[0].shape, (2, 1))
        self.assertEqual(df4.chunks[1].shape, (1, 1))
        self.assertEqual(df4.chunks[0].op.indexes, (slice(None, None, None), slice(None, None, None)))
        self.assertEqual(df4.chunks[1].op.indexes, (slice(None, None, None), slice(None, None, None)))
        self.assertEqual(df4.chunks[0].inputs[0].index, (0, 1))
        self.assertEqual(df4.chunks[0].inputs[0].shape, (2, 1))
        self.assertEqual(df4.chunks[1].inputs[0].index, (1, 1))
        self.assertEqual(df4.chunks[1].inputs[0].shape, (1, 1))

        # plain fancy index
        df5 = df2.iloc[[0], [0, 1, 2]]
        df5.tiles()
        self.assertIsInstance(df5, DataFrame)
        self.assertEqual(df5.shape, (1, 3))
        self.assertEqual(df5.chunk_shape, (1, 2))
        self.assertEqual(df5.chunks[0].shape, (1, 2))
        self.assertEqual(df5.chunks[1].shape, (1, 1))
        np.testing.assert_array_equal(df5.chunks[0].op.indexes[0], [0])
        np.testing.assert_array_equal(df5.chunks[0].op.indexes[1], [0, 1])
        np.testing.assert_array_equal(df5.chunks[1].op.indexes[0], [0])
        np.testing.assert_array_equal(df5.chunks[1].op.indexes[1], [0])
        self.assertEqual(df5.chunks[0].inputs[0].index, (0, 0))
        self.assertEqual(df5.chunks[0].inputs[0].shape, (2, 2))
        self.assertEqual(df5.chunks[1].inputs[0].index, (0, 1))
        self.assertEqual(df5.chunks[1].inputs[0].shape, (2, 1))

        # fancy index
        df6 = df2.iloc[[1, 2], [0, 1, 2]]
        df6.tiles()
        self.assertIsInstance(df6, DataFrame)
        self.assertEqual(df6.shape, (2, 3))
        self.assertEqual(df6.chunk_shape, (2, 2))
        self.assertEqual(df6.chunks[0].shape, (1, 2))
        self.assertEqual(df6.chunks[1].shape, (1, 1))
        self.assertEqual(df6.chunks[2].shape, (1, 2))
        self.assertEqual(df6.chunks[3].shape, (1, 1))
        np.testing.assert_array_equal(df6.chunks[0].op.indexes[0], [1])
        np.testing.assert_array_equal(df6.chunks[0].op.indexes[1], [0, 1])
        np.testing.assert_array_equal(df6.chunks[1].op.indexes[0], [1])
        np.testing.assert_array_equal(df6.chunks[1].op.indexes[1], [0])
        np.testing.assert_array_equal(df6.chunks[2].op.indexes[0], [0])
        np.testing.assert_array_equal(df6.chunks[2].op.indexes[1], [0, 1])
        np.testing.assert_array_equal(df6.chunks[3].op.indexes[0], [0])
        np.testing.assert_array_equal(df6.chunks[3].op.indexes[1], [0])
        self.assertEqual(df6.chunks[0].inputs[0].index, (0, 0))
        self.assertEqual(df6.chunks[0].inputs[0].shape, (2, 2))
        self.assertEqual(df6.chunks[1].inputs[0].index, (0, 1))
        self.assertEqual(df6.chunks[1].inputs[0].shape, (2, 1))
        self.assertEqual(df6.chunks[2].inputs[0].index, (1, 0))
        self.assertEqual(df6.chunks[2].inputs[0].shape, (1, 2))
        self.assertEqual(df6.chunks[3].inputs[0].index, (1, 1))
        self.assertEqual(df6.chunks[3].inputs[0].shape, (1, 1))

        # plain index
        df7 = df2.iloc[1, 2]
        df7.tiles()
        self.assertIsInstance(df7, Series)
        self.assertEqual(df7.chunk_shape, ())
        self.assertEqual(df7.chunks[0].shape, ())
        self.assertEqual(df7.chunks[0].op.indexes, (1, 0))
        self.assertEqual(df7.chunks[0].inputs[0].index, (0, 1))
        self.assertEqual(df7.chunks[0].inputs[0].shape, (2, 1))
