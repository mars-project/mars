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

import unittest
import operator

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from mars.dataframe.core import IndexValue
from mars.dataframe.expression_utils import parse_index
from mars.dataframe.arithmetic.utils import infer_dtypes, infer_index_value


@unittest.skipIf(pd is None, 'pandas not installed')
class Test(unittest.TestCase):
    def testInferDtypes(self):
        data1 = pd.DataFrame([[1, 'a', False]], columns=[2.0, 3.0, 4.0])
        data2 = pd.DataFrame([[1, 3.0, 'b']], columns=[1, 2, 3])

        pd.testing.assert_series_equal(infer_dtypes(data1.dtypes, data2.dtypes, operator.add),
                                      (data1 + data2).dtypes)

    def testInferIndexValue(self):
        # same range index
        index1 = pd.RangeIndex(1, 3)
        index2 = pd.RangeIndex(1, 3)

        ival1 = parse_index(index1)
        ival2 = parse_index(index2)
        oival = infer_index_value(ival1, ival2, operator.add)

        self.assertEqual(oival.key, ival1.key)
        self.assertEqual(oival.key, ival2.key)

        # different range index
        index1 = pd.RangeIndex(1, 3)
        index2 = pd.RangeIndex(2, 4)

        ival1 = parse_index(index1)
        ival2 = parse_index(index2)
        oival = infer_index_value(ival1, ival2, operator.add)

        self.assertIsInstance(oival.value, IndexValue.Int64Index)
        self.assertNotEqual(oival.key, ival1.key)
        self.assertNotEqual(oival.key, ival2.key)

        # same int64 index, all unique
        index1 = pd.Int64Index([1, 2])
        index2 = pd.Int64Index([1, 2])

        ival1 = parse_index(index1)
        ival2 = parse_index(index2)
        oival = infer_index_value(ival1, ival2, operator.add)

        self.assertIsInstance(oival.value, IndexValue.Int64Index)
        self.assertEqual(oival.key, ival1.key)
        self.assertEqual(oival.key, ival2.key)

        # same int64 index, not all unique
        index1 = pd.Int64Index([1, 2, 2])
        index2 = pd.Int64Index([1, 2, 2])

        ival1 = parse_index(index1)
        ival2 = parse_index(index2)
        oival = infer_index_value(ival1, ival2, operator.add)

        self.assertIsInstance(oival.value, IndexValue.Int64Index)
        self.assertNotEqual(oival.key, ival1.key)
        self.assertNotEqual(oival.key, ival2.key)

        # different int64 index
        index1 = pd.Int64Index([1, 2])
        index2 = pd.Int64Index([2, 3])

        ival1 = parse_index(index1)
        ival2 = parse_index(index2)
        oival = infer_index_value(ival1, ival2, operator.add)

        self.assertIsInstance(oival.value, IndexValue.Int64Index)
        self.assertNotEqual(oival.key, ival1.key)
        self.assertNotEqual(oival.key, ival2.key)

        # different index type
        index1 = pd.Int64Index([1, 2])
        index2 = pd.Float64Index([2.0, 3.0])

        ival1 = parse_index(index1)
        ival2 = parse_index(index2)
        oival = infer_index_value(ival1, ival2, operator.add)

        self.assertIsInstance(oival.value, IndexValue.Float64Index)
        self.assertNotEqual(oival.key, ival1.key)
        self.assertNotEqual(oival.key, ival2.key)
