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

import operator
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral

import numpy as np
import pandas as pd

from mars.config import option_context
from mars.dataframe.initializer import DataFrame
from mars.dataframe.core import IndexValue
from mars.dataframe.utils import decide_dataframe_chunk_sizes, decide_series_chunk_size, \
    split_monotonic_index_min_max, build_split_idx_to_origin_idx, parse_index, filter_index_value, \
    infer_dtypes, infer_index_value, validate_axis, fetch_corner_data, suspend_stdio
from mars.session import new_session


class Test(unittest.TestCase):
    def testDecideDataFrameChunks(self):
        with option_context() as options:
            options.chunk_store_limit = 64

            memory_usage = pd.Series([8, 22.2, 4, 2, 11.2], index=list('abcde'))

            shape = (10, 5)
            nsplit = decide_dataframe_chunk_sizes(shape, None, memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_dataframe_chunk_sizes(shape, {0: 4}, memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_dataframe_chunk_sizes(shape, (2, 3), memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_dataframe_chunk_sizes(shape, (10, 3), memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            options.chunk_store_limit = 20

            shape = (10, 5)
            nsplit = decide_dataframe_chunk_sizes(shape, None, memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_dataframe_chunk_sizes(shape, {1: 3}, memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_dataframe_chunk_sizes(shape, (2, 3), memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_dataframe_chunk_sizes(shape, (10, 3), memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

    def testDecideSeriesChunks(self):
        with option_context() as options:
            options.chunk_store_limit = 64

            s = pd.Series(np.empty(50, dtype=np.int64))
            nsplit = decide_series_chunk_size(s.shape, None, s.memory_usage(index=False, deep=True))
            self.assertEqual(len(nsplit), 1)
            self.assertEqual(sum(nsplit[0]), 50)
            self.assertEqual(nsplit[0][0], 8)

    def testParseIndex(self):
        index = pd.Int64Index([])
        parsed_index = parse_index(index)
        self.assertIsInstance(parsed_index.value, IndexValue.Int64Index)
        pd.testing.assert_index_equal(index, parsed_index.to_pandas())

        index = pd.Int64Index([1, 2])
        parsed_index = parse_index(index)  # not parse data
        self.assertIsInstance(parsed_index.value, IndexValue.Int64Index)
        with self.assertRaises(AssertionError):
            pd.testing.assert_index_equal(index, parsed_index.to_pandas())

        parsed_index = parse_index(index, store_data=True)  # parse data
        self.assertIsInstance(parsed_index.value, IndexValue.Int64Index)
        pd.testing.assert_index_equal(index, parsed_index.to_pandas())

        index = pd.RangeIndex(0, 10, 3)
        parsed_index = parse_index(index)
        self.assertIsInstance(parsed_index.value, IndexValue.RangeIndex)
        pd.testing.assert_index_equal(index, parsed_index.to_pandas())

        index = pd.MultiIndex.from_arrays([[0, 1], ['a', 'b'], ['X', 'Y']])
        parsed_index = parse_index(index)  # not parse data
        self.assertIsInstance(parsed_index.value, IndexValue.MultiIndex)
        with self.assertRaises(AssertionError):
            pd.testing.assert_index_equal(index, parsed_index.to_pandas())

        parsed_index = parse_index(index, store_data=True)  # parse data
        self.assertIsInstance(parsed_index.value, IndexValue.MultiIndex)
        pd.testing.assert_index_equal(index, parsed_index.to_pandas())

    def testSplitMonotonicIndexMinMax(self):
        left_min_max = [[0, True, 3, True], [3, False, 5, False]]
        right_min_max = [[1, False, 3, True], [4, False, 6, True]]
        left_splits, right_splits = \
            split_monotonic_index_min_max(left_min_max, True, right_min_max, True)
        self.assertEqual(left_splits,
                         [[(0, True, 1, True), (1, False, 3, True)],
                          [(3, False, 4, True), (4, False, 5, False), (5, True, 6, True)]])
        self.assertEqual(right_splits,
                         [[(0, True, 1, True), (1, False, 3, True)],
                          [(3, False, 4, True), (4, False, 5, False), (5, True, 6, True)]])
        left_splits, right_splits = split_monotonic_index_min_max(right_min_max, False, left_min_max, False)
        self.assertEqual(list(reversed(left_splits)),
                         [[(0, True, 1, True), (1, False, 3, True)],
                          [(3, False, 4, True), (4, False, 5, False), (5, True, 6, True)]])
        self.assertEqual(list(reversed(right_splits)),
                         [[(0, True, 1, True), (1, False, 3, True)],
                          [(3, False, 4, True), (4, False, 5, False), (5, True, 6, True)]])

        left_min_max = [[2, True, 4, True], [8, True, 9, False]]
        right_min_max = [[1, False, 3, True], [4, False, 6, True]]
        left_splits, right_splits = \
            split_monotonic_index_min_max(left_min_max, True, right_min_max, True)
        self.assertEqual(left_splits,
                         [[(1, False, 2, False), (2, True, 3, True), (3, False, 4, True)],
                          [(4, False, 6, True), (8, True, 9, False)]])
        self.assertEqual(right_splits,
                         [[(1, False, 2, False), (2, True, 3, True)],
                          [(3, False, 4, True), (4, False, 6, True), (8, True, 9, False)]])

        left_min_max = [[1, False, 3, True], [4, False, 6, True], [10, True, 12, False], [13, True, 14, False]]
        right_min_max = [[2, True, 4, True], [5, True, 7, False]]
        left_splits, right_splits = \
            split_monotonic_index_min_max(left_min_max, True, right_min_max, True)
        self.assertEqual(left_splits,
                         [[(1, False, 2, False), (2, True, 3, True)],
                          [(3, False, 4, True), (4, False, 5, False), (5, True, 6, True)],
                          [(6, False, 7, False), (10, True, 12, False)],
                          [(13, True, 14, False)]])
        self.assertEqual(right_splits,
                         [[(1, False, 2, False), (2, True, 3, True), (3, False, 4, True)],
                          [(4, False, 5, False), (5, True, 6, True), (6, False, 7, False),
                           (10, True, 12, False), (13, True, 14, False)]])
        left_splits, right_splits = \
            split_monotonic_index_min_max(right_min_max, True, left_min_max, True)
        self.assertEqual(left_splits,
                         [[(1, False, 2, False), (2, True, 3, True), (3, False, 4, True)],
                          [(4, False, 5, False), (5, True, 6, True), (6, False, 7, False),
                           (10, True, 12, False), (13, True, 14, False)]])
        self.assertEqual(right_splits,
                         [[(1, False, 2, False), (2, True, 3, True)],
                          [(3, False, 4, True), (4, False, 5, False), (5, True, 6, True)],
                          [(6, False, 7, False), (10, True, 12, False)],
                          [(13, True, 14, False)]])

        # left min_max like ([.., .., 4 True], [4, False, ..., ...]
        # right min_max like ([..., ..., 4 False], [4, True, ..., ...]
        left_min_max = [[1, False, 4, True], [4, False, 6, True]]
        right_min_max = [[1, False, 4, False], [4, True, 6, True]]
        left_splits, right_splits = split_monotonic_index_min_max(
            left_min_max, True, right_min_max, True)
        self.assertEqual(left_splits,
                         [[(1, False, 4, False), (4, True, 4, True)], [(4, False, 6, True)]])
        self.assertEqual(right_splits,
                         [[(1, False, 4, False)], [(4, True, 4, True), (4, False, 6, True)]])

        # identical index
        left_min_max = [[1, False, 3, True], [4, False, 6, True]]
        right_min_max = [[1, False, 3, True], [4, False, 6, True]]
        left_splits, right_splits = \
            split_monotonic_index_min_max(left_min_max, True, right_min_max, True)
        self.assertEqual(left_splits, [[tuple(it)] for it in left_min_max])
        self.assertEqual(right_splits, [[tuple(it)] for it in left_min_max])

    def testBuildSplitIdxToOriginIdx(self):
        splits = [[(1, False, 2, False), (2, True, 3, True)], [(5, False, 6, True)]]
        res = build_split_idx_to_origin_idx(splits)

        self.assertEqual(res, {0: (0, 0), 1: (0, 1), 2: (1, 0)})

        splits = [[(5, False, 6, True)], [(1, False, 2, False), (2, True, 3, True)]]
        res = build_split_idx_to_origin_idx(splits, increase=False)

        self.assertEqual(res, {0: (1, 0), 1: (1, 1), 2: (0, 0)})

    def testFilterIndexValue(self):
        pd_index = pd.RangeIndex(10)
        index_value = parse_index(pd_index)

        min_max = (0, True, 9, True)
        self.assertEqual(filter_index_value(index_value, min_max).to_pandas().tolist(),
                         pd_index[(pd_index >= 0) & (pd_index <= 9)].tolist())

        min_max = (0, False, 9, False)
        self.assertEqual(filter_index_value(index_value, min_max).to_pandas().tolist(),
                         pd_index[(pd_index > 0) & (pd_index < 9)].tolist())

        pd_index = pd.RangeIndex(1, 11, 3)
        index_value = parse_index(pd_index)

        min_max = (2, True, 10, True)
        self.assertEqual(filter_index_value(index_value, min_max).to_pandas().tolist(),
                         pd_index[(pd_index >= 2) & (pd_index <= 10)].tolist())

        min_max = (2, False, 10, False)
        self.assertEqual(filter_index_value(index_value, min_max).to_pandas().tolist(),
                         pd_index[(pd_index > 2) & (pd_index < 10)].tolist())

        pd_index = pd.RangeIndex(9, -1, -1)
        index_value = parse_index(pd_index)

        min_max = (0, True, 9, True)
        self.assertEqual(filter_index_value(index_value, min_max).to_pandas().tolist(),
                         pd_index[(pd_index >= 0) & (pd_index <= 9)].tolist())

        min_max = (0, False, 9, False)
        self.assertEqual(filter_index_value(index_value, min_max).to_pandas().tolist(),
                         pd_index[(pd_index > 0) & (pd_index < 9)].tolist())

        pd_index = pd.RangeIndex(10, 0, -3)
        index_value = parse_index(pd_index, store_data=False)

        min_max = (2, True, 10, True)
        self.assertEqual(filter_index_value(index_value, min_max).to_pandas().tolist(),
                         pd_index[(pd_index >= 2) & (pd_index <= 10)].tolist())

        min_max = (2, False, 10, False)
        self.assertEqual(filter_index_value(index_value, min_max).to_pandas().tolist(),
                         pd_index[(pd_index > 2) & (pd_index < 10)].tolist())

        pd_index = pd.Int64Index([0, 3, 8])
        index_value = parse_index(pd_index, store_data=True)

        min_max = (2, True, 8, False)
        self.assertEqual(filter_index_value(index_value, min_max, store_data=True).to_pandas().tolist(),
                         pd_index[(pd_index >= 2) & (pd_index < 8)].tolist())

        index_value = parse_index(pd_index)

        min_max = (2, True, 8, False)
        filtered = filter_index_value(index_value, min_max)
        self.assertEqual(len(filtered.to_pandas().tolist()), 0)
        self.assertIsInstance(filtered.value, IndexValue.Int64Index)

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
        oival = infer_index_value(ival1, ival2)

        self.assertEqual(oival.key, ival1.key)
        self.assertEqual(oival.key, ival2.key)

        # different range index
        index1 = pd.RangeIndex(1, 3)
        index2 = pd.RangeIndex(2, 4)

        ival1 = parse_index(index1)
        ival2 = parse_index(index2)
        oival = infer_index_value(ival1, ival2)

        self.assertIsInstance(oival.value, IndexValue.Int64Index)
        self.assertNotEqual(oival.key, ival1.key)
        self.assertNotEqual(oival.key, ival2.key)

        # same int64 index, all unique
        index1 = pd.Int64Index([1, 2])
        index2 = pd.Int64Index([1, 2])

        ival1 = parse_index(index1)
        ival2 = parse_index(index2)
        oival = infer_index_value(ival1, ival2)

        self.assertIsInstance(oival.value, IndexValue.Int64Index)
        self.assertEqual(oival.key, ival1.key)
        self.assertEqual(oival.key, ival2.key)

        # same int64 index, not all unique
        index1 = pd.Int64Index([1, 2, 2])
        index2 = pd.Int64Index([1, 2, 2])

        ival1 = parse_index(index1)
        ival2 = parse_index(index2)
        oival = infer_index_value(ival1, ival2)

        self.assertIsInstance(oival.value, IndexValue.Int64Index)
        self.assertNotEqual(oival.key, ival1.key)
        self.assertNotEqual(oival.key, ival2.key)

        # different int64 index
        index1 = pd.Int64Index([1, 2])
        index2 = pd.Int64Index([2, 3])

        ival1 = parse_index(index1)
        ival2 = parse_index(index2)
        oival = infer_index_value(ival1, ival2)

        self.assertIsInstance(oival.value, IndexValue.Int64Index)
        self.assertNotEqual(oival.key, ival1.key)
        self.assertNotEqual(oival.key, ival2.key)

        # different index type
        index1 = pd.Int64Index([1, 2])
        index2 = pd.Float64Index([2.0, 3.0])

        ival1 = parse_index(index1)
        ival2 = parse_index(index2)
        oival = infer_index_value(ival1, ival2)

        self.assertIsInstance(oival.value, IndexValue.Float64Index)
        self.assertNotEqual(oival.key, ival1.key)
        self.assertNotEqual(oival.key, ival2.key)

        # range index and other index
        index1 = pd.RangeIndex(1, 4)
        index2 = pd.Float64Index([2, 3, 4])

        ival1 = parse_index(index1)
        ival2 = parse_index(index2)
        oival = infer_index_value(ival1, ival2)

        self.assertIsInstance(oival.value, IndexValue.Float64Index)
        self.assertNotEqual(oival.key, ival1.key)
        self.assertNotEqual(oival.key, ival2.key)

        index1 = pd.DatetimeIndex([])
        index2 = pd.RangeIndex(2)

        ival1 = parse_index(index1)
        ival2 = parse_index(index2)
        oival = infer_index_value(ival1, ival2)

        self.assertIsInstance(oival.value, IndexValue.Index)
        self.assertNotEqual(oival.key, ival1.key)
        self.assertNotEqual(oival.key, ival2.key)

    def testValidateAxis(self):
        df = DataFrame(pd.DataFrame(np.random.rand(4, 3)))

        self.assertEqual(validate_axis(0, df), 0)
        self.assertEqual(validate_axis('index', df), 0)
        self.assertEqual(validate_axis(1, df), 1)
        self.assertEqual(validate_axis('columns', df), 1)

        with self.assertRaises(ValueError):
            validate_axis('unknown index', df)

        with self.assertRaises(ValueError):
            validate_axis(object(), df)

        with self.assertRaises(ValueError):
            validate_axis(-1, df)

        with self.assertRaises(ValueError):
            validate_axis(2, df)

        df2 = df[df[0] < 0.5]  # create unknown shape
        self.assertEqual(validate_axis(0, df2), 0)

    def testDataFrameDir(self):
        df = DataFrame(pd.DataFrame(np.random.rand(4, 3), columns=list('ABC')))
        dir_result = set(dir(df))
        for c in df.dtypes.index:
            self.assertIn(c, dir_result)

    def testFetchDataFrameCornerData(self):
        max_rows = pd.get_option('display.max_rows')
        try:
            min_rows = pd.get_option('display.min_rows')
        except KeyError:  # pragma: no cover
            min_rows = max_rows
        sess = new_session()

        for row in (5,
                    max_rows - 2,
                    max_rows - 1,
                    max_rows,
                    max_rows + 1,
                    max_rows + 2,
                    max_rows + 3):
            pdf = pd.DataFrame(np.random.rand(row, 5))
            df = DataFrame(pdf, chunk_size=max_rows // 2)
            sess.run(df, fetch=False)

            corner = fetch_corner_data(df, session=sess)
            self.assertLessEqual(corner.shape[0], max_rows + 2)
            corner_max_rows = max_rows if row <= max_rows else corner.shape[0] - 1
            self.assertEqual(corner.to_string(max_rows=corner_max_rows, min_rows=min_rows),
                             pdf.to_string(max_rows=max_rows, min_rows=min_rows),
                             f'failed when row == {row}')

    def testSuspendStdio(self):
        old_stdout, old_stderr = sys.stdout, sys.stderr

        class _IOWrapper:
            def __init__(self, name=None):
                self.name = name
                self.content = ''

            @staticmethod
            def writable():
                return True

            def write(self, d):
                self.content += d
                return len(d)

        stdout_w = _IOWrapper('stdout')
        stderr_w = _IOWrapper('stderr')
        executor = ThreadPoolExecutor(1)
        try:
            sys.stdout = stdout_w
            sys.stderr = stderr_w

            with suspend_stdio(), suspend_stdio():
                self.assertTrue(sys.stdout.writable())
                self.assertTrue(sys.stderr.writable())

                print('LINE 1', end='\n')
                print('LINE 2', file=sys.stderr, end='\n')
                executor.submit(print, 'LINE T').result()

            print('LINE 1', end='\n')
            print('LINE 2', file=sys.stderr, end='\n')
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        self.assertEqual(stdout_w.content, 'LINE T\nLINE 1\n')
        self.assertEqual(stderr_w.content, 'LINE 2\n')
