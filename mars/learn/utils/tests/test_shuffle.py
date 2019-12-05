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

import unittest

import numpy as np
import pandas as pd

import mars.tensor as mt
import mars.dataframe as md
from mars.learn.utils import shuffle
from mars.learn.utils.shuffle import LearnShuffle
from mars.tiles import get_tiled


class Test(unittest.TestCase):
    def testShuffleExpr(self):
        a = mt.random.rand(10, 3, chunk_size=2)
        b = md.DataFrame(mt.random.rand(10, 5), chunk_size=2)

        new_a, new_b = shuffle(a, b, random_state=0)

        self.assertIs(new_a.op, new_b.op)
        self.assertIsInstance(new_a.op, LearnShuffle)
        self.assertEqual(new_a.shape, a.shape)
        self.assertEqual(new_b.shape, b.shape)
        self.assertNotEqual(b.index_value.key, new_b.index_value.key)

        new_a = new_a.tiles()
        new_b = get_tiled(new_b)

        self.assertEqual(len(new_a.chunks), 10)
        self.assertTrue(np.isnan(new_a.chunks[0].shape[0]))
        self.assertEqual(len(new_b.chunks), 15)
        self.assertTrue(np.isnan(new_b.chunks[0].shape[0]))
        self.assertNotEqual(new_b.chunks[0].index_value.key, new_b.chunks[1].index_value.key)
        self.assertEqual(new_a.chunks[0].op.seeds, new_b.chunks[0].op.seeds)

        c = mt.random.rand(10, 5, 3, chunk_size=2)
        d = md.DataFrame(mt.random.rand(10, 5), chunk_size=(2, 5))

        new_c, new_d = shuffle(c, d, axes=(0, 1), random_state=0)

        self.assertIs(new_c.op, new_d.op)
        self.assertIsInstance(new_c.op, LearnShuffle)
        self.assertEqual(new_c.shape, c.shape)
        self.assertEqual(new_d.shape, d.shape)
        self.assertNotEqual(d.index_value.key, new_d.index_value.key)
        self.assertFalse(np.all(new_d.dtypes.index[:-1] < new_d.dtypes.index[1:]))
        pd.testing.assert_series_equal(d.dtypes, new_d.dtypes.sort_index())

        new_c = new_c.tiles()
        new_d = get_tiled(new_d)

        self.assertEqual(len(new_c.chunks), 5 * 1 * 2)
        self.assertTrue(np.isnan(new_c.chunks[0].shape[0]))
        self.assertEqual(len(new_d.chunks), 5)
        self.assertTrue(np.isnan(new_d.chunks[0].shape[0]))
        self.assertEqual(new_d.chunks[0].shape[1], 5)
        self.assertNotEqual(new_d.chunks[0].index_value.key, new_d.chunks[1].index_value.key)
        pd.testing.assert_series_equal(new_d.chunks[0].dtypes.sort_index(), d.dtypes)
        self.assertEqual(new_c.chunks[0].op.seeds, new_d.chunks[0].op.seeds)
        self.assertEqual(len(new_c.chunks[0].op.seeds), 1)
        self.assertEqual(new_c.chunks[0].op.reduce_sizes, (5,))

        with self.assertRaises(ValueError):
            a = mt.random.rand(10, 5)
            b = mt.random.rand(10, 4, 3)
            shuffle(a, b, axes=1)

        with self.assertRaises(TypeError):
            shuffle(a, b, unknown_param=True)

        self.assertIsInstance(shuffle(mt.random.rand(10, 5)), mt.Tensor)

    @staticmethod
    def _sort(data, axes):
        cur = data
        for ax in axes:
            if ax < data.ndim:
                cur = np.sort(cur, axis=ax)
        return cur

    def testShuffleExecution(self):
        # test consistency
        s1 = np.arange(9).reshape(3, 3)
        s2 = np.arange(1, 10).reshape(3, 3)
        ts1 = mt.array(s1, chunk_size=2)
        ts2 = mt.array(s2, chunk_size=2)

        ret = shuffle(ts1, ts2, axes=[0, 1], random_state=0)
        res1, res2 = ret.execute()

        # calc row index
        s1_col_0 = s1[:, 0].tolist()
        rs1_col_0 = [res1[:, i] for i in range(3) if set(s1_col_0) == set(res1[:, i])][0]
        row_index = [s1_col_0.index(j) for j in rs1_col_0]
        # calc col index
        s1_row_0 = s1[0].tolist()
        rs1_row_0 = [res1[i] for i in range(3) if set(s1_row_0) == set(res1[i])][0]
        col_index = [s1_row_0.index(j) for j in rs1_row_0]
        np.testing.assert_array_equal(res2, s2[row_index][:, col_index])

        # tensor + tensor
        raw1 = np.random.rand(10, 15, 20)
        t1 = mt.array(raw1, chunk_size=8)
        raw2 = np.random.rand(10, 15, 20)
        t2 = mt.array(raw2, chunk_size=5)

        for axes in [(0,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]:
            ret = shuffle(t1, t2, axes=axes, random_state=0)
            res1, res2 = ret.execute()

            self.assertEqual(res1.shape, raw1.shape)
            self.assertEqual(res2.shape, raw2.shape)
            np.testing.assert_array_equal(Test._sort(raw1, axes), Test._sort(res1, axes))
            np.testing.assert_array_equal(Test._sort(raw2, axes), Test._sort(res2, axes))

        # tensor + tensor(more dimension)
        raw3 = np.random.rand(10, 15)
        t3 = mt.array(raw3, chunk_size=(8, 15))
        raw4 = np.random.rand(10, 15, 20)
        t4 = mt.array(raw4, chunk_size=(5, 15, 10))

        for axes in [(1,), (0, 1), (1, 2)]:
            ret = shuffle(t3, t4, axes=axes, random_state=0)
            res3, res4 = ret.execute()

            self.assertEqual(res3.shape, raw3.shape)
            self.assertEqual(res4.shape, raw4.shape)
            np.testing.assert_array_equal(Test._sort(raw3, axes), Test._sort(res3, axes))
            np.testing.assert_array_equal(Test._sort(raw4, axes), Test._sort(res4, axes))

        # tensor + dataframe + series
        raw5 = np.random.rand(10, 15, 20)
        t5 = mt.array(raw5, chunk_size=8)
        raw6 = pd.DataFrame(np.random.rand(10, 15))
        df = md.DataFrame(raw6, chunk_size=(8, 15))
        raw7 = pd.Series(np.random.rand(10))
        series = md.Series(raw7, chunk_size=8)

        for axes in [(0,), (1,), (0, 1), (1, 2), [0, 1, 2]]:
            ret = shuffle(t5, df, series, axes=axes, random_state=0)
            res5, res_df, res_series = ret.execute()

            self.assertEqual(res5.shape, raw5.shape)
            self.assertEqual(res_df.shape, df.shape)
            self.assertEqual(res_series.shape, series.shape)
