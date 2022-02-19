# Copyright 1999-2021 Alibaba Group Holding Ltd.
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
import pytest

from .... import tensor as mt
from .... import dataframe as md
from ....core import tile
from .. import shuffle
from ..shuffle import LearnShuffle


def test_shuffle_expr():
    a = mt.random.rand(10, 3, chunk_size=2)
    b = md.DataFrame(mt.random.rand(10, 5), chunk_size=2)

    new_a, new_b = shuffle(a, b, random_state=0)

    assert new_a.op is new_b.op
    assert isinstance(new_a.op, LearnShuffle)
    assert new_a.shape == a.shape
    assert new_b.shape == b.shape
    assert b.index_value.key != new_b.index_value.key

    new_a, new_b = tile(new_a, new_b)

    assert len(new_a.chunks) == 10
    assert np.isnan(new_a.chunks[0].shape[0])
    assert len(new_b.chunks) == 15
    assert np.isnan(new_b.chunks[0].shape[0])
    assert new_b.chunks[0].index_value.key != new_b.chunks[1].index_value.key
    assert new_a.chunks[0].op.seeds == new_b.chunks[0].op.seeds

    c = mt.random.rand(10, 5, 3, chunk_size=2)
    d = md.DataFrame(mt.random.rand(10, 5), chunk_size=(2, 5))

    new_c, new_d = shuffle(c, d, axes=(0, 1), random_state=0)

    assert new_c.op is new_d.op
    assert isinstance(new_c.op, LearnShuffle)
    assert new_c.shape == c.shape
    assert new_d.shape == d.shape
    assert d.index_value.key != new_d.index_value.key
    assert not np.all(new_d.dtypes.index[:-1] < new_d.dtypes.index[1:])
    pd.testing.assert_series_equal(d.dtypes, new_d.dtypes.sort_index())

    new_c, new_d = tile(new_c, new_d)

    assert len(new_c.chunks) == 5 * 1 * 2
    assert np.isnan(new_c.chunks[0].shape[0])
    assert len(new_d.chunks) == 5
    assert np.isnan(new_d.chunks[0].shape[0])
    assert new_d.chunks[0].shape[1] == 5
    assert new_d.chunks[0].index_value.key != new_d.chunks[1].index_value.key
    pd.testing.assert_series_equal(new_d.chunks[0].dtypes.sort_index(), d.dtypes)
    assert new_c.chunks[0].op.seeds == new_d.chunks[0].op.seeds
    assert len(new_c.chunks[0].op.seeds) == 1
    assert new_c.chunks[0].op.reduce_sizes == (5,)

    with pytest.raises(ValueError):
        a = mt.random.rand(10, 5)
        b = mt.random.rand(10, 4, 3)
        shuffle(a, b, axes=1)

    with pytest.raises(TypeError):
        shuffle(a, b, unknown_param=True)

    assert isinstance(shuffle(mt.random.rand(10, 5)), mt.Tensor)


def _sort(data, axes):
    cur = data
    for ax in axes:
        if ax < data.ndim:
            cur = np.sort(cur, axis=ax)
    return cur


def test_shuffle_execution(setup):
    # test consistency
    s1 = np.arange(9).reshape(3, 3)
    s2 = np.arange(1, 10).reshape(3, 3)
    ts1 = mt.array(s1, chunk_size=2)
    ts2 = mt.array(s2, chunk_size=2)

    ret = shuffle(ts1, ts2, axes=[0, 1], random_state=0)
    res1, res2 = ret.execute().fetch()

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
        res1, res2 = ret.execute().fetch()

        assert res1.shape == raw1.shape
        assert res2.shape == raw2.shape
        np.testing.assert_array_equal(_sort(raw1, axes), _sort(res1, axes))
        np.testing.assert_array_equal(_sort(raw2, axes), _sort(res2, axes))

    # tensor + tensor(more dimension)
    raw3 = np.random.rand(10, 15)
    t3 = mt.array(raw3, chunk_size=(8, 15))
    raw4 = np.random.rand(10, 15, 20)
    t4 = mt.array(raw4, chunk_size=(5, 15, 10))

    for axes in [(1,), (0, 1), (1, 2)]:
        ret = shuffle(t3, t4, axes=axes, random_state=0)
        res3, res4 = ret.execute().fetch()

        assert res3.shape == raw3.shape
        assert res4.shape == raw4.shape
        np.testing.assert_array_equal(_sort(raw3, axes), _sort(res3, axes))
        np.testing.assert_array_equal(_sort(raw4, axes), _sort(res4, axes))

    # tensor + dataframe + series
    raw5 = np.random.rand(10, 15, 20)
    t5 = mt.array(raw5, chunk_size=8)
    t6 = mt.array(raw5[:, 0, 0], chunk_size=6)
    raw6 = pd.DataFrame(np.random.rand(10, 15))
    df = md.DataFrame(raw6, chunk_size=(8, 15))
    raw7 = pd.Series(np.random.rand(10))
    series = md.Series(raw7, chunk_size=8)

    for axes in [(0,), (1,), (0, 1), (1, 2), [0, 1, 2]]:
        ret = shuffle(t5, df, series, t6, axes=axes, random_state=0)
        # skip check nsplits because it's updated
        res5, res_df, res_series, res6 = ret.execute(
            extra_config={"check_nsplits": False}
        ).fetch(extra_config={"check_nsplits": False})

        assert res5.shape == raw5.shape
        assert res_df.shape == df.shape
        assert res_series.shape == series.shape
        assert res6.shape == (raw5.shape[0],)
