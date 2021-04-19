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
import sys

import numpy as np
import pandas as pd
import pytest

import mars.dataframe as md
import mars.tensor as mt
from mars.core.session import get_default_session, \
    new_session, execute, fetch
from mars.deploy.oscar.local import new_cluster
from mars.tests.core import CONFIG_TEST_FILE


@pytest.fixture()
async def create_cluster():
    start_method = os.environ.get('POOL_START_METHOD', 'forkserver') \
        if sys.platform != 'win32' else None
    client = await new_cluster(subprocess_start_method=start_method,
                               config=CONFIG_TEST_FILE,
                               n_worker=1,
                               n_cpu=2)
    async with client:
        yield


@pytest.mark.asyncio
async def test_execute(create_cluster):
    session = get_default_session()
    assert session.address is not None

    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1

    info = await session.execute(b)
    await info
    assert info.result() is None
    assert info.exception() is None
    assert info.progress() == 1
    np.testing.assert_equal(raw + 1, (await session.fetch(b))[0])

    with pytest.raises(ValueError):
        await session.fetch(b + 1)

    with pytest.raises(ValueError):
        await session.fetch(b[b < 0.6])


@pytest.mark.asyncio
async def test_iterative_tiling(create_cluster):
    session = get_default_session()

    raw = np.random.RandomState(0).rand(30, 5)
    raw_df = pd.DataFrame(raw, index=np.arange(1, 31))

    df = md.DataFrame(raw_df, chunk_size=10)
    df = df[df[0] < .7]
    df2 = df.shift(2)

    info = await session.execute(df2)
    await info
    assert info.result() is None
    result = (await session.fetch(df2))[0]

    expected = raw_df[raw_df[0] < .7].shift(2)
    pd.testing.assert_frame_equal(result, expected)

    # test meta
    assert df2.index_value.min_val >= 1
    assert df2.index_value.max_val <= 30


def test_sync_execute():
    start_method = os.environ.get('POOL_START_METHOD', 'forkserver') \
        if sys.platform != 'win32' else None
    session = new_session(n_cpu=2, subprocess_start_method=start_method,
                          default=True)

    with session:
        raw = np.random.RandomState(0).rand(10, 5)
        a = mt.tensor(raw, chunk_size=5).sum(axis=1)
        b = a.execute(show_progress=False)
        assert b is a
        result = a.fetch()
        np.testing.assert_array_equal(result, raw.sum(axis=1))

        c = mt.tensor(raw, chunk_size=5).sum()
        d = session.execute(c)
        assert d is c
        assert abs(session.fetch(d) - raw.sum()) < 0.001

    assert get_default_session() is None


def test_no_default_session():
    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1

    with pytest.warns(Warning):
        execute(b, show_progress=False)

    np.testing.assert_array_equal(fetch(b), raw + 1)
