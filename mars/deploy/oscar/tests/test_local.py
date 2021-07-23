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

import asyncio
import os
import threading
import time
import uuid

import numpy as np
import pandas as pd
import pytest
try:
    import vineyard
except ImportError:
    vineyard = None

import mars.dataframe as md
import mars.tensor as mt
import mars.remote as mr
from mars.config import option_context
from mars.deploy.oscar.session import get_default_async_session, \
    get_default_session, new_session, execute, fetch, stop_server, \
    AsyncSession, _IsolatedWebSession
from mars.deploy.oscar.local import new_cluster
from mars.deploy.oscar.service import load_config
from mars.lib.aio import new_isolation
from mars.storage import StorageLevel
from mars.services.storage import StorageAPI
from mars.tensor.arithmetic.add import TensorAdd
from .modules.utils import ( # noqa: F401; pylint: disable=unused-variable
    cleanup_third_party_modules_output,
    get_output_filenames,
)


CONFIG_TEST_FILE = os.path.join(
    os.path.dirname(__file__), 'local_test_config.yml')

CONFIG_VINEYARD_TEST_FILE = os.path.join(
    os.path.dirname(__file__), 'local_test_with_vineyard_config.yml')


CONFIG_THIRD_PARTY_MODULES_TEST_FILE = os.path.join(
    os.path.dirname(__file__), 'local_test_with_third_parity_modules_config.yml')


params = ['default']
if vineyard is not None:
    params.append('vineyard')


@pytest.mark.parametrize(indirect=True)
@pytest.fixture(params=params)
async def create_cluster(request):
    if request.param == 'default':
        config = CONFIG_TEST_FILE
    elif request.param == 'vineyard':
        config = CONFIG_VINEYARD_TEST_FILE
    start_method = os.environ.get('POOL_START_METHOD', None)
    client = await new_cluster(subprocess_start_method=start_method,
                               config=config,
                               n_worker=2,
                               n_cpu=2,
                               use_uvloop=False)
    async with client:
        if request.param == 'default':
            assert client.session.client is not None
        yield client


def _assert_storage_cleaned(session_id: str,
                            addr: str,
                            level: StorageLevel):

    async def _assert(session_id: str,
                                addr: str,
                                level: StorageLevel):
        storage_api = await StorageAPI.create(session_id, addr)
        assert len(await storage_api.list(level)) == 0
        info = await storage_api.get_storage_level_info(level)
        assert info.used_size == 0

    isolation = new_isolation()
    asyncio.run_coroutine_threadsafe(
        _assert(session_id, addr, level), isolation.loop).result()


@pytest.mark.asyncio
async def test_execute(create_cluster):
    session = get_default_async_session()
    assert session.address is not None
    assert session.session_id is not None

    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1

    info = await session.execute(b)
    await info
    assert info.result() is None
    assert info.exception() is None
    assert info.progress() == 1
    np.testing.assert_equal(raw + 1, await session.fetch(b))

    with pytest.raises(ValueError):
        await session.fetch(b + 1)

    with pytest.raises(ValueError):
        await session.fetch(b[b < 0.6])

    del a, b


@pytest.mark.asyncio
async def test_iterative_tiling(create_cluster):
    session = get_default_async_session()

    raw = np.random.RandomState(0).rand(30, 5)
    raw_df = pd.DataFrame(raw, index=np.arange(1, 31))

    df = md.DataFrame(raw_df, chunk_size=10)
    df = df[df[0] < .7]
    df2 = df.shift(2)

    info = await session.execute(df2)
    await info
    assert info.result() is None
    result = await session.fetch(df2)

    expected = raw_df[raw_df[0] < .7].shift(2)
    pd.testing.assert_frame_equal(result, expected)

    # test meta
    assert df2.index_value.min_val >= 1
    assert df2.index_value.max_val <= 30


@pytest.mark.asyncio
async def test_execute_describe(create_cluster):
    s = np.random.RandomState(0)
    raw = pd.DataFrame(s.rand(100, 4), columns=list('abcd'))
    df = md.DataFrame(raw, chunk_size=30)

    session = get_default_async_session()
    r = df.describe()
    info = await session.execute(r)
    await info
    assert info.result() is None
    assert info.exception() is None
    assert info.progress() == 1
    res = await session.fetch(r)
    pd.testing.assert_frame_equal(res, raw.describe())


@pytest.mark.asyncio
async def test_sync_execute_in_async(create_cluster):
    a = mt.ones((10, 10))
    b = a + 1
    res = b.to_numpy()
    np.testing.assert_array_equal(res, np.ones((10, 10)) + 1)


async def _run_web_session_test(web_address):
    session_id = str(uuid.uuid4())
    session = await AsyncSession.init(web_address, session_id)
    session.as_default()
    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1

    info = await session.execute(b)
    await info
    assert info.result() is None
    assert info.exception() is None
    assert info.progress() == 1
    np.testing.assert_equal(raw + 1, await session.fetch(b))
    del a, b

    AsyncSession.reset_default()
    await session.destroy()


@pytest.mark.asyncio
async def test_web_session(create_cluster):
    session_id = str(uuid.uuid4())
    web_address = create_cluster.web_address
    session = await AsyncSession.init(web_address, session_id)
    assert await session.get_web_endpoint() == web_address
    session.as_default()
    assert isinstance(session._isolated_session, _IsolatedWebSession)
    await test_execute(create_cluster)
    await test_iterative_tiling(create_cluster)
    AsyncSession.reset_default()
    await session.destroy()
    await _run_web_session_test(web_address)


def test_sync_execute():
    session = new_session(n_cpu=2, default=True,
                          web=False, use_uvloop=False)

    # web not started
    assert session._session.client.web_address is None
    assert session.get_web_endpoint() is None

    with session:
        raw = np.random.RandomState(0).rand(10, 5)
        a = mt.tensor(raw, chunk_size=5).sum(axis=1)
        b = a.execute(show_progress=False)
        assert b is a
        result = a.fetch()
        np.testing.assert_array_equal(result, raw.sum(axis=1))

        c = b + 1
        c.execute(show_progress=False)
        result = c.fetch()
        np.testing.assert_array_equal(result, raw.sum(axis=1) + 1)

        c = mt.tensor(raw, chunk_size=5).sum()
        d = session.execute(c)
        assert d is c
        assert abs(session.fetch(d) - raw.sum()) < 0.001

    for worker_pool in session._session.client._cluster._worker_pools:
        _assert_storage_cleaned(session.session_id, worker_pool.external_address,
                                StorageLevel.MEMORY)

    session.stop_server()
    assert get_default_async_session() is None


def test_no_default_session():
    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1

    with pytest.warns(Warning):
        execute(b, show_progress=False)

    np.testing.assert_array_equal(fetch(b), raw + 1)
    assert get_default_async_session() is not None
    stop_server()
    assert get_default_async_session() is None


@pytest.fixture
def setup_session():
    session = new_session(n_cpu=2, default=True, use_uvloop=False)
    assert session.get_web_endpoint() is not None

    with session:
        with option_context({'show_progress': False}):
            yield session

    session.stop_server()


def test_decref(setup_session):
    session = setup_session

    a = mt.ones((10, 10))
    b = mt.ones((10, 10))
    c = b + 1
    d = mt.ones((5, 5))

    a.execute()
    b.execute()
    c.execute()
    d.execute()

    del a
    ref_counts = session._get_ref_counts()
    assert len(ref_counts) == 3
    del b
    ref_counts = session._get_ref_counts()
    assert len(ref_counts) == 3
    del c
    ref_counts = session._get_ref_counts()
    assert len(ref_counts) == 1
    del d
    ref_counts = session._get_ref_counts()
    assert len(ref_counts) == 0

    worker_addr = session._session.client._cluster._worker_pools[0].external_address
    _assert_storage_cleaned(session.session_id, worker_addr, StorageLevel.MEMORY)


def _cancel_when_execute(session, cancelled):
    def run():
        time.sleep(200)

    rs = [mr.spawn(run) for _ in range(10)]
    execute(*rs, cancelled=cancelled)

    assert all(not r._executed_sessions for r in rs)

    del rs
    ref_counts = session._get_ref_counts()
    assert len(ref_counts) == 0

    worker_addr = session._session.client._cluster._worker_pools[0].external_address
    _assert_storage_cleaned(session.session_id, worker_addr, StorageLevel.MEMORY)


class SlowTileAdd(TensorAdd):
    @classmethod
    def tile(cls, op):
        time.sleep(2)
        return (yield from TensorAdd.tile(op))


def _cancel_when_tile(session, cancelled):
    a = mt.tensor([1, 2, 3])
    for i in range(20):
        a = SlowTileAdd(dtype=np.dtype(np.int64))(a, 1)
    execute(a, cancelled=cancelled)

    assert not a._executed_sessions

    del a
    ref_counts = session._get_ref_counts()
    assert len(ref_counts) == 0


@pytest.mark.parametrize(
    'test_func', [_cancel_when_execute, _cancel_when_tile])
def test_cancel(setup_session, test_func):
    session = setup_session

    async def _new_cancel_event():
        return asyncio.Event()

    isolation = new_isolation()
    cancelled = asyncio.run_coroutine_threadsafe(
        _new_cancel_event(), isolation.loop).result()

    def cancel():
        time.sleep(.5)
        cancelled.set()

    t = threading.Thread(target=cancel)
    t.daemon = True
    t.start()

    start = time.time()
    test_func(session, cancelled)
    assert time.time() - start < 20

    # submit another task
    raw = np.random.rand(10, 10)
    t = mt.tensor(raw, chunk_size=(10, 5))
    np.testing.assert_array_equal(t.execute().fetch(), raw)


def test_load_third_party_modules(cleanup_third_party_modules_output):  # noqa: F811
    config = load_config()

    config['third_party_modules'] = set()
    with pytest.raises(TypeError, match='set'):
        new_session(n_cpu=2, default=True,
                    web=False, config=config)

    config['third_party_modules'] = {'supervisor': ['not_exists_for_supervisor']}
    with pytest.raises(ModuleNotFoundError, match='not_exists_for_supervisor'):
        new_session(n_cpu=2, default=True,
                    web=False, config=config)

    config['third_party_modules'] = {'worker': ['not_exists_for_worker']}
    with pytest.raises(ModuleNotFoundError, match='not_exists_for_worker'):
        new_session(n_cpu=2, default=True,
                    web=False, config=config)

    config['third_party_modules'] = ['mars.deploy.oscar.tests.modules.replace_op']
    session = new_session(n_cpu=2, default=True,
                          web=False, config=config)
    # web not started
    assert session._session.client.web_address is None

    with session:
        raw = np.random.RandomState(0).rand(10, 10)
        a = mt.tensor(raw, chunk_size=5)
        b = a + 1
        b.execute(show_progress=False)
        result = b.fetch()

        np.testing.assert_equal(raw - 1, result)

    session.stop_server()
    assert get_default_session() is None

    session = new_session(n_cpu=2, default=True,
                          web=False, config=CONFIG_THIRD_PARTY_MODULES_TEST_FILE)
    # web not started
    assert session._session.client.web_address is None

    with session:
        # 1 supervisor, 1 worker main pool, 2 worker sub pools.
        assert len(get_output_filenames()) == 4

    session.stop_server()
    assert get_default_session() is None
