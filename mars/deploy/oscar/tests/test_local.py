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
import copy
import gc
import os
import subprocess
import sys
import threading
import tempfile
import textwrap
import time
import uuid
import weakref

import numpy as np
import pandas as pd
import psutil
import pytest

try:
    import vineyard
except ImportError:
    vineyard = None

from .... import dataframe as md
from .... import tensor as mt
from .... import remote as mr
from ....config import option_context
from ....core.context import get_context
from ....lib.aio import new_isolation
from ....storage import StorageLevel
from ....services.storage import StorageAPI
from ....services.task.supervisor.task import TaskProcessor
from ....tensor.arithmetic.add import TensorAdd
from ....tests.core import mock, check_dict_structure_same, DICT_NOT_EMPTY
from ..local import new_cluster, _load_config
from ..session import (
    get_default_async_session,
    get_default_session,
    new_session,
    execute,
    fetch,
    fetch_infos,
    stop_server,
    AsyncSession,
    ExecutionInfo,
    Profiling,
    Progress,
    _IsolatedWebSession,
    _execute_with_progress,
)
from ..tests.session import new_test_session
from .modules.utils import (  # noqa: F401; pylint: disable=unused-variable
    cleanup_third_party_modules_output,
    get_output_filenames,
)


CONFIG_TEST_FILE = os.path.join(os.path.dirname(__file__), "local_test_config.yml")

CONFIG_VINEYARD_TEST_FILE = os.path.join(
    os.path.dirname(__file__), "local_test_with_vineyard_config.yml"
)


CONFIG_THIRD_PARTY_MODULES_TEST_FILE = os.path.join(
    os.path.dirname(__file__), "local_test_with_third_parity_modules_config.yml"
)

EXPECT_PROFILING_STRUCTURE = {
    "supervisor": {
        "general": {
            "optimize": 0.0005879402160644531,
            "incref_fetch_tileables": 0.0010840892791748047,
            "stage_*": {
                "tile(*)": 0.008243083953857422,
                "gen_subtask_graph(*)": 0.012202978134155273,
                "run": 0.27870702743530273,
                "total": 0.30318617820739746,
            },
            "total": 0.30951380729675293,
        },
        "serialization": {},
        "most_calls": DICT_NOT_EMPTY,
        "slow_calls": DICT_NOT_EMPTY,
        "band_subtasks": DICT_NOT_EMPTY,
        "slow_subtasks": DICT_NOT_EMPTY,
    }
}
EXPECT_PROFILING_STRUCTURE_NO_SLOW = copy.deepcopy(EXPECT_PROFILING_STRUCTURE)
EXPECT_PROFILING_STRUCTURE_NO_SLOW["supervisor"]["slow_calls"] = {}
EXPECT_PROFILING_STRUCTURE_NO_SLOW["supervisor"]["slow_subtasks"] = {}

params = ["default"]
if vineyard is not None:
    params.append("vineyard")


@pytest.mark.parametrize(indirect=True)
@pytest.fixture(params=params)
async def create_cluster(request):
    if request.param == "default":
        config = CONFIG_TEST_FILE
    elif request.param == "vineyard":
        config = CONFIG_VINEYARD_TEST_FILE
    else:
        config = None
    start_method = os.environ.get("POOL_START_METHOD", None)
    client = await new_cluster(
        subprocess_start_method=start_method,
        config=config,
        n_worker=2,
        n_cpu=4,
        use_uvloop=False,
    )
    async with client:
        if request.param == "default":
            assert client.session.client is not None
        yield client, request.param


def _assert_storage_cleaned(session_id: str, addr: str, level: StorageLevel):
    async def _assert(session_id: str, addr: str, level: StorageLevel):
        storage_api = await StorageAPI.create(session_id, addr)
        assert len(await storage_api.list(level)) == 0
        info = await storage_api.get_storage_level_info(level)
        assert info.used_size == 0

    isolation = new_isolation()
    asyncio.run_coroutine_threadsafe(
        _assert(session_id, addr, level), isolation.loop
    ).result()


@pytest.mark.parametrize("backend", ["mars"])
@pytest.mark.parametrize("_new_session", [new_session, new_test_session])
def test_new_session_backend(_new_session, backend):
    from ....services.task.execution.api import _name_to_config_cls

    config_cls = _name_to_config_cls[backend]
    original_config_init = config_cls.__init__
    original_deploy_band_resources = config_cls.get_deploy_band_resources
    with mock.patch.object(
        config_cls, "__init__", autospec=True
    ) as config_init, mock.patch.object(
        config_cls, "get_deploy_band_resources", autospec=True
    ) as deploy_band_resources:
        return_deploy_band_resources = []

        def _wrap_original_deploy_band_resources(*args, **kwargs):
            nonlocal return_deploy_band_resources
            return_deploy_band_resources = original_deploy_band_resources(
                *args, **kwargs
            )
            return return_deploy_band_resources

        config_init.side_effect = original_config_init
        deploy_band_resources.side_effect = _wrap_original_deploy_band_resources
        sess = _new_session(
            backend=backend, n_cpu=2, web=False, use_uvloop=False, default=True
        )
        try:
            assert config_init.call_count > 0
            assert deploy_band_resources.call_count > 0
            worker_pools = sess.default.client._cluster._worker_pools
            assert len(worker_pools) == len(return_deploy_band_resources)
            a = mt.ones((10, 10))
            b = a + 1
            res = b.to_numpy()
            np.testing.assert_array_equal(res, np.ones((10, 10)) + 1)
        finally:
            sess.stop_server()

    assert get_default_async_session() is None


@pytest.mark.asyncio
async def test_vineyard_operators(create_cluster):
    param = create_cluster[1]
    if param != "vineyard":
        pytest.skip("Vineyard is not enabled")

    session = get_default_async_session()

    # tensor
    raw = np.random.RandomState(0).rand(55, 55)
    a = mt.tensor(raw, chunk_size=15)
    info = await session.execute(a)  # n.b.: pre-execute
    await info

    b = mt.to_vineyard(a)
    info = await session.execute(b)
    await info
    object_id = (await session.fetch(b))[0]

    c = mt.from_vineyard(object_id)
    info = await session.execute(c)
    await info
    tensor = await session.fetch(c)
    np.testing.assert_allclose(tensor, raw)

    # dataframe
    raw = pd.DataFrame({"a": np.arange(0, 55), "b": np.arange(55, 110)})
    a = md.DataFrame(raw, chunk_size=15)
    b = a.to_vineyard()  # n.b.: no pre-execute
    info = await session.execute(b)
    await info
    object_id = (await session.fetch(b))[0][0]

    c = md.from_vineyard(object_id)
    info = await session.execute(c)
    await info
    df = await session.fetch(c)
    pd.testing.assert_frame_equal(df, raw)


@pytest.mark.parametrize(
    "config",
    [
        [
            {
                "enable_profiling": {
                    "slow_calls_duration_threshold": 0,
                    "slow_subtasks_duration_threshold": 0,
                }
            },
            EXPECT_PROFILING_STRUCTURE,
        ],
        [
            {
                "enable_profiling": {
                    "slow_calls_duration_threshold": 1000,
                    "slow_subtasks_duration_threshold": 1000,
                }
            },
            EXPECT_PROFILING_STRUCTURE_NO_SLOW,
        ],
        [{}, {}],
    ],
)
@pytest.mark.asyncio
async def test_execute(create_cluster, config):
    session = get_default_async_session()
    assert session.address is not None
    assert session.session_id is not None

    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1

    extra_config, expect_profiling_structure = config

    info = await session.execute(b, extra_config=extra_config)
    await info
    if extra_config:
        check_dict_structure_same(info.profiling_result(), expect_profiling_structure)
    else:
        assert not info.profiling_result()
    assert info.result() is None
    assert info.exception() is None
    assert info.progress() == 1
    np.testing.assert_equal(raw + 1, await session.fetch(b))

    with pytest.raises(ValueError):
        await session.fetch(b + 1)

    with pytest.raises(ValueError):
        await session.fetch(b[b < 0.6])

    del a, b

    if (
        not isinstance(session._isolated_session, _IsolatedWebSession)
        and session.client
    ):
        worker_pools = session.client._cluster._worker_pools
        await session.destroy()
        for worker_pool in worker_pools:
            if hasattr(worker_pool, "external_address"):
                _assert_storage_cleaned(
                    session.session_id,
                    worker_pool.external_address,
                    StorageLevel.MEMORY,
                )


@pytest.mark.asyncio
async def test_iterative_tiling(create_cluster):
    session = get_default_async_session()

    raw = np.random.RandomState(0).rand(30, 5)
    raw_df = pd.DataFrame(raw, index=np.arange(1, 31))

    df = md.DataFrame(raw_df, chunk_size=10)
    df = df[df[0] < 0.7]
    df2 = df.shift(2)

    info = await session.execute(df2)
    await info
    assert info.result() is None
    result = await session.fetch(df2)

    expected = raw_df[raw_df[0] < 0.7].shift(2)
    pd.testing.assert_frame_equal(result, expected)

    # test meta
    assert df2.index_value.min_val >= 1
    assert df2.index_value.max_val <= 30

    if (
        not isinstance(session._isolated_session, _IsolatedWebSession)
        and session.client
    ):
        worker_pools = session.client._cluster._worker_pools
        await session.destroy()
        for worker_pool in worker_pools:
            if hasattr(worker_pool, "external_address"):
                _assert_storage_cleaned(
                    session.session_id,
                    worker_pool.external_address,
                    StorageLevel.MEMORY,
                )


@pytest.mark.asyncio
async def test_execute_describe(create_cluster):
    s = np.random.RandomState(0)
    raw = pd.DataFrame(s.rand(100, 4), columns=list("abcd"))
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

    if (
        not isinstance(session._isolated_session, _IsolatedWebSession)
        and session.client
    ):
        worker_pools = session.client._cluster._worker_pools
        await session.destroy()
        for worker_pool in worker_pools:
            if hasattr(worker_pool, "external_address"):
                _assert_storage_cleaned(
                    session.session_id,
                    worker_pool.external_address,
                    StorageLevel.MEMORY,
                )


@pytest.mark.asyncio
async def test_execute_apply_closure(create_cluster):
    # DataFrame
    cols = [chr(ord("A") + i) for i in range(10)]
    raw = pd.DataFrame(dict((c, [i**2 for i in range(20)]) for c in cols))
    df = md.DataFrame(raw, chunk_size=5)

    x1 = pd.Series([i for i in range(10**4)])
    y1 = pd.Series([i for i in range(10**4)])

    def dataframe_closure(z1):
        return pd.concat([x1, y1], ignore_index=True)

    session = get_default_async_session()
    df_r = df.apply(dataframe_closure, axis=1)
    df_info = await session.execute(df_r)
    await df_info
    assert df_info.result() is None
    assert df_info.exception() is None
    assert df_info.progress() == 1

    df_result = await session.fetch(df_r)
    df_expected = raw.apply(dataframe_closure, axis=1)
    pd.testing.assert_frame_equal(df_result, df_expected)

    # Series
    idxes = [chr(ord("A") + i) for i in range(20)]
    s_raw = pd.Series([i**2 for i in range(20)], index=idxes)

    series = md.Series(s_raw, chunk_size=5)

    x2, y2 = 1, 2

    def series_closure(z2):
        return [z2 + x2, z2 + y2]

    series_r = series.apply(series_closure, convert_dtype=False)
    series_info = await session.execute(series_r)
    await series_info
    assert series_info.result() is None
    assert series_info.exception() is None
    assert series_info.progress() == 1

    series_result = await session.fetch(series_r)
    series_expected = s_raw.apply(series_closure, convert_dtype=False)
    pd.testing.assert_series_equal(series_result, series_expected)

    if (
        not isinstance(session._isolated_session, _IsolatedWebSession)
        and session.client
    ):
        worker_pools = session.client._cluster._worker_pools
        await session.destroy()
        for worker_pool in worker_pools:
            if hasattr(worker_pool, "external_address"):
                _assert_storage_cleaned(
                    session.session_id,
                    worker_pool.external_address,
                    StorageLevel.MEMORY,
                )


@pytest.mark.asyncio
@pytest.mark.parametrize("multiplier", [1, 3, 4])
async def test_execute_callable_closure(create_cluster, multiplier):
    # DataFrame
    cols = [chr(ord("A") + i) for i in range(10)]
    raw = pd.DataFrame(dict((c, [i**2 for i in range(20)]) for c in cols))
    df = md.DataFrame(raw, chunk_size=5)

    class callable_df:
        __slots__ = "x", "__dict__"

        def __init__(self, multiplier: int = 1):
            self.x = pd.Series([i for i in range(10**multiplier)])
            self.y = pd.Series([i for i in range(10**multiplier)])

        def __call__(self, pdf):
            return pd.concat([self.x, self.y], ignore_index=True)

    session = get_default_async_session()
    cdf = callable_df(multiplier=multiplier)
    df_r = df.apply(cdf, axis=1)
    df_info = await session.execute(df_r)
    await df_info
    assert df_info.result() is None
    assert df_info.exception() is None
    assert df_info.progress() == 1

    df_result = await session.fetch(df_r)
    df_expected = raw.apply(cdf, axis=1)
    pd.testing.assert_frame_equal(df_result, df_expected)

    if (
        not isinstance(session._isolated_session, _IsolatedWebSession)
        and session.client
    ):
        worker_pools = session.client._cluster._worker_pools
        await session.destroy()
        for worker_pool in worker_pools:
            if hasattr(worker_pool, "external_address"):
                _assert_storage_cleaned(
                    session.session_id,
                    worker_pool.external_address,
                    StorageLevel.MEMORY,
                )


@pytest.mark.asyncio
async def test_sync_execute_in_async(create_cluster):
    a = mt.ones((10, 10))
    b = a + 1
    res = b.to_numpy()
    np.testing.assert_array_equal(res, np.ones((10, 10)) + 1)


@pytest.mark.asyncio
async def test_fetch_infos(create_cluster):
    raw = np.random.RandomState(0).rand(30, 5)
    raw_df = pd.DataFrame(raw, index=np.arange(1, 31))

    df = md.DataFrame(raw_df, chunk_size=10)
    df.execute()
    fetched_infos = df.fetch_infos()

    assert "object_id" in fetched_infos
    assert "level" in fetched_infos
    assert "memory_size" in fetched_infos
    assert "store_size" in fetched_infos
    assert "bands" in fetched_infos

    fetched_infos = df.fetch_infos(fields=["object_id", "bands"])
    assert "object_id" in fetched_infos
    assert "bands" in fetched_infos
    assert len(fetched_infos) == 2

    fetch_infos((df, df), fields=None)
    results_infos = mr.ExecutableTuple([df, df]).execute()._fetch_infos()
    assert len(results_infos) == 2
    assert "object_id" in results_infos[0]
    assert "level" in results_infos[0]
    assert "memory_size" in results_infos[0]
    assert "store_size" in results_infos[0]
    assert "bands" in results_infos[0]


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

    # Test spawn a local function by the web session.
    def _my_func():
        print("output from function")

    r = mr.spawn(_my_func)
    info = await session.execute(r)
    await info
    assert info.result() is None
    assert info.exception() is None
    assert info.progress() == 1
    assert "output from function" in str(r.fetch_log(session=session))
    assert "output from function" in str(
        r.fetch_log(session=session, offsets="0k", sizes=[1000])
    )
    assert "output from function" in str(
        r.fetch_log(session=session, offsets={r.op.key: "0k"}, sizes=[1000])
    )

    df = md.DataFrame([1, 2, 3])
    # Test apply a lambda by the web session.
    r = df.apply(lambda x: x)
    info = await session.execute(r)
    await info
    assert info.result() is None
    assert info.exception() is None
    assert info.progress() == 1
    pd.testing.assert_frame_equal(await session.fetch(r), pd.DataFrame([1, 2, 3]))

    AsyncSession.reset_default()
    await session.destroy()


@pytest.mark.parametrize(
    "config",
    [
        [
            {
                "enable_profiling": {
                    "slow_calls_duration_threshold": 0,
                    "slow_subtasks_duration_threshold": 0,
                }
            },
            EXPECT_PROFILING_STRUCTURE,
        ],
        [
            {
                "enable_profiling": {
                    "slow_calls_duration_threshold": 1000,
                    "slow_subtasks_duration_threshold": 1000,
                }
            },
            EXPECT_PROFILING_STRUCTURE_NO_SLOW,
        ],
        [{}, {}],
    ],
)
@pytest.mark.asyncio
async def test_web_session(create_cluster, config):
    client = create_cluster[0]
    session_id = str(uuid.uuid4())
    web_address = client.web_address
    session = await AsyncSession.init(
        web_address, session_id, request_rewriter=lambda x: x
    )
    assert await session.get_web_endpoint() == web_address
    session.as_default()
    assert isinstance(session._isolated_session, _IsolatedWebSession)
    await test_execute(client, config)
    await test_iterative_tiling(client)
    AsyncSession.reset_default()
    await session.destroy()
    await _run_web_session_test(web_address)

    worker_pools = client._cluster._worker_pools
    for worker_pool in worker_pools:
        if hasattr(worker_pool, "external_address"):
            _assert_storage_cleaned(
                session.session_id, worker_pool.external_address, StorageLevel.MEMORY
            )


@pytest.mark.parametrize("config", [{"backend": "mars"}])
def test_sync_execute(config):
    session = new_session(
        backend=config["backend"], n_cpu=2, web=False, use_uvloop=False
    )

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

        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, "test.csv")
            pdf = pd.DataFrame(
                np.random.RandomState(0).rand(100, 10),
                columns=[f"col{i}" for i in range(10)],
            )
            pdf.to_csv(file_path, index=False)

            df = md.read_csv(
                file_path,
                chunk_bytes=os.stat(file_path).st_size / 5,
                incremental_index=True,
            )
            result = df.sum(axis=1).execute().fetch()
            expected = pd.read_csv(file_path).sum(axis=1)
            pd.testing.assert_series_equal(result, expected)

            df = md.read_csv(
                file_path,
                chunk_bytes=os.stat(file_path).st_size / 5,
                incremental_index=True,
            )
            result = df.head(10).execute().fetch()
            expected = pd.read_csv(file_path).head(10)
            pd.testing.assert_frame_equal(result, expected)

    for worker_pool in session._session.client._cluster._worker_pools:
        _assert_storage_cleaned(
            session.session_id, worker_pool.external_address, StorageLevel.MEMORY
        )

    session.stop_server()
    assert get_default_async_session() is None


def test_no_default_session():
    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1

    with pytest.warns(Warning):
        execute(b, show_progress=False)

    np.testing.assert_array_equal(fetch(b), raw + 1)
    fetch_infos(b, fields=None)
    assert get_default_async_session() is not None
    stop_server()
    assert get_default_async_session() is None


@pytest.mark.asyncio
async def test_session_set_progress(create_cluster):
    session = get_default_async_session()
    assert session.address is not None
    assert session.session_id is not None

    def f1(interval: float, count: int):
        for idx in range(count):
            time.sleep(interval)
            get_context().set_progress((1 + idx) * 1.0 / count)

    r = mr.spawn(f1, args=(0.5, 10))

    info = await session.execute(r)

    for _ in range(20):
        if 0 < info.progress() < 1:
            break
        await asyncio.sleep(0.1)
    else:
        raise Exception(f"progress test failed, actual value {info.progress()}.")

    await info
    assert info.result() is None
    assert info.exception() is None
    assert info.progress() == 1


@pytest.mark.asyncio
async def test_session_get_progress(create_cluster):
    session = get_default_async_session()
    assert session.address is not None
    assert session.session_id is not None

    raw = np.random.rand(100, 4)
    t = mt.tensor(raw, chunk_size=50)

    def f1(c):
        time.sleep(0.5)
        return c

    t1 = t.sum()
    t2 = t1.map_chunk(f1)
    r = t2.map_chunk(f1)
    info = await session.execute(r)

    for _ in range(100):
        if 0 < info.progress() < 1:
            break
        await asyncio.sleep(0.1)
    else:
        raise Exception(f"progress test failed, actual value {info.progress()}.")

    await info
    assert info.result() is None
    assert info.exception() is None
    assert info.progress() == 1


@pytest.fixture
def setup_session(request):
    param = getattr(request, "param", {})
    config = param.get("config", {})
    session = new_session(
        backend=config.get("backend", "mars"), n_cpu=2, use_uvloop=False, config=config
    )
    assert session.get_web_endpoint() is not None

    try:
        with session, option_context({"show_progress": False}):
            yield session
    finally:
        session.stop_server()


WeakTaskProcessorRefs = weakref.WeakSet()


class CheckRefTaskProcessor(TaskProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        WeakTaskProcessorRefs.add(self)

    async def run(self):
        # Trigger tileable gc before execute.
        gc.collect()
        return await super().run()

    @staticmethod
    def check_ref_count(count):
        for _ in range(10):
            if len(WeakTaskProcessorRefs) == count:
                break
            time.sleep(1)
        else:
            raise Exception(
                f"Check TaskProcessor weakref failed, expect {count} instances, "
                f"but got {WeakTaskProcessorRefs}"
            )


@pytest.mark.parametrize(
    "setup_session",
    [
        {
            "config": {
                "task.default_config.reserved_finish_tasks": 2,
                "task.task_processor_cls": CheckRefTaskProcessor,
            }
        }
    ],
    indirect=True,
)
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

    CheckRefTaskProcessor.check_ref_count(4)

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

    CheckRefTaskProcessor.check_ref_count(2)

    rs = np.random.RandomState(0)
    pdf = pd.DataFrame({"a": rs.randint(10, size=10), "b": rs.rand(10)})
    df = md.DataFrame(pdf, chunk_size=5)
    df2 = df.groupby("a").agg("mean", method="shuffle")
    result = df2.execute().fetch()
    expected = pdf.groupby("a").agg("mean")
    pd.testing.assert_frame_equal(result, expected)

    CheckRefTaskProcessor.check_ref_count(3)

    del df, df2
    ref_counts = session._get_ref_counts()
    assert len(ref_counts) == 0

    CheckRefTaskProcessor.check_ref_count(2)

    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")
        pdf = pd.DataFrame(
            np.random.RandomState(0).rand(100, 10),
            columns=[f"col{i}" for i in range(10)],
        )
        pdf.to_csv(file_path, index=False)

        df = md.read_csv(file_path, chunk_bytes=os.stat(file_path).st_size / 5)
        df2 = df.head(10)

        result = df2.execute().fetch()
        expected = pdf.head(10)
        pd.testing.assert_frame_equal(result, expected)

        del df, df2

        ref_counts = session._get_ref_counts()
        assert len(ref_counts) == 0

    for a in ((1, 1, 1, 2, 2, 3), [1, 1, 1, 2, 2, 3]):
        splits = mt.split(a, (3, 5))
        assert len(splits) == 3
        splits0 = splits[0].execute().fetch()
        np.testing.assert_array_equal(splits0, (1, 1, 1))
        splits1 = splits[1].execute().fetch()
        np.testing.assert_array_equal(splits1, (2, 2))
        splits2 = splits[2].execute().fetch()
        np.testing.assert_array_equal(splits2, (3,))

    del splits, splits0, splits1, splits2

    gc.collect()
    ref_counts = session._get_ref_counts()
    assert len(ref_counts) == 0

    worker_addr = session._session.client._cluster._worker_pools[0].external_address
    _assert_storage_cleaned(session.session_id, worker_addr, StorageLevel.MEMORY)


def _assert_worker_pool_storage_cleaned(session):
    worker_addr = session._session.client._cluster._worker_pools[0].external_address
    _assert_storage_cleaned(session.session_id, worker_addr, StorageLevel.MEMORY)


def _cancel_when_execute(session, cancelled):
    def run():
        time.sleep(200)

    rs = [mr.spawn(run) for _ in range(10)]
    execute(*rs, cancelled=cancelled)

    assert all(not r._executed_sessions for r in rs)

    del rs
    time.sleep(0.5)
    ref_counts = session._get_ref_counts()
    assert len(ref_counts) == 0


def _cancel_assert_when_execute(session, cancelled):
    _assert_worker_pool_storage_cleaned(session)
    _cancel_when_execute(session, cancelled)


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
    time.sleep(0.5)
    ref_counts = session._get_ref_counts()
    assert len(ref_counts) == 0


@pytest.mark.parametrize("test_func", [_cancel_assert_when_execute, _cancel_when_tile])
def test_cancel(create_cluster, test_func):
    session = get_default_session()

    async def _new_cancel_event():
        return asyncio.Event()

    isolation = new_isolation()
    cancelled = asyncio.run_coroutine_threadsafe(
        _new_cancel_event(), isolation.loop
    ).result()

    def cancel():
        time.sleep(0.5)
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
    config = _load_config()

    config["third_party_modules"] = set()
    with pytest.raises(TypeError, match="set"):
        new_session(n_cpu=2, web=False, config=config)

    config["third_party_modules"] = {"supervisor": ["not_exists_for_supervisor"]}
    with pytest.raises(ModuleNotFoundError, match="not_exists_for_supervisor"):
        new_session(n_cpu=2, web=False, config=config)

    config["third_party_modules"] = {"worker": ["not_exists_for_worker"]}
    with pytest.raises(ModuleNotFoundError, match="not_exists_for_worker"):
        new_session(n_cpu=2, web=False, config=config)

    config["third_party_modules"] = ["mars.deploy.oscar.tests.modules.replace_op"]
    session = new_session(n_cpu=2, web=False, config=config)
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

    session = new_session(
        n_cpu=2, web=False, config=CONFIG_THIRD_PARTY_MODULES_TEST_FILE
    )
    # web not started
    assert session._session.client.web_address is None

    with session:
        # 1 main pool, 3 sub pools(2 worker + 1 io).
        assert len(get_output_filenames()) == 4

    session.stop_server()
    assert get_default_session() is None


@mock.patch("asyncio.base_events.logger")
def test_show_progress_raise_exception(m_log):
    loop = asyncio.get_event_loop()
    event = asyncio.Event()

    class ProgressBar:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *_):
            pass

        def update(self, progress: float):
            pass

    async def _exec():
        progress = Progress()
        profiling = Profiling()
        execution_info = ExecutionInfo(
            asyncio.create_task(event.wait()), progress, profiling, loop, list()
        )
        progress_bar = ProgressBar(True)
        cancel_event = asyncio.Event()
        loop.call_later(2, cancel_event.set)
        await _execute_with_progress(execution_info, progress_bar, 0.01, cancel_event)
        execution_info.get_future().set_exception(Exception("Expect Exception!!!"))

    loop.run_until_complete(_exec())
    assert len(m_log.mock_calls) < 3


min_task_runtime = 2


@pytest.fixture
async def speculative_cluster():
    config = _load_config()
    config["scheduling"]["speculation"]["enabled"] = True
    config["scheduling"]["speculation"]["dry"] = False
    config["scheduling"]["speculation"]["interval"] = 0.5
    config["scheduling"]["speculation"]["threshold"] = 0.2
    config["scheduling"]["speculation"]["min_task_runtime"] = min_task_runtime
    config["scheduling"]["speculation"]["multiplier"] = 2
    config["scheduling"]["speculation"]["max_concurrent_run"] = 10
    config["scheduling"]["subtask_cancel_timeout"] = 0.1
    config["scheduling"]["enable_kill_slot"] = True
    config["storage"]["backends"] = ["plasma"]
    config["storage"]["plasma"]["store_memory"] = 10 * 1024 * 1024
    client = await new_cluster(
        config=config,
        n_worker=5,
        n_cpu=10,
        use_uvloop=False,
    )
    async with client:
        yield client


@pytest.mark.timeout(timeout=500)
@pytest.mark.asyncio
async def test_task_speculation_execution(speculative_cluster):
    series_size = 10

    def time_consuming(start, x):
        print(f"subtask index {x}")
        if (
            x >= series_size - 1
        ):  # leave some workers not excluded from speculative submit.
            if time.time() - start < min_task_runtime:
                print(f"subtask with index {x} starts to hang.")
                time.sleep(1000000)
        return x * x

    from functools import partial

    assert (
        md.Series(list(range(series_size)), chunk_size=1)
        .apply(partial(time_consuming, time.time()))
        .sum()
        .execute()
        .fetch()
        == pd.Series(list(range(series_size))).apply(lambda x: x * x).sum()
    )


def test_naive_code_file():
    code_file = """
    import mars
    import mars.tensor as mt
    import os

    mars.new_session()
    try:
        result_path = os.environ["RESULTPATH"]
        with open(result_path, "w") as outf:
            outf.write(str(mt.ones((10, 10)).sum().execute()))
    finally:
        mars.stop_server()
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            script_path = os.path.join(temp_dir, "test_file.py")
            result_path = os.path.join(temp_dir, "result.txt")

            with open(script_path, "w") as file_obj:
                file_obj.write(textwrap.dedent(code_file))

            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.pathsep.join(sys.path)
            env["RESULTPATH"] = result_path
            proc = subprocess.Popen([sys.executable, script_path], env=env)
            pid = proc.pid
            proc.wait(120)

            with open(result_path, "r") as inp_file:
                assert 100 == int(float(inp_file.read()))
        except subprocess.TimeoutExpired:
            try:
                procs = [psutil.Process(pid)]
                procs.extend(procs[0].children(True))
                for proc in reversed(procs):
                    try:
                        proc.kill()
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                pass
            raise


@pytest.mark.parametrize("config", [{"backend": "mars"}])
def test_fetch_concat(config):
    session = new_session(
        backend=config["backend"], n_cpu=2, web=False, use_uvloop=False
    )
    assert session is not None

    with session:
        data = {"A": [i for i in range(10)]}
        df0 = md.DataFrame(data)
        df1 = df0[["A"]]
        df2 = df0[["A"]]
        df1 = df1.execute()
        df2 = df2.execute()
        df3 = md.concat([df1, df2], axis=1)
        ret = df3.execute()
        df4 = ret.fetch()

        pdf0 = pd.DataFrame(data)
        pdf1 = pdf0[["A"]]
        pdf2 = pdf0[["A"]]
        pdf3 = pd.concat([pdf1, pdf2], axis=1)

        assert pdf3.equals(df4)

    for worker_pool in session._session.client._cluster._worker_pools:
        _assert_storage_cleaned(
            session.session_id, worker_pool.external_address, StorageLevel.MEMORY
        )

    session.stop_server()
    assert get_default_async_session() is None
