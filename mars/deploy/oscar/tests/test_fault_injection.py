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

import os
import pytest
import numpy as np

import mars.tensor as mt
from mars.remote import spawn
from mars.deploy.oscar.local import new_cluster
from mars.deploy.oscar.session import get_default_async_session
from mars.oscar.errors import ServerClosed
from mars.services.tests.fault_injection_manager import (
    FaultType,
    AbstractFaultInjectionManager,
    ExtraConfigKey,
    FaultInjectionError,
    FaultInjectionUnhandledError
)

CONFIG_FILE = os.path.join(
        os.path.dirname(__file__), 'fault_injection_config.yml')
RERUN_SUBTASK_CONFIG_FILE = os.path.join(
        os.path.dirname(__file__), 'fault_injection_config_with_rerun.yml')


@pytest.fixture
async def fault_cluster(request):
    param = getattr(request, "param", {})
    start_method = os.environ.get('POOL_START_METHOD', None)
    client = await new_cluster(subprocess_start_method=start_method,
                               config=param.get('config', CONFIG_FILE),
                               n_worker=2,
                               n_cpu=2)
    async with client:
        yield client


async def create_fault_injection_manager(session_id, address, fault_count, fault_type):
    class FaultInjectionManager(AbstractFaultInjectionManager):
        def __init__(self):
            self._fault_count = fault_count

        def set_fault_count(self, count):
            self._fault_count = count

        def on_execute_operand(self) -> FaultType:
            if self._fault_count > 0:
                self._fault_count -= 1
                return fault_type
            return FaultType.NoFault

    await FaultInjectionManager.create(session_id, address)
    return FaultInjectionManager.name


@pytest.mark.parametrize('fault_and_exception',
                         [[FaultType.Exception,
                           pytest.raises(FaultInjectionError, match='Fault Injection')],
                          [FaultType.UnhandledException,
                            pytest.raises(FaultInjectionUnhandledError, match='Fault Injection Unhandled')],
                          [FaultType.ProcessExit,
                           pytest.raises(ServerClosed)]])
@pytest.mark.asyncio
async def test_fault_inject_subtask_processor(fault_cluster, fault_and_exception):
    fault_type, first_run_raises = fault_and_exception
    name = await create_fault_injection_manager(
        session_id=fault_cluster.session.session_id,
        address=fault_cluster.session.address,
        fault_count=1,
        fault_type=fault_type)
    extra_config = {ExtraConfigKey.FAULT_INJECTION_MANAGER_NAME: name}

    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1

    with first_run_raises:
        b.execute(extra_config=extra_config)

    # execute again may raise an ConnectionRefusedError if the
    # ProcessExit occurred.


@pytest.mark.parametrize('fault_cluster',
                         [{'config': RERUN_SUBTASK_CONFIG_FILE}],
                         indirect=True)
@pytest.mark.parametrize('fault_config',
                         [[FaultType.Exception, 1,
                           pytest.raises(FaultInjectionError, match='Fault Injection')],
                          [FaultType.ProcessExit, 1,
                           pytest.raises(ServerClosed)]])
@pytest.mark.asyncio
async def test_rerun_subtask(fault_cluster, fault_config):
    fault_type, fault_count, expect_raises = fault_config
    name = await create_fault_injection_manager(
        session_id=fault_cluster.session.session_id,
        address=fault_cluster.session.address,
        fault_count=fault_count,
        fault_type=fault_type)
    extra_config = {ExtraConfigKey.FAULT_INJECTION_MANAGER_NAME: name}
    session = get_default_async_session()

    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1

    info = await session.execute(b, extra_config=extra_config)
    await info
    assert info.result() is None
    assert info.exception() is None

    r = await session.fetch(b)
    np.testing.assert_array_equal(r, raw + 1)

    fault_injection_manager = await session.get_remote_object(
            fault_cluster.session.session_id, name)
    await fault_injection_manager.set_fault_count(1)

    # the extra config overwrites the default config.
    extra_config['subtask_max_retries'] = 0
    info = await session.execute(b, extra_config=extra_config)
    with expect_raises:
        await info


@pytest.mark.parametrize('fault_cluster',
                         [{'config': RERUN_SUBTASK_CONFIG_FILE}],
                         indirect=True)
@pytest.mark.asyncio
async def test_rerun_subtask_unhandled(fault_cluster):
    name = await create_fault_injection_manager(
        session_id=fault_cluster.session.session_id,
        address=fault_cluster.session.address,
        fault_count=1,
        fault_type=FaultType.UnhandledException)
    extra_config = {ExtraConfigKey.FAULT_INJECTION_MANAGER_NAME: name}

    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1

    with pytest.raises(FaultInjectionUnhandledError):
        b.execute(extra_config=extra_config)


@pytest.mark.parametrize('fault_cluster',
                         [{'config': RERUN_SUBTASK_CONFIG_FILE}],
                         indirect=True)
@pytest.mark.parametrize('fault_config',
                         [[FaultType.Exception, 1,
                           pytest.raises(FaultInjectionError, match='Fault Injection')],
                          [FaultType.ProcessExit, 1,
                           pytest.raises(ServerClosed)]])
@pytest.mark.asyncio
async def test_retryable(fault_cluster, fault_config):
    fault_type, fault_count, expect_raises = fault_config
    name = await create_fault_injection_manager(
        session_id=fault_cluster.session.session_id,
        address=fault_cluster.session.address,
        fault_count=fault_count,
        fault_type=fault_type)
    extra_config = {ExtraConfigKey.FAULT_INJECTION_MANAGER_NAME: name}

    def f(x):
        return x + 1

    r = spawn(f, args=(1,), retry_when_fail=False)
    with expect_raises:
        r.execute(extra_config=extra_config)
