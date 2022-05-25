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
import traceback
import numpy as np
import pandas as pd

from .... import dataframe as md
from .... import tensor as mt
from ....oscar.errors import ServerClosed
from ....remote import spawn
from ....services.tests.fault_injection_manager import (
    AbstractFaultInjectionManager,
    ExtraConfigKey,
    FaultInjectionError,
    FaultInjectionUnhandledError,
    FaultPosition,
    FaultType,
)
from ....tensor.base.psrs import PSRSConcatPivot
from ..local import new_cluster
from ..session import get_default_async_session

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "fault_injection_config.yml")
RERUN_SUBTASK_CONFIG_FILE = os.path.join(
    os.path.dirname(__file__), "fault_injection_config_with_rerun.yml"
)


@pytest.fixture
async def fault_cluster(request):
    param = getattr(request, "param", {})
    start_method = os.environ.get("POOL_START_METHOD", None)
    client = await new_cluster(
        subprocess_start_method=start_method,
        config=param.get("config", CONFIG_FILE),
        n_worker=2,
        n_cpu=2,
    )
    async with client:
        yield client


async def create_fault_injection_manager(
    session_id, address, fault_count, fault_type, fault_op_types=None
):
    class FaultInjectionManager(AbstractFaultInjectionManager):
        def __init__(self):
            self._fault_count = fault_count

        def set_fault_count(self, count):
            self._fault_count = count

        def get_fault_count(self):
            return self._fault_count

        def get_fault(self, pos: FaultPosition, ctx=None) -> FaultType:
            # Check op types if fault_op_types provided.
            if fault_op_types and type(ctx.get("operand")) not in fault_op_types:
                return FaultType.NoFault
            if self._fault_count.get(pos, 0) > 0:
                self._fault_count[pos] -= 1
                return fault_type
            return FaultType.NoFault

    await FaultInjectionManager.create(session_id, address)
    return FaultInjectionManager.name


@pytest.mark.parametrize(
    "fault_and_exception",
    [
        [
            FaultType.Exception,
            {FaultPosition.ON_EXECUTE_OPERAND: 1},
            pytest.raises(FaultInjectionError, match="Fault Injection"),
            True,
        ],
        [
            FaultType.UnhandledException,
            {FaultPosition.ON_EXECUTE_OPERAND: 1},
            pytest.raises(
                FaultInjectionUnhandledError, match="Fault Injection Unhandled"
            ),
            True,
        ],
        [
            FaultType.ProcessExit,
            {FaultPosition.ON_EXECUTE_OPERAND: 1},
            pytest.raises(ServerClosed),
            False,  # The ServerClosed raised from current process directly.
        ],
        [
            FaultType.Exception,
            {FaultPosition.ON_RUN_SUBTASK: 1},
            pytest.raises(FaultInjectionError, match="Fault Injection"),
            False,
        ],
    ],
)
@pytest.mark.asyncio
async def test_fault_inject_subtask_processor(fault_cluster, fault_and_exception):
    fault_type, fault_count, first_run_raises, check_error_prefix = fault_and_exception
    name = await create_fault_injection_manager(
        session_id=fault_cluster.session.session_id,
        address=fault_cluster.session.address,
        fault_count=fault_count,
        fault_type=fault_type,
    )
    extra_config = {ExtraConfigKey.FAULT_INJECTION_MANAGER_NAME: name}

    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1

    with first_run_raises as ex:
        b.execute(extra_config=extra_config)

    if check_error_prefix:
        assert str(ex.value).count("address") == 1
        assert str(ex.value).count("pid") == 1

    # execute again may raise an ConnectionRefusedError if the
    # ProcessExit occurred.


@pytest.mark.parametrize(
    "fault_cluster", [{"config": RERUN_SUBTASK_CONFIG_FILE}], indirect=True
)
@pytest.mark.parametrize(
    "fault_config",
    [
        [
            FaultType.Exception,
            {FaultPosition.ON_EXECUTE_OPERAND: 1},
            pytest.raises(FaultInjectionError, match="Fault Injection"),
        ],
        [
            FaultType.ProcessExit,
            {FaultPosition.ON_EXECUTE_OPERAND: 1},
            pytest.raises(ServerClosed),
        ],
        [
            FaultType.Exception,
            {FaultPosition.ON_RUN_SUBTASK: 1},
            pytest.raises(FaultInjectionError, match="Fault Injection"),
        ],
    ],
)
@pytest.mark.asyncio
async def test_rerun_subtask(fault_cluster, fault_config):
    fault_type, fault_count, expect_raises = fault_config
    name = await create_fault_injection_manager(
        session_id=fault_cluster.session.session_id,
        address=fault_cluster.session.address,
        fault_count=fault_count,
        fault_type=fault_type,
    )
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
        fault_cluster.session.session_id, name
    )
    await fault_injection_manager.set_fault_count({FaultPosition.ON_EXECUTE_OPERAND: 1})

    # the extra config overwrites the default config.
    extra_config["subtask_max_retries"] = 0
    extra_config["subtask_max_reschedules"] = 0
    info = await session.execute(b, extra_config=extra_config)
    with expect_raises:
        await info


@pytest.mark.parametrize(
    "fault_cluster", [{"config": RERUN_SUBTASK_CONFIG_FILE}], indirect=True
)
@pytest.mark.parametrize(
    "fault_config",
    [
        [FaultType.Exception, {FaultPosition.ON_EXECUTE_OPERAND: 1}, [PSRSConcatPivot]],
        [
            FaultType.ProcessExit,
            {FaultPosition.ON_EXECUTE_OPERAND: 1},
            [PSRSConcatPivot],
        ],
    ],
)
@pytest.mark.asyncio
async def test_rerun_subtask_describe(fault_cluster, fault_config):
    fault_type, fault_count, fault_op_types = fault_config
    name = await create_fault_injection_manager(
        session_id=fault_cluster.session.session_id,
        address=fault_cluster.session.address,
        fault_count=fault_count,
        fault_type=fault_type,
        fault_op_types=fault_op_types,
    )
    extra_config = {ExtraConfigKey.FAULT_INJECTION_MANAGER_NAME: name}
    session = get_default_async_session()

    s = np.random.RandomState(0)
    raw = pd.DataFrame(s.rand(100, 4), columns=list("abcd"))
    df = md.DataFrame(raw, chunk_size=30)

    r = df.describe()
    info = await session.execute(r, extra_config=extra_config)
    await info
    assert info.result() is None
    assert info.exception() is None
    assert info.progress() == 1
    res = await session.fetch(r)
    pd.testing.assert_frame_equal(res, raw.describe())

    fault_injection_manager = await session.get_remote_object(
        fault_cluster.session.session_id, name
    )
    remain_fault_count = await fault_injection_manager.get_fault_count()
    for key in fault_count:
        assert remain_fault_count[key] == 0


@pytest.mark.parametrize(
    "fault_cluster", [{"config": RERUN_SUBTASK_CONFIG_FILE}], indirect=True
)
@pytest.mark.parametrize(
    "fault_config",
    [
        [
            FaultType.UnhandledException,
            {FaultPosition.ON_EXECUTE_OPERAND: 1},
            pytest.raises(FaultInjectionUnhandledError),
            ["_UnhandledException", "handle_fault"],
        ],
        [
            FaultType.Exception,
            {FaultPosition.ON_EXECUTE_OPERAND: 100},
            pytest.raises(FaultInjectionError),
            ["_ExceedMaxRerun", "handle_fault"],
        ],
    ],
)
@pytest.mark.asyncio
async def test_rerun_subtask_fail(fault_cluster, fault_config):
    fault_type, fault_count, expect_raises, exception_match = fault_config
    name = await create_fault_injection_manager(
        session_id=fault_cluster.session.session_id,
        address=fault_cluster.session.address,
        fault_count=fault_count,
        fault_type=fault_type,
    )
    exception_typename, stack_string = exception_match
    extra_config = {ExtraConfigKey.FAULT_INJECTION_MANAGER_NAME: name}

    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1

    with expect_raises as e:
        b.execute(extra_config=extra_config)

    tb_str = "".join(traceback.format_tb(e.tb))
    assert e.value.__wrapname__ == exception_typename, tb_str
    assert e.traceback[-1].name == stack_string, tb_str


@pytest.mark.parametrize(
    "fault_cluster", [{"config": RERUN_SUBTASK_CONFIG_FILE}], indirect=True
)
@pytest.mark.parametrize(
    "fault_config",
    [
        [
            FaultType.Exception,
            {FaultPosition.ON_EXECUTE_OPERAND: 1},
            pytest.raises(FaultInjectionError, match="RemoteFunction"),
            ["_UnretryableException", "handle_fault"],
        ],
        [
            FaultType.ProcessExit,
            {FaultPosition.ON_EXECUTE_OPERAND: 1},
            pytest.raises(ServerClosed),
            ["_UnretryableException", "*"],
        ],
    ],
)
@pytest.mark.asyncio
async def test_retryable(fault_cluster, fault_config):
    fault_type, fault_count, expect_raises, exception_match = fault_config
    name = await create_fault_injection_manager(
        session_id=fault_cluster.session.session_id,
        address=fault_cluster.session.address,
        fault_count=fault_count,
        fault_type=fault_type,
    )
    exception_typename, stack_string = exception_match
    extra_config = {ExtraConfigKey.FAULT_INJECTION_MANAGER_NAME: name}

    def f(x):
        return x + 1

    r = spawn(f, args=(1,), retry_when_fail=False)
    with expect_raises as e:
        r.execute(extra_config=extra_config)

    tb_str = "".join(traceback.format_tb(e.tb))
    assert e.value.__wrapname__ == exception_typename, tb_str
    assert stack_string == "*" or e.traceback[-1].name == stack_string, tb_str
