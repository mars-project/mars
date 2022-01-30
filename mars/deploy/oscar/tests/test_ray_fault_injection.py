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

from ....oscar.errors import ServerClosed
from ....services.tests.fault_injection_manager import (
    FaultInjectionError,
    FaultInjectionUnhandledError,
    FaultPosition,
    FaultType,
)
from ....tensor.base.psrs import PSRSConcatPivot
from ....tests.core import require_ray
from ....utils import lazy_import
from ..ray import new_cluster, _load_config
from ..tests import test_fault_injection

ray = lazy_import("ray")

RAY_CONFIG_FILE = os.path.join(
    os.path.dirname(__file__), "local_test_with_ray_config.yml"
)
FAULT_INJECTION_CONFIG = {
    "third_party_modules": ["mars.services.tests.fault_injection_patch"],
}
SUBTASK_RERUN_CONFIG = {
    "scheduling": {
        "subtask_max_retries": 2,
        "subtask_max_reschedules": 2,
    }
}


@pytest.fixture
async def fault_cluster(request):
    param = getattr(request, "param", {})
    ray_config = _load_config(RAY_CONFIG_FILE)
    ray_config.update(FAULT_INJECTION_CONFIG)
    ray_config.update(param.get("config", {}))
    client = await new_cluster(
        "test_cluster",
        worker_num=2,
        worker_cpu=2,
        worker_mem=1 * 1024**3,
        config=ray_config,
    )
    async with client:
        yield client


@require_ray
@pytest.mark.parametrize(
    "fault_and_exception",
    [
        [
            FaultType.Exception,
            {FaultPosition.ON_EXECUTE_OPERAND: 1},
            pytest.raises(FaultInjectionError, match="Fault Injection"),
        ],
        [
            FaultType.UnhandledException,
            {FaultPosition.ON_EXECUTE_OPERAND: 1},
            pytest.raises(
                FaultInjectionUnhandledError, match="Fault Injection Unhandled"
            ),
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
async def test_fault_inject_subtask_processor(
    ray_start_regular, fault_cluster, fault_and_exception
):
    await test_fault_injection.test_fault_inject_subtask_processor(
        fault_cluster, fault_and_exception
    )


@require_ray
@pytest.mark.parametrize(
    "fault_cluster", [{"config": SUBTASK_RERUN_CONFIG}], indirect=True
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
async def test_rerun_subtask(ray_start_regular, fault_cluster, fault_config):
    await test_fault_injection.test_rerun_subtask(fault_cluster, fault_config)


@require_ray
@pytest.mark.parametrize(
    "fault_cluster", [{"config": SUBTASK_RERUN_CONFIG}], indirect=True
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
async def test_rerun_subtask_describe(ray_start_regular, fault_cluster, fault_config):
    await test_fault_injection.test_rerun_subtask_describe(fault_cluster, fault_config)


@require_ray
@pytest.mark.parametrize(
    "fault_cluster", [{"config": SUBTASK_RERUN_CONFIG}], indirect=True
)
@pytest.mark.asyncio
async def test_rerun_subtask_unhandled(ray_start_regular, fault_cluster):
    await test_fault_injection.test_rerun_subtask_unhandled(fault_cluster)


@require_ray
@pytest.mark.parametrize(
    "fault_cluster", [{"config": SUBTASK_RERUN_CONFIG}], indirect=True
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
    ],
)
@pytest.mark.asyncio
async def test_retryable(ray_start_regular, fault_cluster, fault_config):
    await test_fault_injection.test_retryable(fault_cluster, fault_config)
