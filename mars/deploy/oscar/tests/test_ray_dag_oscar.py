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

from ....tests.core import require_ray
from ....utils import lazy_import
from ..ray import new_cluster, _load_config
from ..tests import test_local

ray = lazy_import("ray")
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "local_test_with_ray_config.yml")


@pytest.fixture
async def create_cluster(request):
    param = getattr(request, "param", {})
    ray_config = _load_config(CONFIG_FILE)
    ray_config.update(param.get("config", {}))
    client = await new_cluster(
        supervisor_mem=1 * 1024**3,
        worker_num=2,
        worker_cpu=2,
        worker_mem=1 * 1024**3,
        backend="ray",
        config=ray_config,
    )
    async with client:
        yield client, param


@require_ray
@pytest.mark.asyncio
async def test_iterative_tiling(ray_start_regular_shared2, create_cluster):
    await test_local.test_iterative_tiling(create_cluster)


@pytest.mark.asyncio
@require_ray
async def test_execute_describe(ray_start_regular_shared2, create_cluster):
    await test_local.test_execute_describe(create_cluster)


@require_ray
@pytest.mark.asyncio
async def test_execute_apply_closure(ray_start_regular_shared2, create_cluster):
    await test_local.test_execute_apply_closure(create_cluster)


@require_ray
@pytest.mark.parametrize("multiplier", [1, 3, 4])
@pytest.mark.asyncio
async def test_execute_callable_closure(
    ray_start_regular_shared2, create_cluster, multiplier
):
    await test_local.test_execute_callable_closure(create_cluster, multiplier)


@require_ray
@pytest.mark.parametrize(
    "create_cluster",
    [
        {
            "config": {
                "task.task_preprocessor_cls": "mars.deploy.oscar.tests.test_clean_up_and_restore_func.RayBackendFuncTaskPreprocessor",
                "subtask.subtask_processor_cls": "mars.deploy.oscar.tests.test_clean_up_and_restore_func.RayBackendFuncSubtaskProcessor",
            }
        }
    ],
    indirect=True,
)
@pytest.mark.asyncio
async def test_ray_dag_oscar_clean_up_and_restore_func(
    ray_start_regular_shared2, create_cluster
):
    await test_local.test_execute_apply_closure(create_cluster)
