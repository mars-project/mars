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

import pytest

from .....tests.core import require_ray
from . import test_task_manager
from .test_task_manager import actor_pool  # noqa: F401


actor_pool = actor_pool


@require_ray
@pytest.mark.parametrize("actor_pool", [{"backend": "ray"}], indirect=True)
@pytest.mark.asyncio
async def test_run_task(ray_start_regular_shared2, actor_pool):
    await test_task_manager.test_run_task(actor_pool)


@require_ray
@pytest.mark.parametrize("actor_pool", [{"backend": "ray"}], indirect=True)
@pytest.mark.asyncio
async def test_run_tasks_with_same_name(ray_start_regular_shared2, actor_pool):
    await test_task_manager.test_run_tasks_with_same_name(actor_pool)


@require_ray
@pytest.mark.parametrize("actor_pool", [{"backend": "ray"}], indirect=True)
@pytest.mark.asyncio
async def test_error_task(ray_start_regular_shared2, actor_pool):
    await test_task_manager.test_error_task(actor_pool)


@require_ray
@pytest.mark.parametrize("actor_pool", [{"backend": "ray"}], indirect=True)
@pytest.mark.asyncio
async def test_iterative_tiling(ray_start_regular_shared2, actor_pool):
    await test_task_manager.test_iterative_tiling(actor_pool)


@require_ray
@pytest.mark.parametrize("actor_pool", [{"backend": "ray"}], indirect=True)
@pytest.mark.asyncio
async def test_numexpr(ray_start_regular_shared2, actor_pool):
    await test_task_manager.test_numexpr(actor_pool)


@require_ray
@pytest.mark.parametrize("actor_pool", [{"backend": "ray"}], indirect=True)
@pytest.mark.asyncio
async def test_optimization(ray_start_regular_shared2, actor_pool):
    await test_task_manager.test_optimization(actor_pool)
