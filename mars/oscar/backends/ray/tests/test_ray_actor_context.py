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

import inspect
import time

import pytest

from .....utils import lazy_import
from .....tests.core import require_ray
from ...mars.tests import test_mars_actor_context
from ...router import Router
from ..backend import RayActorBackend
from ..communication import RayServer
from ..pool import RayMainPool
from ..utils import process_placement_to_address

ray = lazy_import("ray")


@pytest.fixture
async def actor_pool_context():
    pg_name, n_process = f"ray_cluster_{time.time_ns()}", 2
    address = process_placement_to_address(pg_name, 0, process_index=0)
    # Hold actor_handle to avoid actor being freed.
    pg = ray.util.placement_group(
        name=pg_name, bundles=[{"CPU": n_process}], strategy="SPREAD"
    )
    ray.get(pg.ready())
    pg, _ = ray.util.get_placement_group(pg_name), 0
    pool_handle = await RayActorBackend._create_ray_pools(address, n_process)
    await pool_handle.start.remote()

    class ProxyPool:
        def __init__(self, ray_pool_actor_handle):
            self.ray_pool_actor_handle = ray_pool_actor_handle

        def __getattr__(self, item):
            if hasattr(RayMainPool, item) and inspect.isfunction(
                getattr(RayMainPool, item)
            ):

                def call(*args, **kwargs):
                    ray.get(
                        self.ray_pool_actor_handle.actor_pool.remote(
                            item, *args, **kwargs
                        )
                    )

                return call

            return ray.get(self.ray_pool_actor_handle.actor_pool.remote(item))

    yield ProxyPool(pool_handle)
    for addr in [
        process_placement_to_address(pg_name, 0, process_index=i)
        for i in range(n_process)
    ]:
        try:
            # kill main pool first to avoid main pool monitor task recreate sub pool
            ray.kill(ray.get_actor(addr))
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except
            pass
    ray.util.remove_placement_group(pg)
    Router.set_instance(None)
    RayServer.clear()


@require_ray
@pytest.mark.asyncio
async def test_simple_local_actor_pool(ray_start_regular_shared, actor_pool_context):
    await test_mars_actor_context.test_simple_local_actor_pool(actor_pool_context)


@require_ray
@pytest.mark.asyncio
async def test_mars_post_create_pre_destroy(
    ray_start_regular_shared, actor_pool_context
):
    await test_mars_actor_context.test_mars_post_create_pre_destroy(actor_pool_context)


@require_ray
@pytest.mark.asyncio
async def test_mars_create_actor(ray_start_regular_shared, actor_pool_context):
    await test_mars_actor_context.test_mars_create_actor(actor_pool_context)


@require_ray
@pytest.mark.asyncio
async def test_mars_create_actor_error(ray_start_regular_shared, actor_pool_context):
    await test_mars_actor_context.test_mars_create_actor_error(actor_pool_context)


@require_ray
@pytest.mark.asyncio
async def test_mars_send(ray_start_regular_shared, actor_pool_context):
    await test_mars_actor_context.test_mars_send(actor_pool_context)


@require_ray
@pytest.mark.asyncio
async def test_mars_send_error(ray_start_regular_shared, actor_pool_context):
    await test_mars_actor_context.test_mars_send_error(actor_pool_context)


@require_ray
@pytest.mark.asyncio
async def test_mars_tell(ray_start_regular_shared, actor_pool_context):
    await test_mars_actor_context.test_mars_tell(actor_pool_context)


@require_ray
@pytest.mark.asyncio
async def test_mars_batch_method(ray_start_regular_shared, actor_pool_context):
    await test_mars_actor_context.test_mars_batch_method(actor_pool_context)


@require_ray
@pytest.mark.asyncio
async def test_mars_destroy_has_actor(ray_start_regular_shared, actor_pool_context):
    await test_mars_actor_context.test_mars_destroy_has_actor(actor_pool_context)


@require_ray
@pytest.mark.asyncio
async def test_mars_resource_lock(ray_start_regular_shared, actor_pool_context):
    await test_mars_actor_context.test_mars_resource_lock(actor_pool_context)


@require_ray
@pytest.mark.asyncio
async def test_promise_chain(ray_start_regular_shared, actor_pool_context):
    await test_mars_actor_context.test_promise_chain(actor_pool_context)
