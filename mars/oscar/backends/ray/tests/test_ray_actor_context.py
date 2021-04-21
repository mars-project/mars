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

import inspect

import pytest

from .....utils import lazy_import
from ...mars.tests import test_mars_actor_context
from ...router import Router
from ..pool import RayMainPool
from ..utils import process_placement_to_address
from mars.tests.core import require_ray

ray = lazy_import('ray')


pg_name, n_process = 'ray_cluster', 2


@pytest.fixture(scope="module")
def ray_start_regular_shared():
    try:
        from ray.cluster_utils import Cluster
    except ModuleNotFoundError:
        from ray._private.cluster_utils import Cluster
    cluster = Cluster()
    remote_nodes = []
    num_nodes = 3
    for i in range(num_nodes):
        remote_nodes.append(cluster.add_node(num_cpus=10))
        if len(remote_nodes) == 1:
            ray.init()
    if hasattr(ray.util, "get_placement_group"):
        pg = ray.util.placement_group(name=pg_name, bundles=[{'CPU': n_process}], strategy="SPREAD")
        ray.get(pg.ready())
    yield
    ray.shutdown()


@pytest.fixture
def actor_pool_context():
    from mars.serialization.ray import register_ray_serializers, unregister_ray_serializers
    register_ray_serializers()
    address = process_placement_to_address(pg_name, 0, process_index=0)
    # Hold actor_handle to avoid actor being freed.
    if hasattr(ray.util, "get_placement_group"):
        pg, bundle_index = ray.util.get_placement_group(pg_name), 0
    else:
        pg, bundle_index = None, -1
    actor_handle = ray.remote(RayMainPool).options(
        name=address, placement_group=pg, placement_group_bundle_index=bundle_index).remote()
    ray.get(actor_handle.start.remote(address, n_process))

    class ProxyPool:

        def __init__(self, ray_pool_actor_handle):
            self.ray_pool_actor_handle = ray_pool_actor_handle

        def __getattr__(self, item):
            if hasattr(RayMainPool, item) and inspect.isfunction(getattr(RayMainPool, item)):
                def call(*args, **kwargs):
                    ray.get(self.ray_pool_actor_handle.actor_pool.remote(item, *args, **kwargs))
                return call

            return ray.get(self.ray_pool_actor_handle.actor_pool.remote(item))

    yield ProxyPool(actor_handle)
    for addr in [process_placement_to_address(pg_name, 0, process_index=i) for i in range(n_process)]:
        try:
            ray.kill(ray.get_actor(addr))
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except
            pass
    Router.set_instance(None)
    unregister_ray_serializers()


@require_ray
@pytest.mark.asyncio
async def test_simple_local_actor_pool(ray_start_regular_shared, actor_pool_context):
    await test_mars_actor_context.test_simple_local_actor_pool(actor_pool_context)


@require_ray
@pytest.mark.asyncio
async def test_mars_post_create_pre_destroy(ray_start_regular_shared, actor_pool_context):
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
