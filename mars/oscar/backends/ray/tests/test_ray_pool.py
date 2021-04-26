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

import pytest

from mars.serialization.ray import register_ray_serializers, unregister_ray_serializers
from mars.tests.core import require_ray
from .....utils import lazy_import
from ...router import Router
from ..pool import RayMainPool
from ..utils import process_placement_to_address

ray = lazy_import('ray')


@pytest.fixture(scope="module")
def ray_start_regular_shared():
    register_ray_serializers()
    yield ray.init()
    ray.shutdown()
    unregister_ray_serializers()
    Router.set_instance(None)


@require_ray
@pytest.mark.asyncio
async def test_shutdown_sub_pool(ray_start_regular_shared):
    import ray
    pg_name, n_process = 'ray_cluster', 2
    if hasattr(ray.util, "get_placement_group"):
        pg, bundle_index = ray.util.placement_group(name=pg_name, bundles=[{'CPU': n_process}]), 0
        ray.get(pg.ready())
    else:
        pg, bundle_index = None, -1
    address = process_placement_to_address(pg_name, 0, process_index=0)
    actor_handle = ray.remote(RayMainPool).options(
        name=address, placement_group=pg, placement_group_bundle_index=bundle_index).remote()
    await actor_handle.start.remote(address, n_process)
    sub_pool_address1 = process_placement_to_address(pg_name, 0, process_index=1)
    sub_pool_handle1 = ray.get_actor(sub_pool_address1)
    sub_pool_address2 = process_placement_to_address(pg_name, 0, process_index=2)
    sub_pool_handle2 = ray.get_actor(sub_pool_address2)
    await actor_handle.actor_pool.remote('stop_sub_pool', sub_pool_address1, sub_pool_handle1, force=True)
    await actor_handle.actor_pool.remote('stop_sub_pool', sub_pool_address2, sub_pool_handle2, force=False)
    import ray.exceptions
    with pytest.raises(ray.exceptions.RayActorError):
        await sub_pool_handle1.health_check.remote()
        await sub_pool_handle2.health_check.remote()
