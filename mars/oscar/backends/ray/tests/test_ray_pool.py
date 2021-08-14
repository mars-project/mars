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

import pytest

import mars.oscar as mo
from mars.oscar.errors import ServerClosed
from mars.oscar.backends.allocate_strategy import ProcessIndex, MainPool
from mars.oscar.backends.ray.pool import RayMainPool, RayMainActorPool, create_actor_pool, PoolStatus
from mars.oscar.backends.ray.utils import process_placement_to_address
from mars.oscar.context import get_context
from mars.tests.core import require_ray
from mars.utils import lazy_import

ray = lazy_import('ray')


class TestActor(mo.Actor):
    async def kill(self, address, uid):
        actor_ref = await mo.actor_ref(address, uid)
        task = asyncio.create_task(actor_ref.crash())
        return await task

    async def crash(self):
        os._exit(0)


@require_ray
@pytest.mark.asyncio
async def test_main_pool(ray_start_regular):
    pg_name, n_process = 'ray_cluster', 3
    if hasattr(ray.util, "get_placement_group"):
        pg = ray.util.placement_group(name=pg_name, bundles=[{'CPU': n_process}])
        ray.get(pg.ready())
    address = process_placement_to_address(pg_name, 0, process_index=0)
    addresses = RayMainActorPool.get_external_addresses(address, n_process)
    assert addresses == [address] + [process_placement_to_address(pg_name, 0, process_index=i + 1)
                                     for i in range(n_process)]
    assert RayMainActorPool.gen_internal_address(0, address) == address

    main_actor_pool = await create_actor_pool(
        address, n_process=n_process, pool_cls=RayMainActorPool)
    main_actor_pool._monitor_task.cancel()  # avoid sub pool got restarted
    async with main_actor_pool:
        sub_processes = list(main_actor_pool.sub_processes.values())
        assert len(sub_processes) == n_process
        await main_actor_pool.kill_sub_pool(sub_processes[0], force=True)
        assert not (await main_actor_pool.is_sub_pool_alive(sub_processes[0]))
        await main_actor_pool.kill_sub_pool(sub_processes[1], force=False)
        assert not (await main_actor_pool.is_sub_pool_alive(sub_processes[1]))


@require_ray
@pytest.mark.asyncio
async def test_shutdown_sub_pool(ray_start_regular):
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


@require_ray
@pytest.mark.asyncio
async def test_server_closed(ray_start_regular):
    pg_name, n_process = 'ray_cluster', 1
    pg = ray.util.placement_group(name=pg_name, bundles=[{'CPU': n_process}])
    ray.get(pg.ready())
    address = process_placement_to_address(pg_name, 0, process_index=0)
    # start the actor pool
    actor_handle = await mo.create_actor_pool(address, n_process=n_process)
    await actor_handle.actor_pool.remote('start')

    ctx = get_context()
    actor_main = await ctx.create_actor(
        TestActor, address=address, uid='Test-main',
        allocate_strategy=ProcessIndex(0))

    actor_sub = await ctx.create_actor(
        TestActor, address=address, uid='Test-sub',
        allocate_strategy=ProcessIndex(1))

    # test calling from ray driver to ray actor
    task = asyncio.create_task(actor_sub.crash())

    with pytest.raises(ServerClosed):
        # process already died,
        # ServerClosed will be raised
        await task

    # wait for recover of sub pool
    await ctx.wait_actor_pool_recovered(actor_sub.address, address)

    # test calling from ray actor to ray actor
    task = asyncio.create_task(actor_main.kill(actor_sub.address, 'Test-sub'))

    with pytest.raises(ServerClosed):
        await task


@require_ray
@pytest.mark.asyncio
@pytest.mark.parametrize(
    'auto_recover',
    [False, True, 'actor', 'process']
)
async def test_auto_recover(ray_start_regular, auto_recover):
    pg_name, n_process = 'ray_cluster', 1
    pg = ray.util.placement_group(name=pg_name, bundles=[{'CPU': n_process}])
    assert pg.wait(timeout_seconds=20)
    address = process_placement_to_address(pg_name, 0, process_index=0)
    actor_handle = await mo.create_actor_pool(address, n_process=n_process, auto_recover=auto_recover)
    await actor_handle.actor_pool.remote('start')

    ctx = get_context()

    # wait for recover of main pool always returned immediately
    await ctx.wait_actor_pool_recovered(address, address)

    # create actor on main
    actor_ref = await ctx.create_actor(
        TestActor, address=address,
        allocate_strategy=MainPool())

    with pytest.raises(ValueError):
        # cannot kill actors on main pool
        await mo.kill_actor(actor_ref)

    # create actor
    actor_ref = await ctx.create_actor(
        TestActor, address=address,
        allocate_strategy=ProcessIndex(1))
    # kill_actor will cause kill corresponding process
    await ctx.kill_actor(actor_ref)

    if auto_recover:
        await ctx.wait_actor_pool_recovered(actor_ref.address, address)
        sub_pool_address = process_placement_to_address(pg_name, 0, process_index=1)
        sub_pool_handle = ray.get_actor(sub_pool_address)
        assert await sub_pool_handle.actor_pool.remote('health_check') == PoolStatus.HEALTHY

        expect_has_actor = True if auto_recover in ['actor', True] else False
        assert await ctx.has_actor(actor_ref) is expect_has_actor
    else:
        with pytest.raises((ServerClosed, ConnectionError)):
            await ctx.has_actor(actor_ref)

    if 'COV_CORE_SOURCE' in os.environ:
        for addr in [process_placement_to_address(pg_name, 0, process_index=i) for i in range(2)]:
            # must save the local reference until this is fixed:
            # https://github.com/ray-project/ray/issues/7815
            ray_actor = ray.get_actor(addr)
            ray.get(ray_actor.cleanup.remote())
