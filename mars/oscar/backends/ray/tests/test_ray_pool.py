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

from ..... import oscar as mo
from .....tests.core import require_ray, mock
from .....utils import lazy_import
from ....context import get_context
from ....errors import ServerClosed
from ...allocate_strategy import ProcessIndex, MainPool
from ..backend import RayActorBackend
from ..pool import RayMainActorPool, create_actor_pool, RayPoolState
from ..utils import process_placement_to_address, kill_and_wait

ray = lazy_import("ray")


class TestActor(mo.Actor):
    __test__ = False

    async def kill(self, address, uid):
        actor_ref = await mo.actor_ref(address, uid)
        task = asyncio.create_task(actor_ref.crash())
        return await task

    async def crash(self):
        os._exit(0)


@require_ray
@pytest.mark.asyncio
async def test_main_pool(ray_start_regular):
    pg, pg_name, n_process = None, "ray_cluster", 3
    if hasattr(ray.util, "get_placement_group"):
        pg = ray.util.placement_group(name=pg_name, bundles=[{"CPU": n_process}])
        ray.get(pg.ready())
    address = process_placement_to_address(pg_name, 0, process_index=0)
    addresses = RayMainActorPool.get_external_addresses(address, n_process)
    assert addresses == [address] + [
        process_placement_to_address(pg_name, 0, process_index=i + 1)
        for i in range(n_process)
    ]
    assert RayMainActorPool.gen_internal_address(0, address) == address

    pool_handle = await RayActorBackend._create_ray_pools(address, n_process)
    main_actor_pool = await create_actor_pool(
        address,
        n_process=n_process,
        pool_cls=RayMainActorPool,
        sub_pool_handles=pool_handle.sub_pools,
    )
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

    pg_name, n_process = "ray_cluster", 2
    if hasattr(ray.util, "get_placement_group"):
        pg = ray.util.placement_group(name=pg_name, bundles=[{"CPU": n_process}])
        ray.get(pg.ready())
    else:
        pg = None
    address = process_placement_to_address(pg_name, 0, process_index=0)
    pool_handle = await RayActorBackend._create_ray_pools(address, n_process)
    actor_handle = pool_handle.main_pool
    await actor_handle.start.remote()
    sub_pool_address1 = process_placement_to_address(pg_name, 0, process_index=1)
    sub_pool_handle1 = ray.get_actor(sub_pool_address1)
    sub_pool_address2 = process_placement_to_address(pg_name, 0, process_index=2)
    sub_pool_handle2 = ray.get_actor(sub_pool_address2)
    await actor_handle.actor_pool.remote(
        "stop_sub_pool", sub_pool_address1, sub_pool_handle1, force=True
    )
    await actor_handle.actor_pool.remote(
        "stop_sub_pool", sub_pool_address2, sub_pool_handle2, force=False
    )
    assert await sub_pool_handle1.state.remote() == RayPoolState.INIT
    assert await sub_pool_handle2.state.remote() == RayPoolState.INIT


@require_ray
@pytest.mark.asyncio
async def test_server_closed(ray_start_regular):
    pg_name, n_process = "ray_cluster", 1
    pg = ray.util.placement_group(name=pg_name, bundles=[{"CPU": n_process}])
    ray.get(pg.ready())
    address = process_placement_to_address(pg_name, 0, process_index=0)
    # start the actor pool
    actor_handle = await mo.create_actor_pool(address, n_process=n_process)
    await actor_handle.mark_service_ready.remote()

    ctx = get_context()
    actor_main = await ctx.create_actor(
        TestActor, address=address, uid="Test-main", allocate_strategy=ProcessIndex(0)
    )

    actor_sub = await ctx.create_actor(
        TestActor, address=address, uid="Test-sub", allocate_strategy=ProcessIndex(1)
    )

    # test calling from ray driver to ray actor
    task = asyncio.create_task(actor_sub.crash())

    with pytest.raises(ServerClosed):
        # process already died,
        # ServerClosed will be raised
        await task

    # wait for recover of sub pool
    await ctx.wait_actor_pool_recovered(actor_sub.address, address)

    # test calling from ray actor to ray actor
    task = asyncio.create_task(actor_main.kill(actor_sub.address, "Test-sub"))

    with pytest.raises(ServerClosed):
        await task


@require_ray
@pytest.mark.asyncio
@pytest.mark.parametrize("auto_recover", [False, True, "actor", "process"])
async def test_auto_recover(ray_start_regular, auto_recover):
    pg_name, n_process = "ray_cluster", 1
    pg = ray.util.placement_group(name=pg_name, bundles=[{"CPU": n_process}])
    assert pg.wait(timeout_seconds=20)
    address = process_placement_to_address(pg_name, 0, process_index=0)
    actor_handle = await mo.create_actor_pool(
        address, n_process=n_process, auto_recover=auto_recover
    )
    await actor_handle.mark_service_ready.remote()

    ctx = get_context()

    # wait for recover of main pool always returned immediately
    await ctx.wait_actor_pool_recovered(address, address)

    # create actor on main
    actor_ref = await ctx.create_actor(
        TestActor, address=address, allocate_strategy=MainPool()
    )

    with pytest.raises(ValueError):
        # cannot kill actors on main pool
        await mo.kill_actor(actor_ref)

    # create actor
    actor_ref = await ctx.create_actor(
        TestActor, address=address, allocate_strategy=ProcessIndex(1)
    )
    # kill_actor will cause kill corresponding process
    await ctx.kill_actor(actor_ref)

    if auto_recover:
        await ctx.wait_actor_pool_recovered(actor_ref.address, address)
        sub_pool_address = process_placement_to_address(pg_name, 0, process_index=1)
        sub_pool_handle = ray.get_actor(sub_pool_address)
        if auto_recover == "process":
            assert await sub_pool_handle.state.remote() == RayPoolState.POOL_READY
        else:
            assert await sub_pool_handle.state.remote() == RayPoolState.SERVICE_READY

        expect_has_actor = True if auto_recover in ["actor", True] else False
        assert await ctx.has_actor(actor_ref) is expect_has_actor
    else:
        with pytest.raises((ServerClosed, ConnectionError)):
            await ctx.has_actor(actor_ref)

    if "COV_CORE_SOURCE" in os.environ:
        for addr in [
            process_placement_to_address(pg_name, 0, process_index=i) for i in range(2)
        ]:
            # must save the local reference until this is fixed:
            # https://github.com/ray-project/ray/issues/7815
            ray_actor = ray.get_actor(addr)
            ray.get(ray_actor.cleanup.remote())


@require_ray
@pytest.mark.asyncio
@mock.patch("ray.kill")
async def test_kill_and_wait_timeout(fake_ray_kill, ray_start_regular):
    pg_name, n_process = "ray_cluster", 1
    pg = ray.util.placement_group(name=pg_name, bundles=[{"CPU": n_process}])
    ray.get(pg.ready())
    address = process_placement_to_address(pg_name, 0, process_index=0)
    # start the actor pool
    actor_handle = await mo.create_actor_pool(address, n_process=n_process)
    with pytest.raises(Exception, match="not died"):
        await kill_and_wait(actor_handle, timeout=1)
