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
import logging
from typing import Dict

from ....utils import lazy_import, Timer
from ...backend import BaseActorBackend, register_backend
from ..context import MarsActorContext
from .driver import RayActorDriver
from .pool import RayMainPool
from .utils import process_address_to_placement, get_placement_group

ray = lazy_import("ray")

__all__ = ["RayActorBackend"]

logger = logging.getLogger(__name__)


@register_backend
class RayActorBackend(BaseActorBackend):
    @staticmethod
    def name():
        return "ray"

    @staticmethod
    def get_context_cls():
        return MarsActorContext

    @staticmethod
    def get_driver_cls():
        return RayActorDriver

    @classmethod
    async def _create_ray_pools(cls, address: str, n_process: int = None, **kwargs):
        # pop `n_io_process` from kwargs as ray doesn't need this
        kwargs.pop("n_io_process", 0)
        pg_name, bundle_index, _ = process_address_to_placement(address)
        from .pool import RayMainActorPool

        pool_addresses = RayMainActorPool.get_external_addresses(address, n_process)
        assert pool_addresses[0] == address
        pg = get_placement_group(pg_name) if pg_name else None
        num_cpus = kwargs.get("main_pool_cpus", 0)
        sub_pools = {
            sub_pool_address: RayMainActorPool.create_sub_pool(
                address, sub_pool_address
            )
            for sub_pool_address in pool_addresses[1:]
        }
        actor_handle = (
            ray.remote(RayMainPool)
            .options(
                num_cpus=num_cpus,
                name=address,
                max_concurrency=10000000,  # By default, 1000 tasks can be running concurrently.
                max_restarts=-1,  # Auto restarts by ray
                placement_group=pg,
                placement_group_bundle_index=bundle_index,
                placement_group_capture_child_tasks=False,
            )
            .remote(address, n_process, sub_pools, **kwargs)
        )
        pool_handle = RayPoolHandle(actor_handle, sub_pools)
        return pool_handle

    @classmethod
    async def create_actor_pool(cls, address: str, n_process: int = None, **kwargs):
        with Timer() as timer:
            pool_handle = await cls._create_ray_pools(address, n_process, **kwargs)
        logger.info(
            "Submit create actor pool %s took %s seconds.",
            pool_handle.main_pool,
            timer.duration,
        )
        with Timer() as timer:
            await pool_handle.main_pool.start.remote()
        logger.info(
            "Start actor pool %s took %s seconds.",
            pool_handle.main_pool,
            timer.duration,
        )
        return pool_handle


class RayPoolHandle:
    def __init__(
        self,
        main_pool: "ray.actor.ActorHandle",
        sub_pools: Dict[str, "ray.actor.ActorHandle"],
    ):
        self.main_pool = main_pool
        # Hold sub_pool actor handles to avoid gc.
        self.sub_pools = sub_pools

    def __getattr__(self, item):
        if item in ("main_pool", "sub_pools"):  # pragma: no cover
            return object.__getattribute__(self, item)
        return getattr(self.main_pool, item)
