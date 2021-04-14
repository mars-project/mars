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

from ....utils import lazy_import
from ...backend import BaseActorBackend, register_backend
from ..context import MarsActorContext
from .driver import RayActorDriver
from .pool import RayMainPool
from .utils import process_placement_to_address, node_address_to_placement, get_placement_group

ray = lazy_import("ray")

__all__ = ['RayActorBackend']


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

    @staticmethod
    async def create_actor_pool(
        address: str,
        n_process: int = None,
        **kwargs
    ):
        pg_name, bundle_index = node_address_to_placement(address)
        address = process_placement_to_address(pg_name, bundle_index, process_index=0)
        pg = get_placement_group(pg_name) if pg_name else None
        if not pg:
            bundle_index = -1
        actor_handle = ray.remote(RayMainPool).options(
            num_cpus=0,  # main pool doesn't do horse work, mark it use no cpu.
            name=address, placement_group=pg, placement_group_bundle_index=bundle_index).remote()
        ray.get(actor_handle.start.remote(address, n_process, **kwargs))
        return actor_handle
