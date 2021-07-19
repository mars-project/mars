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
from typing import Type

from .... import oscar as mo
from ....oscar.backends.allocate_strategy import IdleLabel
from .runner import SubtaskRunnerActor


class SubtaskManagerActor(mo.Actor):
    def __init__(self, subtask_processor_cls: Type):
        # specify subtask process class
        # for test purpose
        self._subtask_processor_cls = subtask_processor_cls
        self._cluster_api = None

        self._band_slot_runner_refs = dict()

    async def __post_create__(self):
        from ...cluster.api import ClusterAPI
        self._cluster_api = await ClusterAPI.create(self.address)

        band_to_slots = await self._cluster_api.get_bands()
        for band, n_slot in band_to_slots.items():
            await self._create_band_runner_actors(band[1], n_slot)

    async def _create_band_runner_actors(self, band_name: str, n_slots: int):
        strategy = IdleLabel(band_name, 'subtask_runner')
        band = (self.address, band_name)
        for slot_id in range(n_slots):
            self._band_slot_runner_refs[(band_name, slot_id)] = await mo.create_actor(
                SubtaskRunnerActor,
                band,
                subtask_processor_cls=self._subtask_processor_cls,
                uid=SubtaskRunnerActor.gen_uid(band_name, slot_id),
                address=self.address,
                allocate_strategy=strategy)

    async def __pre_destroy__(self):
        await asyncio.gather(*[
            mo.destroy_actor(ref) for ref in self._band_slot_runner_refs.values()
        ])
