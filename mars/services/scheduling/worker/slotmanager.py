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

from typing import Dict, Tuple

from .... import oscar as mo
from ....oscar.backends.allocate_strategy import IdleLabel
from ...cluster import ClusterAPI


class SlotManagerActor(mo.Actor):
    _slot_to_ref: Dict[Tuple[str, int], mo.ActorRef]

    async def __post_create__(self):
        cluster_api = await ClusterAPI.create(self.address)
        bands = await cluster_api.get_bands()

        self._slot_to_ref = dict()
        for band, slot_num in bands.items():
            band_name = band[1]
            for slot_id in range(slot_num):
                self._slot_to_ref[(band_name, slot_id)] = await mo.create_actor(
                    SlotControlActor,
                    uid=SlotControlActor.gen_uid(band_name, slot_id),
                    allocate_strategy=IdleLabel(band_name, "slot_control"),
                    address=self.address,
                )

    async def get_slot_address(self, band_name: str, slot_id: int) -> str:
        return self._slot_to_ref[(band_name, slot_id)].address

    async def kill_slot(self, band_name: str, slot_id: int):
        ref = self._slot_to_ref[(band_name, slot_id)]
        await mo.kill_actor(ref)
        await mo.wait_actor_pool_recovered(ref.address, self.address)


class SlotControlActor(mo.Actor):
    @classmethod
    def gen_uid(cls, band_name: str, slot_id: int):
        return f"{band_name}_{slot_id}slot_control"
