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

from ... import oscar as mo
from ...lib.aio import alru_cache
from .core import Subtask


class SubtaskAPI:
    def __init__(self, address: str):
        self._address = address

    @classmethod
    async def create(cls, address: str) -> "SubtaskAPI":
        return SubtaskAPI(address)

    @alru_cache
    async def _get_runner_ref(self, band_name: str, slot_id: int):
        from .worker.subtask import SubtaskRunnerActor
        return await mo.actor_ref(
            SubtaskRunnerActor.gen_uid(band_name, slot_id), address=self._address)

    async def run_subtask_in_slot(self,
                                  band_name: str,
                                  slot_id: int,
                                  subtask: Subtask):
        """
        Run subtask in current worker

        Parameters
        ----------
        band_name
        subtask
        slot_id

        Returns
        -------

        """
        ref = await self._get_runner_ref(band_name, slot_id)
        return await ref.run_subtask(subtask)

    async def cancel_subtask_in_slot(self, band_name: str, slot_id: int):
        ref = await self._get_runner_ref(band_name, slot_id)
        await ref.cancel_subtask()


class MockSubtaskAPI(SubtaskAPI):
    @classmethod
    async def create(cls, address: str) -> "SubtaskAPI":
        from .worker.subtask import SubtaskManagerActor
        await mo.create_actor(
            SubtaskManagerActor, None,
            uid=SubtaskManagerActor.default_uid(),
            address=address)
        return await super().create(address)
