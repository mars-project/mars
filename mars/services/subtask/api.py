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

from ... import oscar as mo
from ...lib.aio import alru_cache
from .core import Subtask


class SubtaskAPI:
    def __init__(self, address: str):
        self._address = address

    @classmethod
    async def create(cls, address: str) -> "SubtaskAPI":
        return SubtaskAPI(address)

    @alru_cache(cache_exceptions=False)
    async def _get_runner_ref(self, band_name: str, slot_id: int):
        from .worker.runner import SubtaskRunnerActor
        return await mo.actor_ref(
            SubtaskRunnerActor.gen_uid(band_name, slot_id), address=self._address)

    @alru_cache(cache_exceptions=False)
    async def _get_subtask_processor_ref(self, session_id: str,
                                         slot_address: str):
        from .worker.processor import SubtaskProcessorActor
        return await mo.actor_ref(SubtaskProcessorActor.gen_uid(session_id),
                                  address=slot_address)

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
        """
        Cancel subtask running in a worker slot and wait until it is cancelled

        Parameters
        ----------
        band_name : str
            name of a worker band, for instance, 'numa-0'
        slot_id : int
            index of a slot in a band
        """
        ref = await self._get_runner_ref(band_name, slot_id)
        await ref.cancel_subtask()

    async def set_running_operand_progress(self, session_id: str, op_key: str,
                                           slot_address: str, progress: float):
        ref = await self._get_subtask_processor_ref(session_id, slot_address)
        await ref.set_running_op_progress(op_key, progress)


class MockSubtaskAPI(SubtaskAPI):
    @classmethod
    async def create(cls, address: str) -> "SubtaskAPI":
        from .worker.manager import SubtaskManagerActor
        await mo.create_actor(
            SubtaskManagerActor, None,
            uid=SubtaskManagerActor.default_uid(),
            address=address)
        return await super().create(address)
