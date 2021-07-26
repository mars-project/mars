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
import importlib
import logging
from typing import Dict, Optional, Type, Union

from .... import oscar as mo
from ....lib.aio import alru_cache
from ....typing import BandType
from ...cluster import ClusterAPI
from ..core import Subtask, SubtaskResult
from ..errors import SlotOccupiedAlready
from .processor import SubtaskProcessor, SubtaskProcessorActor

logger = logging.getLogger(__name__)


SubtaskRunnerRef = Union["SubtaskRunnerActor", mo.ActorRef]


class SubtaskRunnerActor(mo.Actor):
    _session_id_to_processors: Dict[str, Union[mo.ActorRef, SubtaskProcessorActor]]
    _running_processor: Optional[Union[mo.ActorRef, SubtaskProcessorActor]]
    _last_processor: Optional[Union[mo.ActorRef, SubtaskProcessorActor]]

    @classmethod
    def gen_uid(cls, band_name: str, slot_id: int):
        return f'slot_{band_name}_{slot_id}_subtask_runner'

    def __init__(self,
                 band: BandType,
                 subtask_processor_cls: Type = None):
        self._band = band
        self._subtask_processor_cls = \
            self._get_subtask_processor_cls(subtask_processor_cls)

        self._cluster_api = None

        self._session_id_to_processors = dict()
        self._running_processor = None
        self._last_processor = None

    async def __post_create__(self):
        self._cluster_api = await ClusterAPI.create(address=self.address)

    async def __pre_destroy__(self):
        await asyncio.gather(*[
            mo.destroy_actor(ref) for ref in self._session_id_to_processors.values()
        ])

    @classmethod
    def _get_subtask_processor_cls(cls, subtask_processor_cls):
        if subtask_processor_cls is None:
            return SubtaskProcessor
        else:
            assert isinstance(subtask_processor_cls, str)
            module, class_name = subtask_processor_cls.rsplit('.', 1)
            return getattr(importlib.import_module(module), class_name)

    async def _run_subtask(self, subtask: Subtask):
        processor = await self._init_subtask_processor(subtask)
        self._subtask_info.processor = processor
        return await processor.run()

    @alru_cache(cache_exceptions=False)
    async def _get_supervisor_address(self, session_id: str):
        [address] = await self._cluster_api.get_supervisors_by_keys([session_id])
        return address

    async def run_subtask(self, subtask: Subtask):
        if self._running_processor is not None:  # pragma: no cover
            running_subtask_id = await self._running_processor.get_running_subtask_id()
            # current subtask is still running
            raise SlotOccupiedAlready(
                f'There is subtask(id: {running_subtask_id}) running in {self.uid} '
                f'at {self.address}, cannot run subtask {subtask.subtask_id}')

        session_id = subtask.session_id
        supervisor_address = await self._get_supervisor_address(session_id)
        if session_id not in self._session_id_to_processors:
            self._session_id_to_processors[session_id] = await mo.create_actor(
                SubtaskProcessorActor, session_id, self._band,
                supervisor_address, self._subtask_processor_cls,
                uid=SubtaskProcessorActor.gen_uid(session_id),
                address=self.address)
        processor = self._session_id_to_processors[session_id]
        self._running_processor = self._last_processor = processor
        try:
            result = yield self._running_processor.run(subtask)
        finally:
            self._running_processor = None
        raise mo.Return(result)

    async def get_subtask_result(self) -> SubtaskResult:
        return self._last_processor.result()

    def is_runner_free(self):
        return self._running_processor is None

    async def cancel_subtask(self):
        if self._running_processor is None:
            return
        yield self._running_processor.cancel()
