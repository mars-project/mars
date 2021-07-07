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
import time
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import psutil

from .... import oscar as mo
from ....oscar.backends.allocate_strategy import IdleLabel
from ...core import BandType
from ..core import WorkerSlotInfo

DispatchDumpType = namedtuple('DispatchDumpType', 'free_slots')


class WorkerSlotManagerActor(mo.Actor):
    _band_slot_infos: Dict[str, List[WorkerSlotInfo]]

    def __init__(self):
        self._cluster_api = None
        self._global_slots_ref = None

        self._band_slot_managers = dict()  # type: Dict[str, mo.ActorRef]

    async def __post_create__(self):
        from ...cluster.api import ClusterAPI
        from ..supervisor import GlobalSlotManagerActor

        self._cluster_api = await ClusterAPI.create(self.address)
        [self._global_slots_ref] = await self._cluster_api.get_supervisor_refs([
            GlobalSlotManagerActor.default_uid()])

        band_to_slots = await self._cluster_api.get_bands()
        for band, n_slot in band_to_slots.items():
            self._band_slot_managers[band] = await mo.create_actor(
                BandSlotManagerActor, band, n_slot, self._global_slots_ref,
                uid=BandSlotManagerActor.gen_uid(band[1]),
                address=self.address)

    async def __pre_destroy__(self):
        await asyncio.gather(*[mo.destroy_actor(ref)
                               for ref in self._band_slot_managers.values()])


class BandSlotManagerActor(mo.Actor):
    @classmethod
    def gen_uid(cls, band_name: str):
        return f'{band_name}_band_slot_manager'

    def __init__(self, band: BandType, n_slots: int,
                 global_slots_ref: mo.ActorRef = None):
        super().__init__()
        self._cluster_api = None

        self._band = band
        self._band_name = band[1]
        self._global_slots_ref = global_slots_ref
        self._n_slots = n_slots

        self._semaphore = asyncio.Semaphore(0)
        self._slot_control_refs = dict()
        self._free_slots = set()
        self._slot_kill_events = dict()

        self._slot_to_session_stid = dict()
        self._last_report_time = time.time()

        self._slot_to_proc = dict()
        self._usage_upload_task = None

    async def __post_create__(self):
        from ...cluster.api import ClusterAPI
        try:
            self._cluster_api = await ClusterAPI.create(self.address)
        except mo.ActorNotExist:
            pass

        strategy = IdleLabel(self._band_name, 'worker_slot_control')
        for slot_id in range(self._n_slots):
            self._slot_control_refs[slot_id] = await mo.create_actor(
                BandSlotControlActor,
                self.ref(), self._band_name, slot_id,
                uid=BandSlotControlActor.gen_uid(self._band_name, slot_id),
                address=self.address,
                allocate_strategy=strategy)

        self._usage_upload_task = self.ref().upload_slot_usages.tell_delay(
            periodical=True, delay=1)

    async def __pre_destroy__(self):
        self._usage_upload_task.cancel()

    async def acquire_free_slot(self, session_stid: Tuple[str, str]):
        yield self._semaphore.acquire()
        slot_id = self._free_slots.pop()
        self._slot_to_session_stid[slot_id] = session_stid
        raise mo.Return(slot_id)

    def release_free_slot(self, slot_id: int, pid: Optional[int] = None):
        if pid is not None:
            self._slot_to_proc[slot_id] = proc = psutil.Process(pid)
            # collect initial stats for the process
            proc.cpu_percent(interval=None)

        if slot_id in self._slot_kill_events:
            event = self._slot_kill_events.pop(slot_id)
            event.set()

        self._slot_to_session_stid.pop(slot_id, None)

        if slot_id not in self._free_slots:
            self._free_slots.add(slot_id)
            self._semaphore.release()

    async def kill_slot(self, slot_id):
        assert slot_id not in self._slot_kill_events
        event = self._slot_kill_events[slot_id] = asyncio.Event()
        yield mo.kill_actor(self._slot_control_refs[slot_id])
        yield event.wait()

    async def upload_slot_usages(self, periodical: bool = False):
        delays = []
        slot_infos = []
        for slot_id, proc in self._slot_to_proc.items():
            if slot_id not in self._slot_to_session_stid:
                continue
            session_id, subtask_id = self._slot_to_session_stid[slot_id]

            try:
                usage = proc.cpu_percent(interval=None) / 100.0
            except psutil.NoSuchProcess:  # pragma: no cover
                continue

            slot_infos.append(WorkerSlotInfo(
                slot_id=slot_id,
                session_id=session_id,
                subtask_id=subtask_id,
                processor_usage=usage
            ))

            if self._global_slots_ref is not None:  # pragma: no branch
                # FIXME fix band slot mistake
                delays.append(self._global_slots_ref.update_subtask_slots.delay(
                    self._band[1], session_id, subtask_id, max(1.0, usage)))

        if delays:  # pragma: no branch
            yield self._global_slots_ref.update_subtask_slots.batch(*delays)
        if self._cluster_api is not None:
            await self._cluster_api.set_band_slot_infos(self._band_name, slot_infos)

        if periodical:
            self._usage_upload_task = self.ref().upload_slot_usages.tell_delay(
                periodical=True, delay=1)

    def dump_data(self):
        """
        Get all refs of slots of a queue
        """
        return DispatchDumpType(self._free_slots)


class BandSlotControlActor(mo.Actor):
    @classmethod
    def gen_uid(cls, band_name: str, slot_id: int):
        return f'{band_name}_{slot_id}_band_slot_control'

    def __init__(self, manager_ref, band_name, slot_id: int):
        self._manager_ref = manager_ref
        self._band_name = band_name
        self._slot_id = slot_id
        self._report_task = None

    async def __post_create__(self):
        await self._manager_ref.release_free_slot.tell(self._slot_id, os.getpid())
