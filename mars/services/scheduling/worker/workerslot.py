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

import asyncio
import time
from collections import namedtuple
from typing import Optional, Tuple

import psutil

from .... import oscar as mo
from ....oscar.backends.allocate_strategy import IdleLabel

DispatchDumpType = namedtuple('DispatchDumpType', 'free_slots')


class WorkerSlotManagerActor(mo.Actor):
    def __init__(self):
        self._cluster_api = None
        self._global_slots_ref = None

    async def __post_create__(self):
        from ...cluster.api import ClusterAPI
        from ..supervisor import GlobalSlotManagerActor

        self._cluster_api = await ClusterAPI.create(self.address)
        [self._global_slots_ref] = await self._cluster_api.get_supervisor_refs([
            GlobalSlotManagerActor.default_uid()])

        band_to_slots = await self._cluster_api.get_bands()
        for band, n_slot in band_to_slots.items():
            await mo.create_actor(
                BandSlotManagerActor, band[1], n_slot, self._global_slots_ref,
                uid=BandSlotManagerActor.gen_uid(band[1]),
                address=self.address)


class BandSlotManagerActor(mo.Actor):
    @classmethod
    def gen_uid(cls, band_name: str):
        return f'{band_name}_band_slot_manager'

    def __init__(self, band_name: str, n_slots: int,
                 global_slots_ref: Optional[mo.ActorRef] = None):
        super().__init__()
        self._band_name = band_name
        self._global_slots_ref = global_slots_ref
        self._n_slots = n_slots

        self._semaphore = asyncio.Semaphore(0)
        self._slot_control_refs = dict()
        self._free_slots = set()
        self._slot_kill_events = dict()

        self._slot_to_session_stid = dict()
        self._slot_to_usage = dict()
        self._last_report_time = time.time()

    async def __post_create__(self):
        strategy = IdleLabel(self._band_name, 'worker_slot_control')
        for slot_id in range(self._n_slots):
            self._slot_control_refs[slot_id] = await mo.create_actor(
                BandSlotControlActor,
                self.ref(), self._band_name, slot_id,
                uid=BandSlotControlActor.gen_uid(self._band_name, slot_id),
                address=self.address,
                allocate_strategy=strategy)

    async def acquire_free_slot(self, session_stid: Tuple[str, str]):
        yield self._semaphore.acquire()
        slot_id = self._free_slots.pop()
        self._slot_to_session_stid[slot_id] = session_stid
        raise mo.Return(slot_id)

    def release_free_slot(self, slot_id):
        if slot_id in self._slot_kill_events:
            event = self._slot_kill_events.pop(slot_id)
            event.set()
        self._slot_to_session_stid.pop(slot_id, None)
        self._free_slots.add(slot_id)
        self._semaphore.release()

    async def kill_slot(self, slot_id):
        event = self._slot_kill_events[slot_id] = asyncio.Event()
        await mo.kill_actor(self._slot_control_refs[slot_id])
        return event.wait()

    async def set_slot_usage(self, slot_id: int, usage: float):
        self._slot_to_usage[slot_id] = usage
        if self._global_slots_ref is None:
            return

        if time.time() - self._last_report_time >= 1.0:
            self._last_report_time = time.time()
            delays = []
            for slot_id, (session_id, subtask_id) in self._slot_to_session_stid.items():
                if slot_id not in self._slot_to_usage:
                    continue
                delays.append(self._global_slots_ref.update_subtask_slots.delay(
                    self._band_name, session_id, subtask_id,
                    max(1.0, self._slot_to_usage[slot_id])))
            return self._global_slots_ref.update_subtask_slots.batch(*delays)

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
        await self._manager_ref.release_free_slot.tell(self._slot_id)

        async def report_usage():
            proc = psutil.Process()
            proc.cpu_percent()
            while True:
                await asyncio.sleep(1)
                await self._manager_ref.set_slot_usage.tell(
                    self._slot_id, proc.cpu_percent())

        self._report_task = asyncio.create_task(report_usage())

    async def __pre_destroy__(self):
        if self._report_task:  # pragma: no branch
            self._report_task.cancel()
