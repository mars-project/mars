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
import logging
import os
import time
from typing import Dict, List, NamedTuple, Set, Tuple

import psutil

from .... import oscar as mo
from ....oscar.errors import NoFreeSlot, SlotStateError
from ....oscar.backends.allocate_strategy import IdleLabel
from ....resource import Resource
from ....typing import BandType
from ...cluster import WorkerSlotInfo, ClusterAPI

logger = logging.getLogger(__name__)


class DispatchDumpType(NamedTuple):
    free_slots: Set
    fresh_slots: Set


class WorkerSlotManagerActor(mo.Actor):
    _band_slot_infos: Dict[str, List[WorkerSlotInfo]]

    def __init__(self):
        self._cluster_api = None
        self._global_resource_ref = None

        self._band_slot_managers = dict()  # type: Dict[str, mo.ActorRef]

    async def __post_create__(self):
        self._cluster_api = await ClusterAPI.create(self.address)

        band_to_slots = await self._cluster_api.get_bands()
        for band, n_slot in band_to_slots.items():
            self._band_slot_managers[band] = await mo.create_actor(
                BandSlotManagerActor,
                band,
                n_slot,
                self._global_resource_ref,
                uid=BandSlotManagerActor.gen_uid(band[1]),
                address=self.address,
            )

    async def __pre_destroy__(self):
        await asyncio.gather(
            *[mo.destroy_actor(ref) for ref in self._band_slot_managers.values()]
        )


class BandSlotManagerActor(mo.Actor):
    _free_slots: Set[int]
    _fresh_slots: Set[int]

    @classmethod
    def gen_uid(cls, band_name: str):
        return f"{band_name}_band_slot_manager"

    def __init__(
        self, band: BandType, n_slots: int, global_resource_ref: mo.ActorRef = None
    ):
        super().__init__()
        self._cluster_api = None

        self._band = band
        self._band_name = band[1]
        self._global_resource_ref = global_resource_ref
        self._n_slots = n_slots

        self._semaphore = asyncio.Semaphore(0)
        self._slot_control_refs = dict()
        self._free_slots = set()
        self._fresh_slots = set()
        self._slot_kill_events = dict()

        self._restarting = False
        self._restart_done_event = asyncio.Event()

        self._session_stid_to_slot = dict()
        self._slot_to_session_stid = dict()
        self._last_report_time = time.time()

        self._slot_to_proc = dict()
        self._usage_upload_task = None

    async def __post_create__(self):
        try:
            self._cluster_api = await ClusterAPI.create(self.address)
        except mo.ActorNotExist:
            pass

        strategy = IdleLabel(self._band_name, f"worker_slot_control")
        for slot_id in range(self._n_slots):
            self._slot_control_refs[slot_id] = await mo.create_actor(
                BandSlotControlActor,
                self.ref(),
                self._band_name,
                slot_id,
                uid=BandSlotControlActor.gen_uid(self._band_name, slot_id),
                address=self.address,
                allocate_strategy=strategy,
            )
            self._fresh_slots.add(slot_id)

        self._usage_upload_task = self.ref().upload_slot_usages.tell_delay(
            periodical=True, delay=1
        )

    async def __pre_destroy__(self):
        self._usage_upload_task.cancel()

    async def _get_global_resource_ref(self):
        if self._global_resource_ref is not None:
            return self._global_resource_ref

        from ..supervisor import GlobalResourceManagerActor

        [self._global_resource_ref] = await self._cluster_api.get_supervisor_refs(
            [GlobalResourceManagerActor.default_uid()]
        )
        return self._global_resource_ref

    def get_slot_address(self, slot_id: int):
        return self._slot_control_refs[slot_id].address

    def get_subtask_slot(self, session_stid: Tuple[str, str]):
        return self._session_stid_to_slot.get(session_stid)

    async def acquire_free_slot(self, session_stid: Tuple[str, str], block=True):
        if not block and self._semaphore.locked():
            raise NoFreeSlot(f"No free slot for {session_stid}")
        yield self._semaphore.acquire()
        if self._restarting:
            yield self._restart_done_event.wait()

        slot_id = self._free_slots.pop()
        self._fresh_slots.difference_update([slot_id])
        self._slot_to_session_stid[slot_id] = session_stid
        self._session_stid_to_slot[session_stid] = slot_id
        logger.debug("Slot %d acquired for subtask %r", slot_id, session_stid)
        raise mo.Return(slot_id)

    def release_free_slot(self, slot_id: int, session_stid: Tuple[str, str]):
        acquired_session_stid = self._slot_to_session_stid.pop(slot_id, None)
        if acquired_session_stid is None:
            raise SlotStateError(f"Slot {slot_id} is not acquired.")
        if acquired_session_stid != session_stid:
            raise SlotStateError(
                f"Slot {slot_id} releasing state incorrect, "
                f"the acquired session_stid: {acquired_session_stid}, "
                f"the releasing session_stid: {session_stid}"
            )
        acquired_slot_id = self._session_stid_to_slot.pop(acquired_session_stid)
        assert (
            acquired_slot_id == slot_id
        ), f"{acquired_session_stid}: acquired_slot_id {acquired_slot_id} != slot_id {slot_id}"

        logger.debug("Slot %d released", slot_id)

        if slot_id not in self._free_slots:
            self._free_slots.add(slot_id)
            self._semaphore.release()

    def register_slot(self, slot_id: int, pid: int):
        try:
            self._fresh_slots.add(slot_id)
            if slot_id in self._slot_kill_events:
                event = self._slot_kill_events.pop(slot_id)
                event.set()
            if slot_id in self._slot_to_session_stid:
                # We should release the slot by one role, if the slot is
                # acquired by the SubtaskExecutionActor, then the slot
                # should be released by it, too.
                session_stid = self._slot_to_session_stid[slot_id]
                logger.info(
                    "Slot %s registered by pid %s, current acquired session_stid is %s",
                    slot_id,
                    pid,
                    session_stid,
                )
            else:
                if slot_id not in self._free_slots:
                    self._free_slots.add(slot_id)
                    self._semaphore.release()
        finally:
            # psutil may raises exceptions, but currently we can't handle the register exception,
            # so put it to the finally.
            # TODO(fyrestone): handle register_slot failure.
            self._slot_to_proc[slot_id] = proc = psutil.Process(pid)
            # collect initial stats for the process
            proc.cpu_percent(interval=None)

    async def _kill_slot(self, slot_id: int):
        if slot_id in self._slot_kill_events:
            await self._slot_kill_events[slot_id].wait()
            return

        event = self._slot_kill_events[slot_id] = asyncio.Event()
        # TODO(fyrestone): Make it more reliable. e.g. kill_actor
        # success but the actor does not restart.
        try:
            await mo.kill_actor(self._slot_control_refs[slot_id])
        except ConnectionError:
            pass
        await event.wait()

    async def kill_slot(self, slot_id: int):
        self._free_slots.difference_update([slot_id])
        yield self._kill_slot(slot_id)

    async def restart_free_slots(self):
        if self._restarting:
            yield self._restart_done_event.wait()
            return

        self._restart_done_event = asyncio.Event()
        self._restarting = True
        slot_ids = [
            slot_id for slot_id in self._free_slots if slot_id not in self._fresh_slots
        ]
        if slot_ids:
            yield asyncio.gather(*[self._kill_slot(slot_id) for slot_id in slot_ids])
            logger.info("%d idle slots restarted", len(slot_ids))

        self._restarting = False
        self._restart_done_event.set()

    async def upload_slot_usages(self, periodical: bool = False):
        delays = []
        slot_infos = []
        global_resource_ref = await self._get_global_resource_ref()
        for slot_id, proc in self._slot_to_proc.items():
            if slot_id not in self._slot_to_session_stid:
                continue
            session_id, subtask_id = self._slot_to_session_stid[slot_id]
            cpu_usage, gpu_usage, processor_usage = 0, 0, 0
            if self._band_name.startswith("gpu"):
                processor_usage = gpu_usage = 1
            else:
                try:
                    processor_usage = cpu_usage = (
                        proc.cpu_percent(interval=None) / 100.0
                    )
                except psutil.NoSuchProcess:  # pragma: no cover
                    continue
                except psutil.AccessDenied as e:  # pragma: no cover
                    logger.warning("Access denied when getting cpu percent: %s", e)
                    processor_usage = cpu_usage = 0.0

            slot_infos.append(
                WorkerSlotInfo(
                    slot_id=slot_id,
                    session_id=session_id,
                    subtask_id=subtask_id,
                    processor_usage=processor_usage,
                )
            )

            if global_resource_ref is not None:  # pragma: no branch
                # FIXME fix band slot mistake
                delays.append(
                    global_resource_ref.update_subtask_resources.delay(
                        self._band[1],
                        session_id,
                        subtask_id,
                        Resource(
                            num_cpus=max(1.0, cpu_usage), num_gpus=max(1.0, gpu_usage)
                        ),
                    )
                )

        if delays:  # pragma: no branch
            yield global_resource_ref.update_subtask_resources.batch(*delays)
        if self._cluster_api is not None:
            await self._cluster_api.set_band_slot_infos(self._band_name, slot_infos)

        if periodical:
            self._usage_upload_task = self.ref().upload_slot_usages.tell_delay(
                periodical=True, delay=1
            )

    def dump_data(self):
        """
        Get all refs of slots of a queue
        """
        return DispatchDumpType(self._free_slots, self._fresh_slots)


class BandSlotControlActor(mo.Actor):
    @classmethod
    def gen_uid(cls, band_name: str, slot_id: int):
        return f"{band_name}_{slot_id}_band_slot_control"

    def __init__(self, manager_ref, band_name, slot_id: int):
        self._manager_ref = manager_ref
        self._band_name = band_name
        self._slot_id = slot_id
        self._report_task = None

    async def __post_create__(self):
        self._report_task = asyncio.create_task(self._report_slot_ready())

    async def _report_slot_ready(self):
        from ...cluster.api import ClusterAPI

        try:
            self._cluster_api = await ClusterAPI.create(self.address)
            await self._cluster_api.wait_node_ready()
        except mo.ActorNotExist:
            pass

        await mo.wait_actor_pool_recovered(self.address)
        await self._manager_ref.register_slot.tell(self._slot_id, os.getpid())
