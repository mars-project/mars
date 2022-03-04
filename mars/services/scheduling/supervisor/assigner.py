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
import heapq
import itertools
import random
from collections import defaultdict
from typing import Dict, List, Set

from .... import oscar as mo
from ....core.operand import Fetch, FetchShuffle
from ....typing import BandType
from ...core import NodeRole
from ...subtask import Subtask
from ..errors import NoMatchingSlots, NoAvailableBand


class AssignerActor(mo.Actor):
    _bands: List[BandType]

    @classmethod
    def gen_uid(cls, session_id: str):
        return f"{session_id}_assigner"

    def __init__(self, session_id: str):
        self._session_id = session_id
        self._slots_ref = None

        self._cluster_api = None
        self._meta_api = None

        self._bands = []
        self._address_to_bands = dict()
        self._device_type_to_bands = dict()
        self._band_watch_task = None

    async def __post_create__(self):
        from ...cluster.api import ClusterAPI
        from ...meta.api import MetaAPI

        self._cluster_api = await ClusterAPI.create(self.address)
        self._meta_api = await MetaAPI.create(
            session_id=self._session_id, address=self.address
        )

        async def watch_bands():
            async for bands in self._cluster_api.watch_all_bands(NodeRole.WORKER):
                self._update_bands(list(bands))

        self._band_watch_task = asyncio.create_task(watch_bands())

    async def __pre_destroy__(self):
        if self._band_watch_task is not None:  # pragma: no branch
            self._band_watch_task.cancel()

    def _update_bands(self, bands: List[BandType]):
        self._bands = bands

        grouped_bands = itertools.groupby(sorted(self._bands), key=lambda b: b[0])
        self._address_to_bands = {k: list(v) for k, v in grouped_bands}

        grouped_bands = itertools.groupby(
            sorted(("numa" if b[1].startswith("numa") else "gpu", b) for b in bands),
            key=lambda tp: tp[0],
        )
        self._device_type_to_bands = {
            k: [v[1] for v in tps] for k, tps in grouped_bands
        }

    def _get_device_bands(self, is_gpu: bool):
        band_prefix = "numa" if not is_gpu else "gpu"
        filtered_bands = self._device_type_to_bands.get(band_prefix) or []
        if not filtered_bands:
            raise NoMatchingSlots("gpu" if is_gpu else "cpu")
        return filtered_bands

    def _get_random_band(
        self,
        is_gpu: bool,
        exclude_bands: Set[BandType] = None,
        exclude_bands_force: bool = False,
    ):
        if exclude_bands:
            avail_bands = [
                band
                for band in self._get_device_bands(is_gpu)
                if band not in exclude_bands
            ]
            if avail_bands:
                return random.choice(avail_bands)
            elif exclude_bands_force:
                raise NoAvailableBand(
                    f"No bands available after excluding bands {exclude_bands}"
                )
        return random.choice(self._get_device_bands(is_gpu))

    async def assign_subtasks(
        self,
        subtasks: List[Subtask],
        exclude_bands: Set[BandType] = None,
        exclude_bands_force: bool = False,
    ):
        exclude_bands = exclude_bands or set()
        inp_keys = set()
        selected_bands = dict()
        if not self._bands:
            self._update_bands(
                list(await self._cluster_api.get_all_bands(NodeRole.WORKER))
            )
        for subtask in subtasks:
            is_gpu = any(c.op.gpu for c in subtask.chunk_graph)
            if subtask.expect_bands:
                # exclude expected but unready bands
                expect_available_bands = [
                    expect_band
                    for expect_band in subtask.expect_bands
                    if expect_band in self._bands and expect_band not in exclude_bands
                ]
                # fill in if all expected bands are unready
                if not expect_available_bands:
                    expect_available_bands = [
                        self._get_random_band(
                            is_gpu, exclude_bands, exclude_bands_force
                        )
                    ]
                selected_bands[subtask.subtask_id] = expect_available_bands
                continue
            for indep_chunk in subtask.chunk_graph.iter_indep():
                if isinstance(indep_chunk.op, Fetch):
                    inp_keys.add(indep_chunk.key)
                elif isinstance(indep_chunk.op, FetchShuffle):
                    selected_bands[subtask.subtask_id] = [
                        self._get_random_band(
                            is_gpu, exclude_bands, exclude_bands_force
                        )
                    ]
                    break

        fields = ["store_size", "bands"]
        inp_keys = list(inp_keys)
        metas = await self._meta_api.get_chunk_meta.batch(
            *(self._meta_api.get_chunk_meta.delay(key, fields) for key in inp_keys)
        )

        inp_metas = dict(zip(inp_keys, metas))
        assigns = []
        for subtask in subtasks:
            is_gpu = any(c.op.gpu for c in subtask.chunk_graph)
            band_prefix = "numa" if not is_gpu else "gpu"
            filtered_bands = self._get_device_bands(is_gpu)

            if subtask.subtask_id in selected_bands:
                bands = selected_bands[subtask.subtask_id]
            else:
                band_sizes = defaultdict(lambda: 0)
                for inp in subtask.chunk_graph.iter_indep():
                    if not isinstance(inp.op, Fetch):  # pragma: no cover
                        continue
                    meta = inp_metas[inp.key]
                    for band in meta["bands"]:
                        if not band[1].startswith(band_prefix):
                            sel_bands = [
                                b
                                for b in self._address_to_bands[band[0]]
                                if b[1].startswith(band_prefix)
                                and b not in exclude_bands
                            ]
                            if sel_bands:
                                band = random.choice(sel_bands)
                        if band not in filtered_bands or band in exclude_bands:
                            band = self._get_random_band(
                                is_gpu, exclude_bands, exclude_bands_force
                            )
                        band_sizes[band] += meta["store_size"]
                bands = []
                max_size = -1
                for band, size in band_sizes.items():
                    if size > max_size:
                        bands = [band]
                        max_size = size
                    elif size == max_size:
                        bands.append(band)
            band = random.choice(bands)
            if band in exclude_bands and exclude_bands_force:
                raise NoAvailableBand(
                    f"No bands available for subtask {subtask.subtask_id} after "
                    f"excluded {exclude_bands}"
                )
            if subtask.bands_specified and band not in subtask.expect_bands:
                raise Exception(
                    f"No bands available for subtask {subtask.subtask_id} on bands {subtask.expect_bands} "
                    f"after excluded {exclude_bands}"
                )
            assigns.append(band)
        return assigns

    async def reassign_subtasks(
        self, band_to_queued_num: Dict[BandType, int], used_slots: Dict[BandType, int] = None
    ) -> Dict[BandType, int]:
        used_slots = used_slots or {}
        move_queued_subtasks = {}
        for is_gpu in (False, True):
            band_name_prefix = "numa" if not is_gpu else "gpu"
            device_bands = [
                band for band in self._bands if band[1].startswith(band_name_prefix)
            ]
            if not device_bands:
                continue
            device_bands_set = set(device_bands)
            device_band_to_queued_num = {
                k: v
                for k, v in band_to_queued_num.items()
                if k[1].startswith(band_name_prefix)
            }
            used_bands = device_band_to_queued_num.keys()
            num_used_bands = len(used_bands)
            if num_used_bands == 1:
                [(band, length)] = device_band_to_queued_num.items()
                if length == 0:
                    move_queued_subtasks.update({band: 0})
                    continue
                # no need to balance when there's only one band initially
                if len(device_bands) == 1 and band == device_bands[0]:
                    move_queued_subtasks.update({band: 0})
                    continue
            # approximate total of subtasks moving to each ready band
            num_all_subtasks = sum(device_band_to_queued_num.values()) + sum(used_slots.values())
            # If the remainder is not 0, move in and out may be unequal.
            mean = int(num_all_subtasks / len(device_bands))
            # all_bands (namely) includes:
            # a. ready bands recorded in band_num_queued_subtasks
            # b. ready bands not recorded in band_num_queued_subtasks
            # c. unready bands recorded in band_num_queued_subtasks
            # a. + b. = self._bands, a. + c. = bands in band_num_queued_subtasks
            all_bands = list(
                device_bands_set | set(device_band_to_queued_num.keys())
            )
            # calculate the differential steps of moving subtasks
            # move < 0 means subtasks should move out and vice versa
            # unready bands no longer hold subtasks
            # assuming bands not recorded in band_num_queued_subtasks hold 0 subtasks
            band_move_nums = {}
            for band in all_bands:
                existing_subtask_nums = device_band_to_queued_num.get(band, 0)
                if band in device_bands:
                    band_move_num = mean - existing_subtask_nums - used_slots.get(band, 0)
                    # If slot of band already has some subtasks running, band_move_num may be greater than
                    # existing_subtask_nums.
                    if band_move_num + existing_subtask_nums < 0:
                        band_move_num = -existing_subtask_nums
                else:
                    band_move_num = -existing_subtask_nums
                band_move_nums[band] = band_move_num
            self._balance_move_out_subtasks(band_move_nums)
            assert sum(band_move_nums.values()) == 0, f"band_move_nums {band_move_nums}"
            move_queued_subtasks.update(band_move_nums)
        return dict(sorted(move_queued_subtasks.items(), key=lambda item: item[1]))

    def _balance_move_out_subtasks(self, band_move_nums: Dict[BandType, int]):
        # ensure the balance of moving in and out
        total_move = sum(band_move_nums.values())
        if total_move != 0:
            unit = (total_move > 0) - (total_move < 0)

            class BandHeapItem:
                def __init__(self, band_):
                    self.band = band_

                def __lt__(self, other: "BandHeapItem"):
                    return (band_move_nums.get(self.band, 0) > band_move_nums.get(other.band, 0)) * unit
            bands_queue = [BandHeapItem(band) for band, move_nums in band_move_nums.items() if move_nums * unit > 0]
            heapq.heapify(bands_queue)
            while total_move != 0:
                item = heapq.heappop(bands_queue)
                band_move_nums[item.band] -= unit
                total_move -= unit
                heapq.heappush(bands_queue, item)
