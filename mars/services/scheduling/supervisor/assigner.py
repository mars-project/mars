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
import random
from collections import defaultdict
from typing import List, Dict, Tuple

from .... import oscar as mo
from ....core.operand import Fetch, FetchShuffle
from ...core import NodeRole
from ...subtask import Subtask


class AssignerActor(mo.Actor):
    @classmethod
    def gen_uid(cls, session_id: str):
        return f'{session_id}_assigner'

    def __init__(self, session_id: str):
        self._session_id = session_id
        self._slots_ref = None

        self._cluster_api = None
        self._meta_api = None

        self._bands = []
        self._band_watch_task = None
        self._available_bands = []
        self._available_band_watch_task = None

    async def __post_create__(self):
        from ...cluster.api import ClusterAPI
        from ...meta.api import MetaAPI
        self._cluster_api = await ClusterAPI.create(self.address)
        self._meta_api = await MetaAPI.create(
            session_id=self._session_id, address=self.address)

        self._bands = list(await self._cluster_api.get_all_bands())

        from .globalslot import GlobalSlotManagerActor
        self._slots_ref = await mo.actor_ref(
            GlobalSlotManagerActor.default_uid(), address=self.address)

        self._available_bands = list(await self._slots_ref.get_available_bands())

        async def watch_bands():
            async for bands in self._cluster_api.watch_all_bands(NodeRole.WORKER):
                self._bands = list(bands)

        self._band_watch_task = asyncio.create_task(watch_bands())

        async def watch_available_bands():
            while True:
                self._available_bands = list(await self._slots_ref.watch_available_bands())

        self._available_band_watch_task = asyncio.create_task(watch_available_bands())

    async def __pre_destroy__(self):
        if self._band_watch_task is not None:  # pragma: no branch
            self._band_watch_task.cancel()

        if self._available_band_watch_task is not None:  # pragma: no branch
            self._available_band_watch_task.cancel()

    async def assign_subtasks(self, subtasks: List[Subtask]):
        inp_keys = set()
        selected_bands = dict()
        for subtask in subtasks:
            if subtask.expect_bands:
                if all(expect_band in self._available_bands
                       for expect_band in subtask.expect_bands):
                    # pass if all expected bands are available
                    selected_bands[subtask.subtask_id] = subtask.expect_bands
                else:
                    # exclude expected but blocked bands
                    expect_available_bands = [expect_band
                                              for expect_band in subtask.expect_bands
                                              if expect_band in self._available_bands]
                    # fill in if all expected bands are blocked
                    if not expect_available_bands:
                        expect_available_bands = [self.reassign_band()]
                    selected_bands[subtask.subtask_id] = expect_available_bands
                continue
            for indep_chunk in subtask.chunk_graph.iter_indep():
                if isinstance(indep_chunk.op, Fetch):
                    inp_keys.add(indep_chunk.key)
                elif isinstance(indep_chunk.op, FetchShuffle):
                    if not self._bands:
                        self._bands = list(await self._cluster_api.get_all_bands(
                            NodeRole.WORKER))
                    selected_bands[subtask.subtask_id] = [self.reassign_band()]
                    break

        fields = ['store_size', 'bands']
        inp_keys = list(inp_keys)
        metas = await self._meta_api.get_chunk_meta.batch(
            *(self._meta_api.get_chunk_meta.delay(key, fields) for key in inp_keys)
        )

        inp_metas = dict(zip(inp_keys, metas))
        assigns = []
        for subtask in subtasks:
            if subtask.subtask_id in selected_bands:
                bands = selected_bands[subtask.subtask_id]
            else:
                band_sizes = defaultdict(lambda: 0)
                for inp in subtask.chunk_graph.iter_indep():
                    if not isinstance(inp.op, Fetch):  # pragma: no cover
                        continue
                    meta = inp_metas[inp.key]
                    for band in meta['bands']:
                        if band not in self._available_bands:
                            band = self.reassign_band()
                        band_sizes[band] += meta['store_size']
                bands = []
                max_size = -1
                for band, size in band_sizes.items():
                    if size > max_size:
                        bands = [band]
                        max_size = size
                    elif size == max_size:
                        bands.append(band)
            assigns.append(random.choice(bands))
        return assigns

    def reassign_band(self):
        assert self._available_bands
        return random.choice(self._available_bands)

    async def reassign_subtasks(self, band_num_queued_subtasks: Dict[Tuple, int]) -> Dict[Tuple, int]:
        used_bands = band_num_queued_subtasks.keys()
        # select available bands which may contain new available unused ones
        bands_to_assign = [used_band for used_band in used_bands if used_band in self._available_bands]
        # approximate total of subtasks in each band
        mean = int(sum(band_num_queued_subtasks.values()) / len(bands_to_assign))
        # calculate the differential steps of moving subtasks
        # move < 0 means subtasks should move out and vice versa
        # blocked bands no longer hold subtasks
        move_queued_subtasks = {band: mean - num if band in self._available_bands else -num
                                for band, num in band_num_queued_subtasks.items()}
        # ensure the balance of moving in and out
        total_move = sum(move_queued_subtasks.values())
        if total_move != 0:
            move_queued_subtasks[random.choice(self._available_bands)] -= total_move
        return dict(sorted(move_queued_subtasks.items(), key=lambda item: item[1]))
