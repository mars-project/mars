# Copyright 1999-2022 Alibaba Group Holding Ltd.
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
import sys
import threading
import traceback
from typing import Dict, List

from ... import oscar as mo
from ...oscar.backends.allocate_strategy import ProcessIndex


class ProcessInfoManagerActor(mo.StatelessActor):
    _process_refs: List[mo.ActorRef]

    def __init__(self):
        self._process_refs = []
        self._pool_configs = []

    async def __post_create__(self):
        index = 0
        while True:
            try:
                ref = await mo.create_actor(
                    ProcessInfoActor,
                    process_index=index,
                    uid=ProcessInfoActor.gen_uid(index),
                    address=self.address,
                    allocate_strategy=ProcessIndex(index),
                )
            except IndexError:
                break

            index += 1
            self._process_refs.append(ref)

        self._pool_configs = await asyncio.gather(
            *[ref.get_pool_config() for ref in self._process_refs]
        )

    async def get_pool_configs(self) -> List[Dict]:
        return self._pool_configs

    async def get_thread_stacks(self) -> List[Dict[int, List[str]]]:
        stack_tasks = [
            asyncio.create_task(ref.get_thread_stacks()) for ref in self._process_refs
        ]
        await asyncio.wait(stack_tasks, return_when=asyncio.ALL_COMPLETED)

        results = []
        for fut in stack_tasks:
            try:
                results.append(fut.result())
            except (mo.ActorNotExist, mo.ServerClosed):
                results.append(None)
        return results


class ProcessInfoActor(mo.StatelessActor):
    def __init__(self, process_index: int = 0):
        self._process_index = process_index
        self._pool_config = None

    async def __post_create__(self):
        self._pool_config = await mo.get_pool_config(self.address)

    @classmethod
    def gen_uid(cls, process_index: int):
        return f"process_info_{process_index}"

    def get_pool_config(self) -> dict:
        idx = self._pool_config.get_process_index(self.address)
        return self._pool_config.get_pool_config(idx)

    @classmethod
    def get_thread_stacks(cls) -> Dict[str, List[str]]:
        frames = sys._current_frames()
        stacks = dict()
        for th in threading.enumerate():
            tid = getattr(th, "native_id", th.ident)
            stack_key = f"{tid}:{th.name}"
            stacks[stack_key] = traceback.format_stack(frames[th.ident])
        return stacks
