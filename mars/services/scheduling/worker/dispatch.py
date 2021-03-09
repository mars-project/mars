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
from collections import deque

from .... import oscar as mo


class RequestItem:
    event: asyncio.Event
    is_cancelled: bool

    def __init__(self, event: asyncio.Event, is_cancelled: bool = False):
        self.event = event
        self.is_cancelled = is_cancelled


class DispatchActor(mo.Actor):
    def __init__(self):
        super().__init__()
        self._free_slots = set()
        self._all_slots = set()
        self._global_requests = deque()

    async def acquire_free_slot(self):
        if self._free_slots:
            return self._free_slots.pop()

        event = asyncio.Event()
        req_item = RequestItem(event)

        async def slot_waiter():
            try:
                await event.wait()
                ref = self._free_slots.pop()
            except asyncio.CancelledError:
                req_item.is_cancelled = True
                self._apply_next_request()
                raise
            return ref

        self._global_requests.append(req_item)
        return slot_waiter()

    def register_free_slot(self, ref):
        self._free_slots.add(ref)
        self._all_slots.add(ref)
        self._apply_next_request()

    def _apply_next_request(self):
        if not self._free_slots:
            return
        while self._global_requests:
            req_item = self._global_requests.popleft()
            if not req_item.is_cancelled:
                req_item.event.set()
                break

    def get_slots(self):
        """
        Get all refs of slots of a queue
        """
        return list(self._all_slots)
