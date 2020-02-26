# -*- coding: utf-8 -*-
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
import logging
import os

from .utils import SchedulerActor
from .. import kvstore
from ..config import options
from ..utils import wait_results

logger = logging.getLogger(__name__)


class KVStoreActor(SchedulerActor):
    """
    Actor handling reading and writing to an external KV store.
    """
    def __init__(self):
        super().__init__()
        self._store = kvstore.get(options.kv_store)

    async def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())
        await super().post_create()

    async def read(self, item, recursive=False, sort=False):
        return await self._store.read(item, recursive=recursive, sort=sort)

    async def read_batch(self, items, recursive=False, sort=False):
        futures = [asyncio.ensure_future(self._store.read(item, recursive=recursive, sort=sort))
                   for item in items]
        return (await wait_results(futures))[0]

    async def write(self, key, value):
        return await self._store.write(key, value)

    async def write_batch(self, items):
        wrap = lambda x: (x,) if not isinstance(x, tuple) else x
        futures = [asyncio.ensure_future(self.write(*wrap(it))) for it in items]
        await wait_results(futures)

    async def delete(self, key, dir=False, recursive=False, silent=False):
        try:
            return await self._store.delete(key, dir=dir, recursive=recursive)
        except KeyError:
            if not silent:
                raise
