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
import contextlib
import sys

from .file import AioFileObject, AioFilesystem
from .lru import alru_cache
from .parallelism import AioEvent


if sys.version_info[:2] < (3, 7):
    # patch run and get_running_loop etc for python 3.6
    from ._runners import get_running_loop, run

    asyncio.run = run
    asyncio.get_running_loop = get_running_loop
    asyncio.create_task = asyncio.ensure_future

    # patch async generator
    from async_generator import asynccontextmanager
    contextlib.asynccontextmanager = asynccontextmanager


def create_lock(loop=None):
    async def _create_lock():
        return asyncio.Lock()

    loop = loop or asyncio.get_event_loop()
    return loop.run_until_complete(_create_lock())
