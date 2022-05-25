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

import os
import sys

import pytest

from ..... import oscar as mo


class DummyActor(mo.Actor):
    def __init__(self, value):
        super().__init__()

        if value < 0:
            raise ValueError("value < 0")
        self.value = value

    async def add(self, value):
        if not isinstance(value, int):
            raise TypeError("add number must be int")
        self.value += value
        return self.value


@pytest.fixture
async def actor_pool_context():
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    pool = await mo.create_actor_pool(
        "test://127.0.0.1", n_process=2, subprocess_start_method=start_method
    )
    async with pool:
        yield pool


@pytest.mark.asyncio
async def test_simple(actor_pool_context):
    pool = actor_pool_context
    actor_ref = await mo.create_actor(
        DummyActor,
        100,
        address=pool.external_address,
        allocate_strategy=mo.allocate_strategy.RandomSubPool(),
    )
    assert await actor_ref.add(1) == 101
