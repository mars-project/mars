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

import pytest

from .... import oscar as mo
from ..procinfo import ProcessInfoManagerActor


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool(
        "127.0.0.1", n_process=2, labels=["main", "numa-0", "gpu-0"]
    )
    async with pool:
        yield pool


@pytest.mark.asyncio
async def test_proc_info(actor_pool):
    address = actor_pool.external_address
    manager_ref = await mo.create_actor(
        ProcessInfoManagerActor,
        uid=ProcessInfoManagerActor.default_uid(),
        address=address,
    )  # type: ProcessInfoManagerActor | mo.ActorRef
    pool_cfgs = await manager_ref.get_pool_configs()
    for cfg, expect_label in zip(pool_cfgs, ["main", "numa-0", "gpu-0"]):
        assert cfg["label"] == expect_label
    stacks = await manager_ref.get_thread_stacks()
    assert len(stacks) == len(pool_cfgs)
