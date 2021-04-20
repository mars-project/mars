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


import pytest
import numpy as np

import mars.oscar as mo
import mars.tensor as mt
from mars.services import start_services, NodeRole
from mars.services.session import SessionAPI
from mars.core import TileableGraph, TileableGraphBuilder


@pytest.mark.asyncio
async def test_meta_service():
    pool = await mo.create_actor_pool('127.0.0.1', n_process=0)

    async with pool:
        config = {
            "services": ["cluster", "session", "meta", "task"],
            "cluster": {
                "backend": "fixed",
                "lookup_address": pool.external_address,
            },
            "meta": {
                "store": "dict"
            }
        }
        await start_services(
            NodeRole.SUPERVISOR, config, address=pool.external_address)

        session_api = await SessionAPI.create(pool.external_address)
        session_id = 'test_session'
        session_address = await session_api.create_session(session_id)
        assert session_address == pool.external_address
        assert await session_api.has_session(session_id) is True
        assert await session_api.get_session_address(session_id) == session_address
        await session_api.delete_session(session_id)
        assert await session_api.has_session(session_id) is False


@pytest.mark.asyncio
async def test_last_idle_time():
    pool = await mo.create_actor_pool('127.0.0.1', n_process=0)

    async with pool:
        config = {
            "services": ["cluster", "session", "meta", "task"],
            "cluster": {
                "backend": "fixed",
                "lookup_address": pool.external_address,
            },
            "meta": {
                "store": "dict"
            }
        }
        await start_services(
            NodeRole.SUPERVISOR, config, address=pool.external_address)

        session_api = await SessionAPI.create(pool.external_address)
        session_id = 'test_session'
        session_address = await session_api.create_session(session_id)
        assert session_address == pool.external_address
        last_idle_time = await session_api.last_idle_time()
        assert last_idle_time is not None
        assert await session_api.last_idle_time() == last_idle_time
        # Submit a graph
        from ...task.api import TaskAPI
        task_api = await TaskAPI.create(session_id, session_address)
        raw = np.random.RandomState(0).rand(10, 10)
        a = mt.tensor(raw, chunk_size=5)
        b = a + 1

        graph = TileableGraph([b.data])
        next(TileableGraphBuilder(graph).build())
        await task_api.submit_tileable_graph(graph, fuse_enabled=False)

        # The last idle time is changed.
        new_last_idle_time = await session_api.last_idle_time()
        assert new_last_idle_time != last_idle_time
        assert await session_api.last_idle_time() == new_last_idle_time
        assert await session_api.get_session_address(session_id) == session_address
        await session_api.delete_session(session_id)
        assert await session_api.has_session(session_id) is False
