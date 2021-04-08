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

import mars.oscar as mo
from mars.services import start_services, NodeRole
from mars.services.session import SessionAPI


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
