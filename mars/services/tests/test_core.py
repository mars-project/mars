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

import pytest
from tornado import httpclient

from ... import oscar as mo
from ...utils import get_next_port
from .. import (
    NodeRole,
    start_services,
    stop_services,
    create_service_session,
    destroy_service_session,
)


@pytest.fixture
async def actor_pool_context():
    pool = await mo.create_actor_pool(f"127.0.0.1:{get_next_port()}", n_process=0)
    await pool.start()
    try:
        yield pool
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_start_service(actor_pool_context):
    from .test_svcs.test_svc1.supervisor import SvcSessionActor1

    pool = actor_pool_context
    web_port = get_next_port()
    config = {
        "services": [["test_svc1"], "test_svc2", "test_warn_svc", "web"],
        "modules": "mars.services.tests.test_svcs",
        "test_svc1": {"uid": "TestActor1", "arg1": "val1"},
        "test_svc2": {"uid": "TestActor2", "arg2": "val2", "ref": "TestActor1"},
        "web": {"port": web_port},
    }
    with pytest.warns(RuntimeWarning) as record:
        await start_services(NodeRole.SUPERVISOR, config, address=pool.external_address)
        assert "test_warn_svc" in str(record[-1].message)

    ref1 = await mo.actor_ref("TestActor1", address=pool.external_address)
    ref2 = await mo.actor_ref("TestActor2", address=pool.external_address)
    assert await ref1.get_arg() == "val1"
    assert await ref2.get_arg() == "val1:val2"

    with pytest.raises(ImportError):
        await start_services(
            NodeRole.SUPERVISOR,
            {"services": ["non-exist-svc"]},
            address=pool.external_address,
        )

    session_id = "test_session"
    await create_service_session(
        NodeRole.SUPERVISOR,
        config,
        session_id=session_id,
        address=pool.external_address,
    )
    assert await mo.has_actor(
        mo.create_actor_ref(
            uid=SvcSessionActor1.gen_uid(session_id), address=pool.external_address
        )
    )
    await destroy_service_session(
        NodeRole.SUPERVISOR,
        config,
        session_id=session_id,
        address=pool.external_address,
    )
    assert not await mo.has_actor(
        mo.create_actor_ref(
            uid=SvcSessionActor1.gen_uid(session_id), address=pool.external_address
        )
    )

    client = httpclient.AsyncHTTPClient()
    resp = await client.fetch(f"http://127.0.0.1:{web_port}/test_actor1/test_api")
    assert resp.body.decode() == "val1"

    await stop_services(NodeRole.SUPERVISOR, config, address=pool.external_address)
    assert not await mo.has_actor(
        mo.create_actor_ref("TestActor1", address=pool.external_address)
    )
    assert not await mo.has_actor(
        mo.create_actor_ref("TestActor2", address=pool.external_address)
    )
