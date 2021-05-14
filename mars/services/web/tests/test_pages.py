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

import pytest
from tornado.httpclient import AsyncHTTPClient

import mars.oscar as mo
from mars.services import start_services, NodeRole
from mars.services.web import WebActor


@pytest.fixture
async def actor_pools():
    async def start_pool(is_worker: bool):
        if is_worker:
            kw = dict(
                n_process=2,
                labels=['main'] + ['numa-0'] * 2,
                subprocess_start_method='spawn'
            )
        else:
            kw = dict(n_process=0,
                      subprocess_start_method='spawn')
        pool = await mo.create_actor_pool('127.0.0.1', **kw)
        await pool.start()
        return pool

    sv_pool, worker_pool = await asyncio.gather(
        start_pool(False), start_pool(True)
    )
    config = {
        "services": ["cluster", "session", "lifecycle", "meta", "task",
                     "web"],
        "cluster": {
            "backend": "fixed",
            "lookup_address": sv_pool.external_address,
            "resource": {"numa-0": 2}
        },
        "meta": {
            "store": "dict"
        },
        "task": {}
    }
    await start_services(
        NodeRole.SUPERVISOR, config, address=sv_pool.external_address)
    await start_services(
        NodeRole.WORKER, config, address=worker_pool.external_address)

    web_ref = await mo.actor_ref(WebActor.default_uid(),
                                 address=sv_pool.external_address)
    web_addr = await web_ref.get_web_address()

    yield sv_pool, worker_pool, web_addr
    await asyncio.gather(sv_pool.stop(), worker_pool.stop())


@pytest.mark.asyncio
async def test_web_ui(actor_pools):
    sv_pool, worker_pool, web_addr = actor_pools

    client = AsyncHTTPClient()
    res = await client.fetch(web_addr)
    assert res.code == 200

    res = await client.fetch(f'{web_addr}/supervisor')

    print(web_addr)
    await asyncio.sleep(100000)
