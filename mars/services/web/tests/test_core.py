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

import os
import sys

import pytest
from tornado.httpclient import HTTPError

import mars.oscar as mo
from mars.services.web import WebActor, web_api, MarsServiceWebAPIHandler, \
    MarsWebAPIClientMixin
from mars.utils import get_next_port


class TestAPIHandler(MarsServiceWebAPIHandler):
    _root_pattern = '/api/test/(?P<test_id>[^/]+)'

    @web_api('', method='get')
    def get_method_root(self, test_id):
        self.write(f'get_root_value_{test_id}')

    @web_api('', method='post')
    def post_method_root(self, test_id):
        self.write(f'post_root_value_{test_id}')

    @web_api('subtest/(?P<subtest_id>[^/]+)', method='get')
    def get_method_sub_patt(self, test_id, subtest_id):
        self.write(f'get_sub_value_{test_id}_{subtest_id}')

    @web_api('subtest/(?P<subtest_id>[^/]+)', method='get',
             arg_filter={'action': 'a1'})
    async def get_method_sub_patt_match_arg1(self, test_id, subtest_id):
        self.write(f'get_sub_value_{test_id}_{subtest_id}_action1')

    @web_api('subtest/(?P<subtest_id>[^/]+)', method='get',
             arg_filter={'action': 'a2'})
    async def get_method_sub_patt_match_arg2(self, test_id, subtest_id):
        self.write(f'get_sub_value_{test_id}_{subtest_id}_action2')

    @web_api('subtest_error', method='get')
    def get_with_error(self, test_id):
        raise ValueError


@pytest.fixture
async def actor_pool():
    start_method = os.environ.get('POOL_START_METHOD', 'forkserver') \
        if sys.platform != 'win32' else None
    pool = await mo.create_actor_pool('127.0.0.1', n_process=0,
                                      subprocess_start_method=start_method)
    async with pool:
        web_config = {
            'host': '127.0.0.1',
            'port': get_next_port(),
            'web_handlers': {TestAPIHandler.get_root_pattern(): TestAPIHandler},
        }
        await mo.create_actor(
            WebActor, web_config, address=pool.external_address)
        yield pool, web_config['port']


class SimpleWebClient(MarsWebAPIClientMixin):
    async def fetch(self, path, **kwargs):
        return await self._request_url(path, **kwargs)


@pytest.mark.asyncio
async def test_web_api(actor_pool):
    pool, web_port = actor_pool

    client = SimpleWebClient()

    res = await client.fetch(f'http://localhost:{web_port}/api/test/test_id')
    assert res.body.decode() == 'get_root_value_test_id'

    res = await client.fetch(f'http://localhost:{web_port}/api/test/test_id',
                             method='POST', body=b'')
    assert res.body.decode() == 'post_root_value_test_id'

    res = await client.fetch(f'http://localhost:{web_port}/api/test/test_id/subtest/sub_tid')
    assert res.body.decode() == 'get_sub_value_test_id_sub_tid'

    res = await client.fetch(f'http://localhost:{web_port}/api/test/test_id/'
                             f'subtest/sub_tid?action=a1')
    assert res.body.decode() == 'get_sub_value_test_id_sub_tid_action1'

    res = await client.fetch(f'http://localhost:{web_port}/api/test/test_id/'
                             f'subtest/sub_tid?action=a2')
    assert res.body.decode() == 'get_sub_value_test_id_sub_tid_action2'

    with pytest.raises(HTTPError) as excinfo:
        await client.fetch(f'http://localhost:{web_port}/api/test/test_id/non_exist')
    assert excinfo.value.code == 404

    with pytest.raises(ValueError):
        await client.fetch(f'http://localhost:{web_port}/api/test/test_id/subtest_error')
