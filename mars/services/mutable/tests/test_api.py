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

import sys
import tempfile

import pytest

from .... import oscar as mo
from ....utils import get_next_port
from ...cluster import MockClusterAPI
from ...meta import MockMetaAPI
from ...session import MockSessionAPI
from ...web import WebActor
from ...storage import MockStorageAPI
from ..api.web import MutableWebAPIHandler


@pytest.mark.asyncio
async def test_web_mutable_api():
    from ..api.web import WebMutableAPI

    tempdir = tempfile.mkdtemp()
    start_method = 'fork' if sys.platform != 'win32' else None
    pool = await mo.create_actor_pool('127.0.0.1', 1,
                                    subprocess_start_method=start_method)

    async with pool:
        session_id = 'mock_session_id'
        await MockClusterAPI.create(
            address=pool.external_address)
        await MockSessionAPI.create(
            session_id=session_id,
            address=pool.external_address)
        await MockMetaAPI.create(
            session_id=session_id,
            address=pool.external_address)
        await MockStorageAPI.create(
            address=pool.external_address,
            session_id=session_id,
            storage_configs={'shared_memory': dict(),
                             'disk': dict(root_dirs=[tempdir])})

        web_config = {
            'port': get_next_port(),
            'web_handlers': {
                MutableWebAPIHandler.get_root_pattern(): MutableWebAPIHandler
            }
        }
        await mo.create_actor(WebActor, web_config, address=pool.external_address)

        web_mutable_api = WebMutableAPI(
            session_id, f'http://127.0.0.1:{web_config["port"]}')

        tensor = await web_mutable_api.create_mutable_tensor(shape=(10, 10), dtype='int', chunk_size=(5, 5), name='mytensor', default_value=2)
        await tensor.write(((1,), (1,)), 1)
        [value] = await tensor[1, 1]
        assert value == 1

        await MockStorageAPI.cleanup(pool.external_address)
        await MockClusterAPI.cleanup(pool.external_address)
