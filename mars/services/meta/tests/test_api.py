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

import sys

import pytest

import mars.dataframe as md
import mars.oscar as mo
import mars.remote as mr
import mars.tensor as mt
from mars.oscar.backends.mars.allocate_strategy import RandomSubPool
from mars.services.meta.api import MetaAPI
from mars.services.meta.core import MetaStoreActor


t = mt.random.rand(10, 10)
t = t.tiles()
df = md.DataFrame(t)
df = df.tiles()
series = df[0]
series = series.tiles()
index = df.index
index = index.tiles()
obj = mr.spawn(lambda: 3)
obj = obj.tiles()

test_objects = [t, df, series, index, obj]


@pytest.mark.asyncio
@pytest.mark.parametrize('obj', test_objects)
async def test_meta_mock_api(obj):
    start_method = 'fork' if sys.platform != 'win32' else None
    pool = await mo.create_actor_pool('127.0.0.1', 2,
                                      subprocess_start_method=start_method)
    async with pool:
        session_id = 'mock_session_id'

        await mo.create_actor(MetaStoreActor, 'mock', session_id,
                              address=pool.external_address,
                              uid=MetaStoreActor.gen_uid(session_id),
                              allocate_strategy=RandomSubPool())

        meta_api = await MetaAPI.create(
            dict(session_id=session_id,
                 supervisor_address=pool.external_address))

        await meta_api.set_tileable_meta(obj)
        meta = await meta_api.get_tileable_meta(obj.key,
                                                fields=['nsplits'],
                                                tileable_type=type(obj))
        assert meta['nsplits'] == obj.nsplits
        await meta_api.del_tileable_meta(obj.key,
                                         tileable_type=type(obj))
        with pytest.raises(KeyError):
            await meta_api.get_tileable_meta(obj.key,
                                             tileable_type=type(obj))

        chunk = obj.chunks[0]

        await meta_api.set_chunk_meta(chunk)
        meta = await meta_api.get_chunk_meta(chunk.key,
                                             fields=['index'],
                                             chunk_type=type(chunk))
        assert meta['index'] == chunk.index
        await meta_api.del_chunk_meta(chunk.key,
                                      chunk_type=type(chunk))
        with pytest.raises(KeyError):
            await meta_api.get_chunk_meta(chunk.key,
                                          chunk_type=type(chunk))
