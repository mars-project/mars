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

import mars.oscar as mo

import numpy as np
import pytest

from mars import tensor as mt
from mars.core import tile
from mars.services.cluster import MockClusterAPI
from mars.services.lifecycle.supervisor.tracker import LifecycleTrackerActor
from mars.services.meta import MockMetaAPI
from mars.services.session import MockSessionAPI
from mars.services.storage import MockStorageAPI, DataNotExist


@pytest.mark.asyncio
async def test_tracker():
    pool = await mo.create_actor_pool('127.0.0.1', n_process=0)

    async with pool:
        addr = pool.external_address
        session_id = 'test_session'
        await MockClusterAPI.create(addr)
        await MockSessionAPI.create(addr, session_id=session_id)
        meta_api = await MockMetaAPI.create(session_id, addr)
        storage_api = await MockStorageAPI.create(session_id, addr)

        try:
            tracker = await mo.create_actor(
                LifecycleTrackerActor, session_id,
                uid=LifecycleTrackerActor.gen_uid(session_id),
                address=pool.external_address)

            t = mt.random.rand(15, 5, chunk_size=5)
            t = tile(t)

            tileable_key = t.key
            chunk_keys = []
            for c in t.chunks:
                chunk_keys.append(c.key)
                await meta_api.set_chunk_meta(c, bands=[(addr, 'numa-0')])
                await storage_api.put(c.key, np.random.rand(5, 5))

            await tracker.track(tileable_key, chunk_keys)
            await tracker.incref_tileables([tileable_key])
            await tracker.incref_chunks(chunk_keys[:2])
            await tracker.decref_chunks(chunk_keys[:2])
            await tracker.decref_tileables([tileable_key])
            assert len(await tracker.get_all_chunk_ref_counts()) == 0

            for chunk_key in chunk_keys:
                with pytest.raises(KeyError):
                    await meta_api.get_chunk_meta(chunk_key)
            for chunk_key in chunk_keys:
                with pytest.raises(DataNotExist):
                    await storage_api.get(chunk_key)
        finally:
            await MockStorageAPI.cleanup(pool.external_address)
