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

import pytest

from .... import dataframe as md
from .... import oscar as mo
from .... import remote as mr
from .... import tensor as mt
from ....core import tile
from ....utils import get_next_port
from ... import start_services, stop_services, NodeRole
from ...cluster import MockClusterAPI
from ...session import MockSessionAPI, SessionAPI
from .. import MockMetaAPI, MetaAPI, WorkerMetaAPI, WebMetaAPI


t = mt.random.rand(10, 10)
df = md.DataFrame(t)
series = df[0]
index = df.index
obj = mr.spawn(lambda: 3)
t, df, series, index, obj = tile(t, df, series, index, obj)

test_objects = [t, df, series, index, obj]


@pytest.mark.asyncio
@pytest.mark.parametrize("obj", test_objects)
async def test_meta_mock_api(obj):
    start_method = "fork" if sys.platform != "win32" else None
    pool = await mo.create_actor_pool(
        "127.0.0.1", 2, subprocess_start_method=start_method
    )
    async with pool:
        session_id = "mock_session_id"

        await MockClusterAPI.create(pool.external_address)
        await MockSessionAPI.create(pool.external_address, session_id=session_id)
        meta_api = await MockMetaAPI.create(
            session_id=session_id, address=pool.external_address
        )

        await meta_api.set_tileable_meta(obj)
        meta = await meta_api.get_tileable_meta(obj.key, fields=["nsplits"])
        assert meta["nsplits"] == obj.nsplits
        await meta_api.del_tileable_meta(obj.key)
        with pytest.raises(KeyError):
            await meta_api.get_tileable_meta(obj.key)

        chunk = obj.chunks[0]

        await meta_api.set_chunk_meta(chunk, bands=[(pool.external_address, "numa-0")])
        meta = await meta_api.get_chunk_meta(chunk.key, fields=["index", "bands"])
        assert meta["index"] == chunk.index
        assert meta["bands"] == [(pool.external_address, "numa-0")]

        for i in range(2):
            band = (f"1.2.3.{i}:1234", "numa-0")
            await meta_api.add_chunk_bands(chunk.key, [band])
            meta = await meta_api.get_chunk_meta(chunk.key, fields=["bands"])
            assert band in meta["bands"]
        meta = await meta_api.get_chunk_meta(chunk.key, fields=["bands"])
        band = meta["bands"][0]
        chunks = await meta_api.get_band_chunks(band)
        assert chunk.key in chunks
        await meta_api.remove_chunk_bands(chunk.key, [band])
        meta = await meta_api.get_chunk_meta(chunk.key, fields=["bands"])
        assert band not in meta["bands"]

        await meta_api.del_chunk_meta(chunk.key)
        with pytest.raises(KeyError):
            await meta_api.get_chunk_meta(chunk.key)

        await MockClusterAPI.cleanup(pool.external_address)


@pytest.mark.asyncio
async def test_worker_meta_api():
    supervisor_pool = await mo.create_actor_pool("127.0.0.1", n_process=0)
    worker_pool = await mo.create_actor_pool("127.0.0.1", n_process=0)

    async with supervisor_pool, worker_pool:
        config = {
            "services": ["cluster", "session", "meta", "web"],
            "cluster": {
                "backend": "fixed",
                "lookup_address": supervisor_pool.external_address,
            },
            "meta": {"store": "dict"},
        }
        await start_services(
            NodeRole.SUPERVISOR, config, address=supervisor_pool.external_address
        )
        await start_services(
            NodeRole.WORKER, config, address=worker_pool.external_address
        )

        session_id = "test_session"
        session_api = await SessionAPI.create(supervisor_pool.external_address)
        await session_api.create_session(session_id)

        worker_meta_api = await WorkerMetaAPI.create(
            session_id=session_id, address=worker_pool.external_address
        )
        await worker_meta_api.set_tileable_meta(t)
        meta = await worker_meta_api.get_tileable_meta(t.key, fields=["nsplits"])
        assert meta["nsplits"] == t.nsplits
        await worker_meta_api.del_tileable_meta(t.key)
        with pytest.raises(KeyError):
            await worker_meta_api.get_tileable_meta(t.key)

        await stop_services(
            NodeRole.WORKER, config, address=worker_pool.external_address
        )
        await stop_services(
            NodeRole.SUPERVISOR, config, address=supervisor_pool.external_address
        )


@pytest.mark.asyncio
async def test_meta_web_api():
    pool = await mo.create_actor_pool("127.0.0.1", n_process=0)
    web_port = get_next_port()

    async with pool:
        config = {
            "services": ["cluster", "session", "meta", "web"],
            "cluster": {
                "backend": "fixed",
                "lookup_address": pool.external_address,
            },
            "meta": {"store": "dict"},
            "web": {
                "port": web_port,
            },
        }
        await start_services(NodeRole.SUPERVISOR, config, address=pool.external_address)

        session_id = "test_session"
        session_api = await SessionAPI.create(pool.external_address)
        await session_api.create_session(session_id)

        t = mt.random.rand(10, 10)
        t = tile(t)

        meta_api = await MetaAPI.create(session_id, pool.external_address)
        web_api = WebMetaAPI(session_id, f"http://localhost:{web_port}")

        await meta_api.set_chunk_meta(
            t.chunks[0], bands=[(pool.external_address, "numa-0")]
        )
        meta = await web_api.get_chunk_meta(t.chunks[0].key, fields=["shape", "bands"])
        assert set(meta.keys()) == {"shape", "bands"}

        with pytest.raises(KeyError):
            await web_api.get_chunk_meta("non-exist-key")

        await stop_services(NodeRole.SUPERVISOR, config, address=pool.external_address)
