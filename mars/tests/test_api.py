#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import unittest
from collections import OrderedDict

from mars.utils import get_next_port
from mars.scheduler import GraphActor, ResourceActor, SessionManagerActor,\
    GraphState, ChunkMetaClient, ChunkMetaActor
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.api import MarsAPI
from mars.tests.core import aio_case, patch_method, create_actor_pool


@unittest.skipIf(sys.platform == 'win32', 'does not run under windows')
@aio_case
class Test(unittest.TestCase):
    @staticmethod
    def _create_pool():
        pool = None

        class _AsyncContextManager:
            async def __aenter__(self):
                nonlocal pool
                endpoint = '127.0.0.1:%d' % get_next_port()
                pool = create_actor_pool(n_process=1, address=endpoint)
                await pool.__aenter__()

                await pool.create_actor(SchedulerClusterInfoActor, [endpoint],
                                        uid=SchedulerClusterInfoActor.default_uid())
                await pool.create_actor(SessionManagerActor, uid=SessionManagerActor.default_uid())
                await pool.create_actor(ResourceActor, uid=ResourceActor.default_uid())

                api = MarsAPI(endpoint)
                return pool, api

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await pool.__aexit__(exc_type, exc_val, exc_tb)

        return _AsyncContextManager()

    async def testApi(self, *_):
        async with self._create_pool() as (pool, api), \
                patch_method(GraphActor.execute_graph, new_async=True):
            self.assertEqual(0, await api.count_workers())

            session_manager = await api.get_session_manager()

            session_id = 'mock_session_id'
            await api.create_session(session_id)
            self.assertEqual(1, len(await session_manager.get_sessions()))
            await api.delete_session(session_id)
            self.assertEqual(0, len(await session_manager.get_sessions()))
            await api.create_session(session_id)

            serialized_graph = 'mock_serialized_graph'
            graph_key = 'mock_graph_key'
            targets = 'mock_targets'
            await api.submit_graph(session_id, serialized_graph, graph_key, targets)
            graph_uid = GraphActor.gen_uid(session_id, graph_key)
            graph_ref = await api.get_actor_ref(graph_uid)
            self.assertTrue(await pool.has_actor(graph_ref))

            state = await api.get_graph_state(session_id, graph_key)
            self.assertEqual(GraphState('preparing'), state)
            exc_info = await api.get_graph_exc_info(session_id, graph_key)
            self.assertIsNone(exc_info)

            await api.stop_graph(session_id, graph_key)
            state = await api.get_graph_state(session_id, graph_key)
            self.assertEqual(GraphState('cancelled'), state)
            exc_info = await api.get_graph_exc_info(session_id, graph_key)
            self.assertIsNone(exc_info)

            await api.delete_graph(session_id, graph_key)
            self.assertFalse(await pool.has_actor(graph_ref))

    async def testGetTensorNsplits(self, *_):
        session_id = 'mock_session_id'
        graph_key = 'mock_graph_key'
        tensor_key = 'mock_tensor_key'
        serialized_graph = 'mock_serialized_graph'

        async with self._create_pool() as (pool, api), \
                patch_method(GraphActor.get_tileable_metas), \
                patch_method(ChunkMetaClient.batch_get_chunk_shape):

            graph_uid = GraphActor.gen_uid(session_id, graph_key)
            await pool.create_actor(GraphActor, session_id, serialized_graph, graph_key, uid=graph_uid)
            await pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())

            mock_indexes = [
                [(((3, 4, 5, 6),), OrderedDict(zip([(0, ), (1,), (2,), (3,)],
                                                   ['chunk_key1', 'chunk_key2', 'chunk_key3', 'chunk_key4'])))],
                [(((3, 2), (4, 2)), OrderedDict(zip([(0, 0), (0, 1), (1, 0), (1, 1)],
                                                    ['chunk_key1', 'chunk_key2', 'chunk_key3', 'chunk_key4'])))]
            ]
            mock_shapes = [
                [(3,), (4,), (5,), (6,)],
                [(3, 4), (3, 2), (2, 4), (2, 2)]
            ]

            GraphActor.get_tileable_metas.side_effect = mock_indexes
            ChunkMetaClient.batch_get_chunk_shape.side_effect = mock_shapes

            nsplits = await api.get_tileable_nsplits(session_id, graph_key, tensor_key)
            self.assertEqual(((3, 4, 5, 6),), nsplits)

            nsplits = await api.get_tileable_nsplits(session_id, graph_key, tensor_key)
            self.assertEqual(((3, 2), (4, 2)), nsplits)
