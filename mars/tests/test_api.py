#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import unittest

from mars.compat import OrderedDict
from mars.actors import create_actor_pool
from mars.utils import get_next_port
from mars.scheduler import GraphActor, ResourceActor, SessionManagerActor,\
    GraphState, ChunkMetaClient, ChunkMetaActor
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.api import MarsAPI
from mars.tests.core import patch_method


class Test(unittest.TestCase):
    def setUp(self):
        endpoint = '127.0.0.1:%d' % get_next_port()
        self.endpoint = endpoint
        self.pool = create_actor_pool(n_process=1, backend='gevent', address=endpoint)
        self.pool.create_actor(SchedulerClusterInfoActor, [endpoint],
                               uid=SchedulerClusterInfoActor.default_uid())
        self.pool.create_actor(SessionManagerActor, uid=SessionManagerActor.default_uid())
        self.pool.create_actor(ResourceActor, uid=ResourceActor.default_uid())

        self.api = MarsAPI(endpoint)

    def tearDown(self):
        self.pool.stop()

    @patch_method(GraphActor.execute_graph)
    def testApi(self, *_):
        self.assertEqual(0, self.api.count_workers())

        session_id = 'mock_session_id'
        self.api.create_session(session_id)
        self.assertEqual(1, len(self.api.session_manager.get_sessions()))
        self.api.delete_session(session_id)
        self.assertEqual(0, len(self.api.session_manager.get_sessions()))
        self.api.create_session(session_id)

        serialized_graph = 'mock_serialized_graph'
        graph_key = 'mock_graph_key'
        targets = 'mock_targets'
        self.api.submit_graph(session_id, serialized_graph, graph_key, targets)
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = self.api.get_actor_ref(graph_uid)
        self.assertTrue(self.pool.has_actor(graph_ref))

        state = self.api.get_graph_state(session_id, graph_key)
        self.assertEqual(GraphState('preparing'), state)
        exc_info = self.api.get_graph_exc_info(session_id, graph_key)
        self.assertIsNone(exc_info)

        self.api.stop_graph(session_id, graph_key)
        state = self.api.get_graph_state(session_id, graph_key)
        self.assertEqual(GraphState('cancelled'), state)
        exc_info = self.api.get_graph_exc_info(session_id, graph_key)
        self.assertIsNone(exc_info)

        self.api.delete_graph(session_id, graph_key)
        self.assertFalse(self.pool.has_actor(graph_ref))

    @patch_method(GraphActor.get_tileable_meta)
    @patch_method(ChunkMetaClient.batch_get_chunk_shape)
    def testGetTensorNsplits(self, *_):
        session_id = 'mock_session_id'
        graph_key = 'mock_graph_key'
        tensor_key = 'mock_tensor_key'
        serialized_graph = 'mock_serialized_graph'

        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        self.pool.create_actor(GraphActor, session_id, serialized_graph, graph_key, uid=graph_uid)
        self.pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())

        mock_indexes = [
            (((3, 4, 5, 6),), OrderedDict(zip([(0, ), (1,), (2,), (3,)],
                                              ['chunk_key1', 'chunk_key2', 'chunk_key3', 'chunk_key4']))),
            (((3, 2), (4, 2)), OrderedDict(zip([(0, 0), (0, 1), (1, 0), (1, 1)],
                                               ['chunk_key1', 'chunk_key2', 'chunk_key3', 'chunk_key4'])))
        ]
        mock_shapes = [
            [(3,), (4,), (5,), (6,)],
            [(3, 4), (3, 2), (2, 4), (2, 2)]
        ]

        GraphActor.get_tileable_meta.side_effect = mock_indexes
        ChunkMetaClient.batch_get_chunk_shape.side_effect = mock_shapes

        nsplits = self.api.get_tileable_nsplits(session_id, graph_key, tensor_key)
        self.assertEqual(((3, 4, 5, 6),), nsplits)

        nsplits = self.api.get_tileable_nsplits(session_id, graph_key, tensor_key)
        self.assertEqual(((3, 2), (4, 2)), nsplits)
