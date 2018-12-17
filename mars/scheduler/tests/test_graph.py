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

import uuid
import unittest

import gevent

import mars.tensor as mt
from mars.cluster_info import ClusterInfoActor
from mars.scheduler import GraphActor, ResourceActor, ChunkMetaActor, AssignerActor
from mars.utils import serialize_graph, get_next_port
from mars.actors import create_actor_pool


class Test(unittest.TestCase):

    def testGraphActor(self):
        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())

        arr = mt.random.randint(10, size=(10, 8), chunks=4)
        arr_add = mt.random.randint(10, size=(10, 8), chunks=4)
        arr2 = arr + arr_add

        graph = arr2.build_graph(compose=False)
        serialized_graph = serialize_graph(graph)
        chunked_graph = arr2.build_graph(compose=False, tiled=True)

        addr = '127.0.0.1:%d' % get_next_port()
        with create_actor_pool(n_process=1, backend='gevent', address=addr) as pool:
            pool.create_actor(ClusterInfoActor, [pool.cluster_info.address],
                              uid=ClusterInfoActor.default_name())
            resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_name())
            pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_name())
            pool.create_actor(AssignerActor, uid=AssignerActor.gen_name(session_id))
            graph_ref = pool.create_actor(GraphActor, session_id, graph_key, serialized_graph,
                                          uid=GraphActor.gen_name(session_id, graph_key))

            graph_ref.prepare_graph(compose=False)
            fetched_graph = graph_ref.get_chunk_graph()
            self.assertIsNotNone(fetched_graph)
            self.assertEqual(len(chunked_graph), len(fetched_graph))

            graph_ref.scan_node()
            op_infos = graph_ref.get_operand_info()
            for n in fetched_graph:
                depth = op_infos[n.op.key]['optimize']['depth']
                self.assertIsNotNone(depth)
                successor_size = op_infos[n.op.key]['optimize']['successor_size']
                self.assertIsNotNone(successor_size)
                descendant_size = op_infos[n.op.key]['optimize']['descendant_size']
                self.assertIsNotNone(descendant_size)

            def write_mock_meta():
                resource_ref.set_worker_meta('localhost:12345', dict(hardware=dict(cpu_total=4)))
                resource_ref.set_worker_meta('localhost:23456', dict(hardware=dict(cpu_total=4)))

            v = gevent.spawn(write_mock_meta)
            v.join()

            graph_ref.place_initial_chunks()
            op_infos = graph_ref.get_operand_info()

            for n in fetched_graph:
                if fetched_graph.count_predecessors(n) != 0:
                    continue
                target_worker = op_infos[n.op.key]['target_worker']
                self.assertIsNotNone(target_worker)

            graph_ref.create_operand_actors(_clean_io_meta=False)
            op_infos = graph_ref.get_operand_info()

            for n in fetched_graph:
                self.assertEqual(op_infos[n.op.key]['op_name'], type(n.op).__name__)

                io_meta = op_infos[n.op.key]['io_meta']
                orig_io_meta = dict(
                    predecessors=list(set(pn.op.key for pn in fetched_graph.iter_predecessors(n))),
                    successors=list(set(sn.op.key for sn in fetched_graph.iter_successors(n))),
                    input_chunks=list(set(pn.key for pn in fetched_graph.iter_predecessors(n))),
                    chunks=list(c.key for c in n.op.outputs),
                )
                self.assertSetEqual(set(io_meta['predecessors']), set(orig_io_meta['predecessors']))
                self.assertSetEqual(set(io_meta['successors']), set(orig_io_meta['successors']))
                self.assertSetEqual(set(io_meta['input_chunks']), set(orig_io_meta['input_chunks']))
                self.assertSetEqual(set(io_meta['chunks']), set(orig_io_meta['chunks']))

                self.assertEqual(op_infos[n.op.key]['output_size'], sum(ch.nbytes for ch in n.op.outputs))

    def testGraphWithSplit(self):
        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())

        arr = mt.ones(12, chunks=4)
        arr_split = mt.split(arr, 2)
        arr_sum = arr_split[0] + arr_split[1]

        graph = arr_sum.build_graph(compose=False)
        serialized_graph = serialize_graph(graph)
        chunked_graph = arr_sum.build_graph(compose=False, tiled=True)

        addr = '127.0.0.1:%d' % get_next_port()
        with create_actor_pool(n_process=1, backend='gevent', address=addr) as pool:
            pool.create_actor(ClusterInfoActor, [pool.cluster_info.address],
                              uid=ClusterInfoActor.default_name())
            resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_name())
            pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_name())
            pool.create_actor(AssignerActor, uid=AssignerActor.gen_name(session_id))
            graph_ref = pool.create_actor(GraphActor, session_id, graph_key, serialized_graph,
                                          uid=GraphActor.gen_name(session_id, graph_key))

            graph_ref.prepare_graph(compose=False)
            fetched_graph = graph_ref.get_chunk_graph()
            self.assertIsNotNone(fetched_graph)
            self.assertEqual(len(chunked_graph), len(fetched_graph))

            graph_ref.scan_node()
            op_infos = graph_ref.get_operand_info()
            for n in fetched_graph:
                depth = op_infos[n.op.key]['optimize']['depth']
                self.assertIsNotNone(depth)
                successor_size = op_infos[n.op.key]['optimize']['successor_size']
                self.assertIsNotNone(successor_size)
                descendant_size = op_infos[n.op.key]['optimize']['descendant_size']
                self.assertIsNotNone(descendant_size)

            def write_mock_meta():
                resource_ref.set_worker_meta('localhost:12345', dict(hardware=dict(cpu_total=4)))
                resource_ref.set_worker_meta('localhost:23456', dict(hardware=dict(cpu_total=4)))

            v = gevent.spawn(write_mock_meta)
            v.join()

            graph_ref.place_initial_chunks()
            op_infos = graph_ref.get_operand_info()

            for n in fetched_graph:
                if fetched_graph.count_predecessors(n) != 0:
                    continue
                target_worker = op_infos[n.op.key]['target_worker']
                self.assertIsNotNone(target_worker)

            graph_ref.create_operand_actors(_clean_io_meta=False)
            op_infos = graph_ref.get_operand_info()

            for n in fetched_graph:
                self.assertEqual(op_infos[n.op.key]['op_name'], type(n.op).__name__)

                io_meta = op_infos[n.op.key]['io_meta']
                orig_io_meta = dict(
                    predecessors=list(set(pn.op.key for pn in fetched_graph.iter_predecessors(n))),
                    successors=list(set(sn.op.key for sn in fetched_graph.iter_successors(n))),
                    input_chunks=list(set(pn.key for pn in fetched_graph.iter_predecessors(n))),
                    chunks=list(c.key for c in n.op.outputs),
                )
                self.assertSetEqual(set(io_meta['predecessors']), set(orig_io_meta['predecessors']))
                self.assertSetEqual(set(io_meta['successors']), set(orig_io_meta['successors']))
                self.assertSetEqual(set(io_meta['input_chunks']), set(orig_io_meta['input_chunks']))
                self.assertSetEqual(set(io_meta['chunks']), set(orig_io_meta['chunks']))

                self.assertEqual(op_infos[n.op.key]['output_size'], sum(ch.nbytes for ch in n.op.outputs))

    def testSameKey(self, *_):
        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())

        arr = mt.ones((5, 5), chunks=3)
        arr2 = mt.concatenate((arr, arr))

        graph = arr2.build_graph(compose=False)
        serialized_graph = serialize_graph(graph)
        chunked_graph = arr2.build_graph(compose=False, tiled=True)

        addr = '127.0.0.1:%d' % get_next_port()
        with create_actor_pool(n_process=1, backend='gevent', address=addr) as pool:
            pool.create_actor(ClusterInfoActor, [pool.cluster_info.address],
                              uid=ClusterInfoActor.default_name())
            resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_name())
            pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_name())
            pool.create_actor(AssignerActor, uid=AssignerActor.gen_name(session_id))
            graph_ref = pool.create_actor(GraphActor, session_id, graph_key, serialized_graph,
                                          uid=GraphActor.gen_name(session_id, graph_key))

            graph_ref.prepare_graph(compose=False)
            fetched_graph = graph_ref.get_chunk_graph()
            self.assertIsNotNone(fetched_graph)
            self.assertEqual(len(chunked_graph), len(fetched_graph))

            graph_ref.scan_node()
            op_infos = graph_ref.get_operand_info()
            for n in fetched_graph:
                depth = op_infos[n.op.key]['optimize']['depth']
                self.assertIsNotNone(depth)
                successor_size = op_infos[n.op.key]['optimize']['successor_size']
                self.assertIsNotNone(successor_size)
                descendant_size = op_infos[n.op.key]['optimize']['descendant_size']
                self.assertIsNotNone(descendant_size)

            def write_mock_meta():
                resource_ref.set_worker_meta('localhost:12345', dict(hardware=dict(cpu_total=4)))
                resource_ref.set_worker_meta('localhost:23456', dict(hardware=dict(cpu_total=4)))

            v = gevent.spawn(write_mock_meta)
            v.join()

            graph_ref.place_initial_chunks()
            op_infos = graph_ref.get_operand_info()

            for n in fetched_graph:
                if fetched_graph.count_predecessors(n) != 0:
                    continue
                target_worker = op_infos[n.op.key]['target_worker']
                self.assertIsNotNone(target_worker)

            graph_ref.create_operand_actors(_clean_io_meta=False)
            op_infos = graph_ref.get_operand_info()

            for n in fetched_graph:
                self.assertEqual(op_infos[n.op.key]['op_name'], type(n.op).__name__)

                io_meta = op_infos[n.op.key]['io_meta']
                orig_io_meta = dict(
                    predecessors=list(set(pn.op.key for pn in fetched_graph.iter_predecessors(n))),
                    successors=list(set(sn.op.key for sn in fetched_graph.iter_successors(n))),
                    input_chunks=list(set(pn.key for pn in fetched_graph.iter_predecessors(n))),
                    chunks=list(c.key for c in n.op.outputs),
                )
                self.assertSetEqual(set(io_meta['predecessors']), set(orig_io_meta['predecessors']))
                self.assertSetEqual(set(io_meta['successors']), set(orig_io_meta['successors']))
                self.assertSetEqual(set(io_meta['input_chunks']), set(orig_io_meta['input_chunks']))
                self.assertSetEqual(set(io_meta['chunks']), set(orig_io_meta['chunks']))

                self.assertEqual(op_infos[n.op.key]['output_size'], sum(ch.nbytes for ch in n.op.outputs))
