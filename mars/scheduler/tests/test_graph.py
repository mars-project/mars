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

import uuid
import unittest

import mars.tensor as mt
from mars.scheduler import GraphActor, GraphMetaActor, ResourceActor, ChunkMetaActor, \
    AssignerActor, GraphState, OperandState
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.tests.core import aio_case
from mars.utils import serialize_graph, get_next_port
from mars.graph import DAG
from mars.tests.core import patch_method, create_actor_pool


@aio_case
class Test(unittest.TestCase):
    def prepare_graph_in_pool(self, expr, clean_io_meta=True, compose=False):
        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())

        graph = expr.build_graph(compose=compose)
        serialized_graph = serialize_graph(graph)
        chunked_graph = expr.build_graph(compose=compose, tiled=True)

        addr = '127.0.0.1:%d' % get_next_port()
        pool_ctx = create_actor_pool(n_process=1, address=addr)

        this = self

        async def _empty_patch(*_, **__):
            pass

        patch_ctx = patch_method(ResourceActor._broadcast_sessions, _empty_patch)

        class _AsyncContextManager:
            async def __aenter__(self):
                patch_ctx.__enter__()
                pool = await pool_ctx.__aenter__()
                await pool.create_actor(SchedulerClusterInfoActor, [pool.cluster_info.address],
                                        uid=SchedulerClusterInfoActor.default_uid())
                resource_ref = await pool.create_actor(ResourceActor, uid=ResourceActor.default_uid())
                await pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())
                await pool.create_actor(AssignerActor, uid=AssignerActor.gen_uid(session_id))
                graph_ref = await pool.create_actor(GraphActor, session_id, graph_key, serialized_graph,
                                                    uid=GraphActor.gen_uid(session_id, graph_key))

                await graph_ref.prepare_graph(compose=compose)
                fetched_graph = await graph_ref.get_chunk_graph()
                this.assertIsNotNone(fetched_graph)
                this.assertEqual(len(chunked_graph), len(fetched_graph))

                await graph_ref.analyze_graph(do_placement=False)
                op_infos = await graph_ref.get_operand_info()
                for n in fetched_graph:
                    depth = op_infos[n.op.key]['optimize']['depth']
                    this.assertIsNotNone(depth)
                    successor_size = op_infos[n.op.key]['optimize']['successor_size']
                    this.assertIsNotNone(successor_size)
                    descendant_size = op_infos[n.op.key]['optimize']['descendant_size']
                    this.assertIsNotNone(descendant_size)

                await resource_ref.set_worker_meta('localhost:12345', dict(hardware=dict(cpu_total=4)))
                await resource_ref.set_worker_meta('localhost:23456', dict(hardware=dict(cpu_total=4)))

                await graph_ref.analyze_graph()
                op_infos = await graph_ref.get_operand_info()

                for n in fetched_graph:
                    if fetched_graph.count_predecessors(n) != 0:
                        continue
                    target_worker = op_infos[n.op.key]['target_worker']
                    this.assertIsNotNone(target_worker)

                await graph_ref.create_operand_actors(_clean_info=clean_io_meta)
                op_infos = await graph_ref.get_operand_info()

                if not clean_io_meta:
                    orig_metas = dict()
                    for n in fetched_graph:
                        try:
                            meta = orig_metas[n.op.key]
                        except KeyError:
                            meta = orig_metas[n.op.key] = dict(
                                predecessors=set(), successors=set(), input_chunks=set(), chunks=set()
                            )
                        meta['predecessors'].update([pn.op.key for pn in fetched_graph.iter_predecessors(n)])
                        meta['successors'].update([sn.op.key for sn in fetched_graph.iter_successors(n)])
                        meta['input_chunks'].update([pn.key for pn in fetched_graph.iter_predecessors(n)])
                        meta['chunks'].update([c.key for c in n.op.outputs])

                    for n in fetched_graph:
                        this.assertEqual(op_infos[n.op.key]['op_name'], type(n.op).__name__)

                        io_meta = op_infos[n.op.key]['io_meta']
                        orig_io_meta = orig_metas[n.op.key]

                        this.assertSetEqual(set(io_meta['predecessors']), set(orig_io_meta['predecessors']))
                        this.assertSetEqual(set(io_meta['successors']), set(orig_io_meta['successors']))
                        this.assertSetEqual(set(io_meta['input_chunks']), set(orig_io_meta['input_chunks']))
                        this.assertSetEqual(set(io_meta['chunks']), set(orig_io_meta['chunks']))
                return pool, graph_ref

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await pool_ctx.__aexit__(exc_type, exc_val, exc_tb)
                patch_ctx.__exit__(exc_type, exc_val, exc_tb)

        return _AsyncContextManager()

    async def testSimpleGraphPreparation(self, *_):
        arr = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr_add = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr2 = arr + arr_add
        async with self.prepare_graph_in_pool(arr2, clean_io_meta=False):
            pass

    async def testSplitPreparation(self, *_):
        arr = mt.ones(12, chunk_size=4)
        arr_split = mt.split(arr, 2)
        arr_sum = arr_split[0] + arr_split[1]
        async with self.prepare_graph_in_pool(arr_sum, clean_io_meta=False):
            pass

    async def testSameKeyPreparation(self, *_):
        arr = mt.ones((5, 5), chunk_size=3)
        arr2 = mt.concatenate((arr, arr))
        async with self.prepare_graph_in_pool(arr2, clean_io_meta=False):
            pass

    async def testShufflePreparation(self, *_):
        a = mt.ones((31, 27), chunk_size=10)
        b = a.reshape(27, 31)
        b.op.extra_params['_reshape_with_shuffle'] = True
        async with self.prepare_graph_in_pool(b, compose=False):
            pass

    async def testFusePreparation(self, *_):
        from mars.tensor.fuse.core import TensorFuseChunk
        arr = mt.ones((5, 5), chunk_size=3)
        arr2 = (arr + 5) * 2
        async with self.prepare_graph_in_pool(arr2, compose=True) as (pool, graph_ref):
            out_graph = await graph_ref.get_chunk_graph()
            self.assertTrue(all(isinstance(v.op, TensorFuseChunk) for v in out_graph))

            self.assertEqual(len(await graph_ref.get_operand_info(state=OperandState.FREED)), 0)

    async def testMultipleAddPreparation(self, *_):
        import numpy as np
        import operator
        from functools import reduce

        base_arr = np.random.random((100, 100))
        a = mt.array(base_arr)
        sumv = reduce(operator.add, [a[:10, :10] for _ in range(10)])
        async with self.prepare_graph_in_pool(sumv):
            pass

    async def testGraphTermination(self, *_):
        from mars.tensor.arithmetic.add import TensorAdd

        arr = mt.random.random((8, 2), chunk_size=2)
        arr2 = arr + 1
        async with self.prepare_graph_in_pool(arr2) as (pool, graph_ref):
            out_graph = await graph_ref.get_chunk_graph()
            for c in out_graph:
                if not isinstance(c.op, TensorAdd):
                    continue
                self.assertNotEqual(await graph_ref.get_state(), GraphState.SUCCEEDED)
                await graph_ref.add_finished_terminal(c.op.key)

            self.assertEqual(await graph_ref.get_state(), GraphState.SUCCEEDED)

        arr = mt.random.random((8, 2), chunk_size=2)
        arr2 = arr + 1
        async with self.prepare_graph_in_pool(arr2) as (pool, graph_ref):
            out_graph = await graph_ref.get_chunk_graph()
            for c in out_graph:
                if not isinstance(c.op, TensorAdd):
                    continue
                self.assertNotEqual(await graph_ref.get_state(), GraphState.FAILED)
                await graph_ref.add_finished_terminal(c.op.key, GraphState.FAILED)

            self.assertEqual(await graph_ref.get_state(), GraphState.FAILED)

    async def testEmptyGraph(self, *_):
        session_id = str(uuid.uuid4())

        addr = '127.0.0.1:%d' % get_next_port()
        async with create_actor_pool(n_process=1, address=addr) as pool:
            await pool.create_actor(SchedulerClusterInfoActor, [pool.cluster_info.address],
                                    uid=SchedulerClusterInfoActor.default_uid())
            resource_ref = await pool.create_actor(ResourceActor, uid=ResourceActor.default_uid())
            await pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())
            await pool.create_actor(AssignerActor, uid=AssignerActor.gen_uid(session_id))

            await resource_ref.set_worker_meta('localhost:12345', dict(hardware=dict(cpu_total=4)))
            await resource_ref.set_worker_meta('localhost:23456', dict(hardware=dict(cpu_total=4)))

            graph_key = str(uuid.uuid4())
            serialized_graph = serialize_graph(DAG())

            graph_ref = await pool.create_actor(GraphActor, session_id, graph_key, serialized_graph,
                                                uid=GraphActor.gen_uid(session_id, graph_key))
            await graph_ref.execute_graph()
            self.assertEqual(await graph_ref.get_state(), GraphState.SUCCEEDED)

    async def testErrorOnPrepare(self, *_):
        session_id = str(uuid.uuid4())

        addr = '127.0.0.1:%d' % get_next_port()
        async with create_actor_pool(n_process=1, address=addr) as pool:
            await pool.create_actor(SchedulerClusterInfoActor, [pool.cluster_info.address],
                                    uid=SchedulerClusterInfoActor.default_uid())
            resource_ref = await pool.create_actor(ResourceActor, uid=ResourceActor.default_uid())
            await pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())
            await pool.create_actor(AssignerActor, uid=AssignerActor.default_uid())

            await resource_ref.set_worker_meta('localhost:12345', dict(hardware=dict(cpu_total=4)))
            await resource_ref.set_worker_meta('localhost:23456', dict(hardware=dict(cpu_total=4)))

            # error occurred in create_operand_actors
            graph_key = str(uuid.uuid4())
            expr = mt.random.random((8, 2), chunk_size=2) + 1
            graph = expr.build_graph(compose=False)
            serialized_graph = serialize_graph(graph)

            graph_ref = await pool.create_actor(GraphActor, session_id, graph_key, serialized_graph,
                                                uid=GraphActor.gen_uid(session_id, graph_key))

            async def _mock_raises(*_, **__):
                raise RuntimeError

            with patch_method(GraphActor.create_operand_actors, new=_mock_raises), \
                    self.assertRaises(RuntimeError):
                await graph_ref.execute_graph()
            self.assertEqual(await graph_ref.get_state(), GraphState.FAILED)
            await graph_ref.destroy()

            # interrupted during create_operand_actors
            graph_key = str(uuid.uuid4())
            graph_ref = await pool.create_actor(GraphActor, session_id, graph_key, serialized_graph,
                                                uid=GraphActor.gen_uid(session_id, graph_key))

            async def _mock_cancels(*_, **__):
                graph_meta_ref = pool.actor_ref(GraphMetaActor.gen_uid(session_id, graph_key))
                await graph_meta_ref.set_state(GraphState.CANCELLING)

            with patch_method(GraphActor.create_operand_actors, new=_mock_cancels):
                await graph_ref.execute_graph()
            self.assertEqual(await graph_ref.get_state(), GraphState.CANCELLED)

            # interrupted during previous steps
            graph_key = str(uuid.uuid4())
            graph_ref = await pool.create_actor(GraphActor, session_id, graph_key, serialized_graph,
                                                uid=GraphActor.gen_uid(session_id, graph_key))

            async def _mock_cancels(*_, **__):
                graph_meta_ref = pool.actor_ref(GraphMetaActor.gen_uid(session_id, graph_key))
                await graph_meta_ref.set_state(GraphState.CANCELLING)
                return dict()

            with patch_method(GraphActor._assign_initial_workers, new=_mock_cancels):
                await graph_ref.execute_graph()
            self.assertEqual(await graph_ref.get_state(), GraphState.CANCELLED)
