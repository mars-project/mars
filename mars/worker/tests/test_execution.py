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
import weakref

import numpy as np
from numpy.testing import assert_array_equal

from mars.actors import create_actor_pool
from mars.utils import get_next_port, serialize_graph
from mars.cluster_info import ClusterInfoActor
from mars.scheduler import ChunkMetaActor
from mars.worker.tests.base import WorkerCase
from mars.worker import *


class Test(WorkerCase):
    def testExecute(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        with create_actor_pool(n_process=1, backend='gevent', address=pool_address) as pool:
            pool.create_actor(ClusterInfoActor, schedulers=[pool_address],
                              uid=ClusterInfoActor.default_name())
            cache_ref = pool.create_actor(
                ChunkHolderActor, self.plasma_storage_size, uid=ChunkHolderActor.default_name())
            pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_name())
            pool.create_actor(TaskQueueActor, uid=TaskQueueActor.default_name())
            pool.create_actor(DispatchActor, uid=DispatchActor.default_name())
            pool.create_actor(QuotaActor, 1024 * 1024, uid=MemQuotaActor.default_name())
            pool.create_actor(CpuCalcActor)
            pool.create_actor(ExecutionActor, uid=ExecutionActor.default_name())

            try:
                with self.run_actor_test(pool) as test_actor:
                    import mars.tensor as mt
                    from mars.tensor.expressions.datasource import TensorOnes, TensorFetchChunk
                    arr = mt.ones((10, 8), chunk_size=10)
                    arr_add = mt.ones((10, 8), chunk_size=10)
                    arr2 = arr + arr_add
                    graph = arr2.build_graph(compose=False, tiled=True)

                    for chunk in graph:
                        if isinstance(chunk.op, TensorOnes):
                            chunk._op = TensorFetchChunk(
                                dtype=chunk.dtype, _outputs=[weakref.ref(o) for o in chunk.op.outputs],
                                _key=chunk.op.key)

                    session_id = str(uuid.uuid4())
                    chunk_holder_ref = test_actor.promise_ref(ChunkHolderActor.default_name())

                    refs = test_actor._chunk_store.put(session_id, arr.chunks[0].key,
                                                       np.ones((10, 8), dtype=np.int16))
                    chunk_holder_ref.register_chunk(session_id, arr.chunks[0].key)
                    del refs

                    refs = test_actor._chunk_store.put(session_id, arr_add.chunks[0].key,
                                                       np.ones((10, 8), dtype=np.int16))
                    chunk_holder_ref.register_chunk(session_id, arr_add.chunks[0].key)
                    del refs

                    executor_ref = test_actor.promise_ref(ExecutionActor.default_name())

                    def _validate(_):
                        data = test_actor._chunk_store.get(session_id, arr2.chunks[0].key)
                        assert_array_equal(data, 2 * np.ones((10, 8)))

                    executor_ref.enqueue_graph(session_id, str(id(graph)), serialize_graph(graph),
                                               dict(chunks=[arr2.chunks[0].key]), None, _promise=True) \
                        .then(lambda *_: executor_ref.start_execution(session_id, str(id(graph)), _promise=True)) \
                        .then(_validate) \
                        .then(lambda *_: test_actor.set_result(None)) \
                        .catch(lambda *exc: test_actor.set_result(exc, False))

                self.get_result()
            finally:
                pool.destroy_actor(cache_ref)
