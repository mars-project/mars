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
import gevent
from numpy.testing import assert_array_equal

from mars.actors import create_actor_pool
from mars.compat import six
from mars.utils import get_next_port, serialize_graph
from mars.cluster_info import ClusterInfoActor
from mars.scheduler import ChunkMetaActor
from mars.worker.tests.base import WorkerCase
from mars.worker import *
from mars.worker.chunkstore import PlasmaKeyMapActor
from mars.worker.utils import WorkerActor


class ExecuteTestActor(WorkerActor):
    def __init__(self):
        super(ExecuteTestActor, self).__init__()
        self._exc_info = None
        self._finished = False

    def run_test(self):
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

        chunk_holder_ref = self.promise_ref(ChunkHolderActor.default_name())

        refs = self._chunk_store.put(session_id, arr.chunks[0].key, np.ones((10, 8), dtype=np.int16))
        chunk_holder_ref.register_chunk(session_id, arr.chunks[0].key)
        del refs

        refs = self._chunk_store.put(session_id, arr_add.chunks[0].key, np.ones((10, 8), dtype=np.int16))
        chunk_holder_ref.register_chunk(session_id, arr_add.chunks[0].key)
        del refs

        executor_ref = self.promise_ref(ExecutionActor.default_name())

        def _validate(_):
            data = self._chunk_store.get(session_id, arr2.chunks[0].key)
            assert_array_equal(data, 2 * np.ones((10, 8)))

        executor_ref.execute_graph(session_id, str(id(graph)), serialize_graph(graph),
                                   dict(chunks=[arr2.chunks[0].key]), None, _promise=True) \
            .then(_validate) \
            .catch(lambda *exc: setattr(self, '_exc_info', exc)) \
            .then(lambda *_: setattr(self, '_finished', True))

    def get_exc_info(self):
        return self._finished, self._exc_info


class Test(WorkerCase):
    def testExecute(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        with create_actor_pool(n_process=1, backend='gevent', address=pool_address) as pool:
            pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_name())
            pool.create_actor(ClusterInfoActor, schedulers=[pool_address],
                              uid=ClusterInfoActor.default_name())
            cache_ref = pool.create_actor(ChunkHolderActor, self.plasma_storage_size,
                                          uid=ChunkHolderActor.default_name())
            pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_name())
            pool.create_actor(DispatchActor, uid=DispatchActor.default_name())
            pool.create_actor(QuotaActor, 1024 * 1024, uid=MemQuotaActor.default_name())
            pool.create_actor(CpuCalcActor)
            pool.create_actor(ExecutionActor, uid=ExecutionActor.default_name())

            try:
                test_ref = pool.create_actor(ExecuteTestActor)
                test_ref.run_test()
                while not test_ref.get_exc_info()[0]:
                    gevent.sleep(0.1)
                exc_info = test_ref.get_exc_info()[1]
                if exc_info:
                    six.reraise(*exc_info)
            finally:
                pool.destroy_actor(cache_ref)
