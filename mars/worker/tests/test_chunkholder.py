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
import numpy as np
from functools import partial

import gevent

from mars.actors import create_actor_pool
from mars.compat import six, mock
from mars.config import options
from mars.utils import get_next_port, calc_data_size
from mars import promise
from mars.errors import StoreFull, SpillExhausted
from mars.cluster_info import ClusterInfoActor
from mars.scheduler.kvstore import KVStoreActor
from mars.worker.tests.base import WorkerCase
from mars.worker import *
from mars.worker.utils import WorkerActor


class CacheTestActor(WorkerActor):
    def __init__(self):
        super(CacheTestActor, self).__init__()
        self._exc_info = None
        self._finished = False

    def run_test_cache(self):
        session_id = str(uuid.uuid4())

        chunk_holder_ref = self.promise_ref('ChunkHolderActor')
        chunk_store = self._chunk_store

        data_list = []
        for _ in range(9):
            data_id = str(uuid.uuid4())
            data = np.random.randint(0, 32767, (655360,), np.int16)
            data_list.append((data_id, data))

        def _put_chunk(data_key, data, *_):
            def _handle_reject(*exc):
                if issubclass(exc[0], SpillExhausted):
                    return
                six.reraise(*exc)

            try:
                ref = chunk_store.put(session_id, data_key, data)
                chunk_holder_ref.register_chunk(session_id, data_key)
                gevent.sleep(0.5)
                del ref
            except StoreFull:
                return chunk_holder_ref.spill_size(calc_data_size(data) * 2, _promise=True) \
                    .then(partial(_put_chunk, data_key, data), _handle_reject)

        data_promises = []
        for data_id, data in data_list:
            data_promises.append(promise.Promise(done=True) \
                                 .then(partial(_put_chunk, data_id, data)))

        def assert_true(v):
            assert v

        last_id = data_list[-1][0]
        p = promise.all_(data_promises) \
            .then(lambda *_: assert_true(chunk_store.contains(session_id, last_id))) \
            .then(lambda *_: ensure_chunk(self, session_id, last_id)) \
            .then(lambda *_: assert_true(chunk_store.contains(session_id, last_id)))

        first_id = data_list[0][0]
        p = p.then(lambda *_: assert_true(not chunk_store.contains(session_id, first_id))) \
            .then(lambda *_: ensure_chunk(self, session_id, first_id)) \
            .then(lambda *_: assert_true(chunk_store.contains(session_id, first_id)))

        p = p.then(lambda *_: chunk_holder_ref.unregister_chunk(session_id, first_id)) \
            .then(lambda *_: self._plasma_client.evict(128)) \
            .then(lambda *_: assert_true(not chunk_store.contains(session_id, first_id)))

        p = p.then(lambda *_: chunk_holder_ref.unregister_chunk(session_id, last_id)) \
            .then(lambda *_: self._plasma_client.evict(128)) \
            .then(lambda *_: assert_true(not chunk_store.contains(session_id, last_id)))

        p.catch(lambda *exc: setattr(self, '_exc_info', exc)) \
            .then(lambda *_: setattr(self, '_finished', True))

    def run_test_ensure_timeout(self):
        from mars.worker.chunkholder import ensure_chunk
        promises = [
            ensure_chunk(self, str(uuid.uuid4()), str(uuid.uuid4())),
            ensure_chunk(self, str(uuid.uuid4()), str(uuid.uuid4())),
        ]
        promise.all_(promises) \
            .then(lambda *_: setattr(self, '_exc_info', None), lambda *exc: setattr(self, '_exc_info', exc)) \
            .then(lambda *_: setattr(self, '_finished', True))

    def get_exc_info(self):
        return self._finished, self._exc_info


class Test(WorkerCase):
    def setUp(self):
        import logging
        logging.basicConfig(level=logging.DEBUG)
        options.worker.min_spill_size = 0

    def tearDown(self):
        options.worker.min_spill_size = '5%'

    def testHolder(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        with create_actor_pool(n_process=1, backend='gevent', address=pool_address) as pool:
            pool.create_actor(ClusterInfoActor, schedulers=[pool_address],
                              uid=ClusterInfoActor.default_name())
            pool.create_actor(KVStoreActor, uid=KVStoreActor.default_name())
            pool.create_actor(DispatchActor, uid='DispatchActor')
            pool.create_actor(QuotaActor, 1024 * 1024 * 10, uid='MemQuotaActor')
            cache_ref = pool.create_actor(ChunkHolderActor, self._plasma_helper._size, uid='ChunkHolderActor')
            pool.create_actor(SpillActor)

            try:
                test_ref = pool.create_actor(CacheTestActor)
                test_ref.run_test_cache()
                while not test_ref.get_exc_info()[0]:
                    gevent.sleep(0.1)
                exc_info = test_ref.get_exc_info()[1]
                if exc_info:
                    six.reraise(*exc_info)
            finally:
                pool.destroy_actor(cache_ref)

    @mock.patch(SpillActor.__module__ + '.SpillActor.load')
    def testEnsureTimeout(self, *_):
        from mars.errors import PromiseTimeout

        pool_address = '127.0.0.1:%d' % get_next_port()
        with create_actor_pool(n_process=1, backend='gevent', address=pool_address) as pool:
            pool.create_actor(ClusterInfoActor, schedulers=[pool_address],
                              uid=ClusterInfoActor.default_name())
            pool.create_actor(KVStoreActor, uid='KVStoreActor')
            pool.create_actor(DispatchActor, uid='DispatchActor')
            pool.create_actor(QuotaActor, 1024 * 1024 * 10, uid='MemQuotaActor')
            pool.create_actor(SpillActor)
            cache_ref = pool.create_actor(ChunkHolderActor, self._plasma_helper._size, uid='ChunkHolderActor')

            try:
                options.worker.prepare_data_timeout = 2
                test_ref = pool.create_actor(CacheTestActor)
                test_ref.run_test_ensure_timeout()
                while not test_ref.get_exc_info()[0]:
                    gevent.sleep(0.1)
                exc_info = test_ref.get_exc_info()[1]
                self.assertIsNotNone(exc_info)
                self.assertIsInstance(exc_info[1], PromiseTimeout)
            finally:
                options.worker.prepare_data_timeout = 600
                pool.destroy_actor(cache_ref)
