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

import multiprocessing
import os
import time
import uuid
from functools import partial

import gevent
from mars import promise
from mars.actors import FunctionActor, create_actor_pool
from mars.cluster_info import ClusterInfoActor
from mars.config import options
from mars.errors import StoreFull
from mars.scheduler.kvstore import KVStoreActor
from mars.utils import get_next_port, calc_data_size
from mars.worker import *
from mars.worker.distributor import WorkerDistributor
from mars.worker.chunkstore import PlasmaChunkStore
from mars.worker.tests.base import WorkerCase
from mars.worker.utils import WorkerActor
from pyarrow import plasma


class HolderActor(FunctionActor):
    def __init__(self):
        super(HolderActor, self).__init__()
        self._state = 0

    def trigger(self):
        self._state = 1

    def obtain(self):
        return self._state


class WorkerRegistrationTestActor(WorkerActor):
    def __init__(self):
        super(WorkerRegistrationTestActor, self).__init__()
        self._finished = False

    def get_finished(self):
        return self._finished

    def register(self, session_id, chunk_keys):
        import numpy as np

        cache_ref = self.promise_ref(ChunkHolderActor.default_name())

        left_keys = set(chunk_keys)

        def _put_chunk(chunk_key, data, spill_times=1):
            try:
                refs = self._chunk_store.put(session_id, chunk_key, data)
                cache_ref.register_chunk(session_id, chunk_key)
                del refs
                left_keys.remove(chunk_key)
            except StoreFull:
                return cache_ref.spill_size(2 * spill_times * calc_data_size(data), _promise=True) \
                    .then(partial(_put_chunk, chunk_key, data, 2 * spill_times))

        promises = []
        for idx, chunk_key in enumerate(chunk_keys):
            data = np.ones((640 * 1024,), dtype=np.int16) * idx
            promises.append(promise.Promise(done=True) \
                .then(partial(_put_chunk, chunk_key, data)))
        promise.all_(promises).then(lambda *_: setattr(self, '_finished', True))


def run_transfer_worker(pool_address, session_id, plasma_socket, chunk_keys,
                        spill_dir, msg_queue):
    from mars.config import options
    from mars.utils import PlasmaProcessHelper

    options.worker.plasma_socket = plasma_socket
    options.worker.spill_directory = spill_dir

    plasma_helper = PlasmaProcessHelper(size=1024 * 1024 * 10, socket=options.worker.plasma_socket)
    try:
        plasma_helper.run()

        with create_actor_pool(n_process=2, backend='gevent', distributor=WorkerDistributor(2),
                               address=pool_address) as pool:
            try:
                pool.create_actor(ClusterInfoActor, schedulers=[pool_address],
                                  uid=ClusterInfoActor.default_name())
                pool.create_actor(KVStoreActor, uid=KVStoreActor.default_name())
                pool.create_actor(DispatchActor, uid=DispatchActor.default_name())
                pool.create_actor(QuotaActor, 1024 * 1024 * 20, uid=MemQuotaActor.default_name())
                holder_ref = pool.create_actor(HolderActor, uid='HolderActor')
                chunk_holder_ref = pool.create_actor(ChunkHolderActor, plasma_helper._size,
                                                     uid=ChunkHolderActor.default_name())
                pool.create_actor(SpillActor)

                pool.create_actor(SenderActor, uid='%s' % str(uuid.uuid4()))
                pool.create_actor(SenderActor, uid='%s' % str(uuid.uuid4()))

                pool.create_actor(ReceiverActor, uid='%s' % str(uuid.uuid4()))
                pool.create_actor(ReceiverActor, uid='%s' % str(uuid.uuid4()))

                register_actor = pool.create_actor(WorkerRegistrationTestActor)
                register_actor.register(session_id, chunk_keys)

                check_time = time.time()
                while not register_actor.get_finished():
                    gevent.sleep(0.5)
                    if time.time() - check_time > 60:
                        raise SystemError('Wait result timeout')
                register_actor.destroy()

                msg_queue.put(1)
                check_time = time.time()
                while not holder_ref.obtain():
                    gevent.sleep(1)
                    if time.time() - check_time > 60:
                        raise SystemError('Wait result timeout')
            finally:
                pool.destroy_actor(chunk_holder_ref)
    finally:
        plasma_helper.stop()


class Test(WorkerCase):
    def setUp(self):
        super(Test, self).setUp()
        self._old_block_size = options.worker.transfer_block_size
        options.worker.transfer_block_size = 4 * 1024

    def tearDown(self):
        super(Test, self).tearDown()
        options.worker.transfer_block_size = self._old_block_size

    def testSimpleTransfer(self):
        import tempfile
        session_id = str(uuid.uuid4())

        local_pool_addr = 'localhost:%d' % get_next_port()
        remote_pool_addr = 'localhost:%d' % get_next_port()
        remote_chunk_keys = [str(uuid.uuid4()) for _ in range(9)]
        msg_queue = multiprocessing.Queue()

        remote_plasma_socket = '/tmp/plasma_%d_%d.sock' % (os.getpid(), id(run_transfer_worker))
        remote_spill_dir = os.path.join(tempfile.gettempdir(),
                                        'mars_spill_%d_%d' % (os.getpid(), id(run_transfer_worker)))

        proc = multiprocessing.Process(
            target=run_transfer_worker,
            args=(remote_pool_addr, session_id, remote_plasma_socket,
                  remote_chunk_keys, remote_spill_dir, msg_queue)
        )
        proc.start()
        try:
            msg_queue.get(30)
        except:
            if proc.is_alive():
                proc.terminate()
            raise

        with create_actor_pool(n_process=1, distributor=WorkerDistributor(1),
                               backend='gevent', address=local_pool_addr) as pool:
            pool.create_actor(ClusterInfoActor, schedulers=[local_pool_addr],
                              uid=ClusterInfoActor.default_name())
            kv_store_ref = pool.create_actor(KVStoreActor, uid=KVStoreActor.default_name())
            pool.create_actor(DispatchActor, uid=DispatchActor.default_name())
            pool.create_actor(QuotaActor, 1024 * 1024 * 20, uid=MemQuotaActor.default_name())
            cache_ref = pool.create_actor(ChunkHolderActor, self.plasma_storage_size,
                                          uid=ChunkHolderActor.default_name())
            pool.create_actor(SpillActor)

            sender_refs = [
                pool.create_actor(SenderActor, uid='w:1:%s' % str(uuid.uuid4())),
                pool.create_actor(SenderActor, uid='w:2:%s' % str(uuid.uuid4())),
            ]

            receiver_refs = [
                pool.create_actor(ReceiverActor, uid='w:1:%s' % str(uuid.uuid4())),
                pool.create_actor(ReceiverActor, uid='w:1:%s' % str(uuid.uuid4())),
                pool.create_actor(ReceiverActor, uid='w:2:%s' % str(uuid.uuid4())),
                pool.create_actor(ReceiverActor, uid='w:2:%s' % str(uuid.uuid4())),
            ]

            try:
                for data_id in (-1, 1):
                    chunk_key = remote_chunk_keys[data_id]

                    with self.run_actor_test(pool) as test_actor:
                        from mars.worker.spill import build_spill_file_name
                        from mars.serialize import dataserializer
                        from numpy.testing import assert_array_equal

                        remote_dispatch_ref = test_actor.promise_ref(
                            DispatchActor.default_name(), address=remote_pool_addr)
                        remote_plasma_client = plasma.connect(remote_plasma_socket, '', 0)
                        remote_store = PlasmaChunkStore(remote_plasma_client, kv_store_ref)

                        def _call_send_data(sender_uid):
                            sender_ref = test_actor.promise_ref(sender_uid, address=remote_pool_addr)
                            return sender_ref.send_data(session_id, chunk_key, local_pool_addr, _promise=True)

                        def _test_data_exist(*_):
                            try:
                                local_data = test_actor._chunk_store.get(session_id, chunk_key)
                            except KeyError:
                                with open(build_spill_file_name(chunk_key), 'rb') as spill_file:
                                    local_data = dataserializer.load(spill_file)

                            try:
                                remote_data = remote_store.get(session_id, chunk_key)
                            except KeyError:
                                with open(build_spill_file_name(chunk_key, remote_spill_dir), 'rb') as spill_file:
                                    remote_data = dataserializer.load(spill_file)
                            assert_array_equal(local_data, remote_data)

                            del local_data, remote_data

                        remote_dispatch_ref.get_free_slot('sender', _promise=True) \
                            .then(_call_send_data) \
                            .then(_test_data_exist) \
                            .then(
                            lambda *_: test_actor.set_result(chunk_key),
                            lambda *exc: test_actor.set_result(exc, False),
                        )
                    self.assertEqual(self.get_result(60), chunk_key)

                remote_holder_ref = pool.actor_ref('HolderActor', address=remote_pool_addr)
                remote_holder_ref.trigger()
            finally:
                for ref in sender_refs:
                    pool.destroy_actor(ref)
                for ref in receiver_refs:
                    pool.destroy_actor(ref)
                pool.destroy_actor(cache_ref)

                os.unlink(remote_plasma_socket)
                if proc.is_alive():
                    proc.terminate()
