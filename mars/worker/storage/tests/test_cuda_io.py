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
import weakref

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from mars.serialize import dataserializer
from mars.tests.core import require_cupy, require_cudf
from mars.utils import get_next_port, lazy_import
from mars.worker import WorkerDaemonActor, QuotaActor, MemQuotaActor
from mars.worker.tests.base import WorkerCase
from mars.worker.storage import *

cp = lazy_import('cupy', globals=globals(), rename='cp')
cudf = lazy_import('cudf', globals=globals())


@require_cupy
@require_cudf
class Test(WorkerCase):
    def testCudaMemPutAndGet(self):
        test_addr = '127.0.0.1:%d' % get_next_port()
        with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())
            pool.create_actor(QuotaActor, 1024 ** 2, uid=MemQuotaActor.default_uid())
            pool.create_actor(CudaHolderActor)

            test_data = np.random.random((10, 10))
            test_suites = [
                (test_data, cp.ndarray, cp.asnumpy, assert_allclose),
                (pd.Series(test_data.flatten()), cudf.Series,
                 lambda o: o.to_pandas(), pd.testing.assert_series_equal),
                (pd.DataFrame(dict(col=test_data.flatten())), cudf.DataFrame,
                 lambda o: o.to_pandas(), pd.testing.assert_frame_equal),
            ]

            for data, cuda_type, move_to_mem, assert_obj_equal in test_suites:
                ser_data = dataserializer.serialize(data)

                session_id = str(uuid.uuid4())
                data_key1 = str(uuid.uuid4())
                data_key2 = str(uuid.uuid4())

                storage_client = test_actor.storage_client
                handler = storage_client.get_storage_handler((0, DataStorageDevice.CUDA))

                handler.put_objects(session_id, [data_key1], [data])
                self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key1])[0]),
                                 [(0, DataStorageDevice.CUDA)])
                self.assertIsInstance(handler.get_objects(session_id, [data_key1])[0], cuda_type)
                assert_obj_equal(data, move_to_mem(handler.get_objects(session_id, [data_key1])[0]))

                handler.delete(session_id, [data_key1])
                self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key1])[0]), [])
                with self.assertRaises(KeyError):
                    handler.get_objects(session_id, [data_key1])

                handler.put_objects(session_id, [data_key2], [ser_data], serialized=True)
                self.assertIsInstance(handler.get_objects(session_id, [data_key2])[0], cuda_type)
                assert_obj_equal(data, move_to_mem(handler.get_objects(session_id, [data_key2])[0]))
                handler.delete(session_id, [data_key2])

    def testCudaMemLoad(self):
        test_addr = '127.0.0.1:%d' % get_next_port()
        with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            pool.create_actor(QuotaActor, 1024 ** 2, uid=MemQuotaActor.default_uid())
            pool.create_actor(CudaHolderActor)

            pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            pool.create_actor(SharedHolderActor, uid=SharedHolderActor.default_uid())

            data1 = np.random.random((10, 10))
            data2 = np.random.random((10, 10))
            ser_data1 = dataserializer.serialize(data1)

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())
            data_key2 = str(uuid.uuid4())

            storage_client = test_actor.storage_client
            handler = storage_client.get_storage_handler((0, DataStorageDevice.CUDA))

            # load from bytes io
            disk_handler = storage_client.get_storage_handler((0, DataStorageDevice.DISK))
            with disk_handler.create_bytes_writer(
                    session_id, data_key1, ser_data1.total_bytes) as writer:
                ser_data1.write_to(writer)

            handler.load_from_bytes_io(session_id, [data_key1], disk_handler) \
                .then(lambda *_: test_actor.set_result(None),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            self.get_result(5)
            self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key1])[0]),
                             [(0, DataStorageDevice.CUDA), (0, DataStorageDevice.DISK)])

            disk_handler.delete(session_id, [data_key1])

            data_load = handler.get_objects(session_id, [data_key1])[0]
            ref_data = weakref.ref(data_load)
            del data_load
            handler.delete(session_id, [data_key1])
            self.assertIsNone(ref_data())

            # load from object io
            shared_handler = storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))
            shared_handler.put_objects(session_id, [data_key2], [data2])

            handler.load_from_object_io(session_id, [data_key2], shared_handler) \
                .then(lambda *_: test_actor.set_result(None),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            self.get_result(5)
            self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key2])[0]),
                             [(0, DataStorageDevice.CUDA), (0, DataStorageDevice.SHARED_MEMORY)])

            shared_handler.delete(session_id, [data_key2])

            data_load = handler.get_objects(session_id, [data_key2])[0]
            ref_data = weakref.ref(data_load)
            del data_load
            handler.delete(session_id, [data_key2])
            self.assertIsNone(ref_data())
