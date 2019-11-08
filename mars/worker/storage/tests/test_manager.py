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

from mars.utils import get_next_port
from mars.worker import WorkerDaemonActor
from mars.worker.tests.base import WorkerCase
from mars.worker.storage import *


class Test(WorkerCase):
    def testStorageManager(self):
        test_addr = '127.0.0.1:%d' % get_next_port()
        with self.create_pool(n_process=1, address=test_addr) as pool:
            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())
            data_key2 = str(uuid.uuid4())

            pool.create_actor(WorkerDaemonActor,
                              uid=WorkerDaemonActor.default_uid())
            manager_ref = pool.create_actor(StorageManagerActor,
                                            uid=StorageManagerActor.default_uid())

            self.assertIsNone(manager_ref.get_data_locations(session_id, 'NON_EXIST'))

            manager_ref.register_data(session_id, data_key1,
                                      (0, DataStorageDevice.SHARED_MEMORY), 1024, (16, 8))
            manager_ref.register_data(session_id, data_key1,
                                      (1, DataStorageDevice.PROC_MEMORY), 1024, (16, 8))
            manager_ref.register_data(session_id, data_key1,
                                      (0, DataStorageDevice.DISK), 2048)
            self.assertEqual([(0, DataStorageDevice.SHARED_MEMORY), (0, DataStorageDevice.DISK),
                              (1, DataStorageDevice.PROC_MEMORY)],
                             sorted(manager_ref.get_data_locations(session_id, data_key1)))
            self.assertEqual(2048, manager_ref.get_data_size(session_id, data_key1))
            self.assertEqual((16, 8), manager_ref.get_data_shape(session_id, data_key1))

            manager_ref.register_data(session_id, data_key2,
                                      (0, DataStorageDevice.SHARED_MEMORY), 1024)
            manager_ref.register_data(session_id, data_key2,
                                      (1, DataStorageDevice.PROC_MEMORY), 1024)
            self.assertEqual([(0, DataStorageDevice.SHARED_MEMORY), (1, DataStorageDevice.PROC_MEMORY)],
                             sorted(manager_ref.get_data_locations(session_id, data_key2)))

            manager_ref.unregister_data(session_id, data_key2,
                                        (0, DataStorageDevice.SHARED_MEMORY))
            self.assertEqual([(1, DataStorageDevice.PROC_MEMORY)],
                             sorted(manager_ref.get_data_locations(session_id, data_key2)))
            self.assertEqual(1024, manager_ref.get_data_size(session_id, data_key2))
            self.assertEqual([data_key1],
                             list(manager_ref.filter_exist_keys(session_id, [data_key1, data_key2, 'non-exist'],
                                                                [(0, DataStorageDevice.SHARED_MEMORY)])))

            manager_ref.unregister_data(session_id, data_key2,
                                        (1, DataStorageDevice.PROC_MEMORY))
            self.assertIsNone(manager_ref.get_data_locations(session_id, data_key2))
            self.assertIsNone(manager_ref.get_data_size(session_id, data_key2))
            self.assertIsNone(manager_ref.get_data_shape(session_id, data_key2))

            manager_ref.register_data(session_id, data_key2,
                                      (1, DataStorageDevice.PROC_MEMORY), 1024)
            manager_ref.handle_process_down([1])
            self.assertEqual([(0, DataStorageDevice.SHARED_MEMORY), (0, DataStorageDevice.DISK)],
                             sorted(manager_ref.get_data_locations(session_id, data_key1)))
            self.assertIsNone(manager_ref.get_data_locations(session_id, data_key2))
            self.assertIsNone(manager_ref.get_data_size(session_id, data_key2))
            self.assertIsNone(manager_ref.get_data_shape(session_id, data_key2))
