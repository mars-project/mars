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

import os
import sys
import unittest

from mars.actors import create_actor_pool
from mars.config import options
from mars.scheduler.kvstore import KVStoreActor
from mars.tests.core import EtcdProcessHelper
from mars.utils import get_next_port


class Test(unittest.TestCase):
    def tearDown(self):
        super(Test, self).tearDown()
        options.kv_store = ':inproc:'

    @unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
    @unittest.skipIf('CI' not in os.environ and not EtcdProcessHelper().is_installed(),
                     'does not run without etcd')
    def testKVStoreActor(self):
        etcd_port = get_next_port()
        proc_helper = EtcdProcessHelper(port_range_start=etcd_port)
        options.kv_store = 'etcd://127.0.0.1:%s' % etcd_port
        with proc_helper.run(), create_actor_pool(n_process=1, backend='gevent') as pool:
            store_ref = pool.create_actor(KVStoreActor, uid=KVStoreActor.default_uid())

            store_ref.write('/node/v1', 'value1')
            store_ref.write('/node/v2', 'value2')
            store_ref.write_batch([
                ('/node/v2', 'value2'),
                ('/node/v3', 'value3'),
            ])

            self.assertEqual(store_ref.read('/node/v1').value, 'value1')
            self.assertListEqual([v.value for v in store_ref.read_batch(['/node/v2', '/node/v3'])],
                                 ['value2', 'value3'])
