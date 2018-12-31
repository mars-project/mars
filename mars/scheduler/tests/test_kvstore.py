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

from mars.actors import create_actor_pool
from mars.scheduler.kvstore import KVStoreActor
from mars.tests.core import EtcdProcessHelper


class Test(unittest.TestCase):
    def testKVStoreActor(self):
        proc_helper = EtcdProcessHelper(port_range_start=54131)
        with proc_helper.run(), create_actor_pool(n_process=1, backend='gevent') as pool:
            store_ref = pool.create_actor(KVStoreActor, uid=KVStoreActor.default_name())

            store_ref.write('/node/v1', 'value1')
            store_ref.write('/node/v2', 'value2')
            store_ref.write_batch([
                ('/node/v2', 'value2'),
                ('/node/v3', 'value3'),
            ])

            self.assertEqual(store_ref.read('/node/v1').value, 'value1')
            self.assertListEqual([v.value for v in store_ref.read_batch(['/node/v2', '/node/v3'])],
                                 ['value2', 'value3'])
