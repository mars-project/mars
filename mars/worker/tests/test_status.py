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

import gevent

from mars.actors import create_actor_pool
from mars.cluster_info import ClusterInfoActor
from mars.scheduler import KVStoreActor
from mars.worker.tests.base import WorkerCase
from mars.worker import *
from mars.utils import get_next_port


class Test(WorkerCase):
    def testStatus(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        with create_actor_pool(n_process=1, backend='gevent', address=pool_address) as pool:
            pool.create_actor(ClusterInfoActor, schedulers=[pool_address], uid=ClusterInfoActor.default_name())
            pool.create_actor(KVStoreActor, uid='KVStoreActor')
            pool.create_actor(ChunkHolderActor, self.plasma_storage_size, uid=ChunkHolderActor.default_name())
            pool.create_actor(StatusActor, '127.0.0.1:1234', uid=StatusActor.default_name())

            def delay_read():
                gevent.sleep(2)
                return self._kv_store.read('/workers/meta', recursive=True)

            gl = gevent.spawn(delay_read)
            gl.join()
            v = gl.value
            print(v)
