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

import shutil
import tempfile

import gevent

from mars.actors import create_actor_pool
from mars.config import options
from mars.scheduler import ResourceActor
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.worker.tests.base import WorkerCase
from mars.worker import ChunkHolderActor, StatusActor
from mars.worker.utils import WorkerClusterInfoActor
from mars.utils import get_next_port


class Test(WorkerCase):
    def testStatus(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        old_spill_dir = options.worker.spill_directory
        dir_name = options.worker.spill_directory = tempfile.mkdtemp(prefix='temp-mars-spill-')
        try:
            with create_actor_pool(n_process=1, backend='gevent', address=pool_address) as pool:
                pool.create_actor(SchedulerClusterInfoActor, [pool_address],
                                  uid=SchedulerClusterInfoActor.default_uid())
                pool.create_actor(WorkerClusterInfoActor, [pool_address],
                                  uid=WorkerClusterInfoActor.default_uid())

                resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_uid())
                pool.create_actor(ChunkHolderActor, self.plasma_storage_size,
                                  uid=ChunkHolderActor.default_uid())
                status_ref = pool.create_actor(StatusActor, pool_address,
                                               uid=StatusActor.default_uid())
                status_ref.enable_status_upload()

                status_ref.update_slots(dict(cpu=4))
                status_ref.update_stats(dict(min_est_finish_time=10))

                def delay_read():
                    gevent.sleep(1.5)
                    return resource_ref.get_workers_meta()

                gl = gevent.spawn(delay_read)
                gl.join()
                v = gl.value
                self.assertIsNotNone(v)

                pool.destroy_actor(status_ref)
        finally:
            options.worker.spill_directory = old_spill_dir
            shutil.rmtree(dir_name)
