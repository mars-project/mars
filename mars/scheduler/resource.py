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

import copy
import json
import logging
import os
import time
from datetime import datetime, timedelta

from .kvstore import KVStoreActor
from .utils import SchedulerActor
from ..config import options


logger = logging.getLogger(__name__)


class ResourceActor(SchedulerActor):
    """
    Actor managing free resources on workers
    """
    def __init__(self):
        super(ResourceActor, self).__init__()
        self._meta_cache = dict()
        self._kv_store_ref = None

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        super(ResourceActor, self).post_create()
        self.ref().clean_worker()

        self._kv_store_ref = self.ctx.actor_ref(KVStoreActor.default_name())
        if not self.ctx.has_actor(self._kv_store_ref):
            self._kv_store_ref = None

    @classmethod
    def default_name(cls):
        return 's:%s' % cls.__name__

    def clean_worker(self):
        """
        Remove worker when it does not update its status for a long time
        """
        timeout = options.scheduler.status_timeout
        for worker in list(self._meta_cache.keys()):
            worker_meta = self._meta_cache[worker]
            if 'update_time' not in worker_meta:
                continue

            last_time = datetime.strptime(worker_meta['update_time'], '%Y-%m-%d %H:%M:%S')
            time_delta = timedelta(seconds=timeout)
            if last_time + time_delta < datetime.now():
                del self._meta_cache[worker]

        self.ref().clean_worker(_tell=True, _delay=1)

    def get_worker_count(self):
        return len(self._meta_cache)

    def get_workers_meta(self):
        return copy.deepcopy(self._meta_cache)

    def set_worker_meta(self, worker, worker_meta):
        self._meta_cache[worker] = worker_meta
        if self._kv_store_ref is not None:
            self._kv_store_ref.write('/workers/meta/%s' % worker, json.dumps(worker_meta),
                                     _tell=True, _wait=False)
            self._kv_store_ref.write('/workers/meta_timestamp', str(int(time.time())),
                                     _tell=True, _wait=False)
