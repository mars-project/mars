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
import time
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta

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
        self._worker_allocations = dict()

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        super(ResourceActor, self).post_create()
        self.ref().clean_worker()

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

    def get_workers_meta(self):
        return copy.deepcopy(self._meta_cache)

    def set_worker_meta(self, worker, worker_meta):
        self._meta_cache[worker] = worker_meta

    def allocate_resource(self, session_id, op_key, endpoint, alloc_dict):
        """
        Try allocate resource for operands
        :param session_id: session id
        :param op_key: operand key
        :param endpoint: worker endpoint
        :param alloc_dict: allocation dict, listing resources needed by the operand
        :return: True if allocated successfully
        """
        worker_stats = self._meta_cache[endpoint]['hardware']
        if endpoint not in self._worker_allocations:
            self._worker_allocations[endpoint] = dict()
        if session_id not in self._worker_allocations[endpoint]:
            self._worker_allocations[endpoint][session_id] = dict()
        worker_allocs = self._worker_allocations[endpoint][session_id]

        res_used = defaultdict(lambda: 0)
        free_alloc_keys = []
        for alloc_op_key, alloc in worker_allocs.items():
            d, t = alloc
            for k, v in d.items():
                res_used[k] += v
        for k in free_alloc_keys:
            self.deallocate_resource(session_id, k, endpoint)

        for k, v in alloc_dict.items():
            res_used[k] += v

        sufficient = True
        for k, v in res_used.items():
            if res_used[k] > worker_stats.get(k, 0):
                sufficient = False
                break

        if sufficient:
            self._worker_allocations[endpoint][session_id][op_key] = (alloc_dict, time.time())
            return True
        else:
            return False

    def deallocate_resource(self, session_id, op_key, endpoint):
        """
        Deallocate resource
        :param session_id: session id
        :param op_key: operand key
        :param endpoint: worker endpoint
        """
        try:
            del self._worker_allocations[endpoint][session_id][op_key]
        except KeyError:
            pass
