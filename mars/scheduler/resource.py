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

import json
import logging
import os
import time

from .kvstore import KVStoreActor
from .session import SessionManagerActor, SessionActor
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
        self._worker_blacklist_time = dict()
        self._kv_store_ref = None

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        super(ResourceActor, self).post_create()
        self.set_cluster_info_ref()

        self._kv_store_ref = self.ctx.actor_ref(KVStoreActor.default_name())
        if not self.ctx.has_actor(self._kv_store_ref):
            self._kv_store_ref = None

        self.ref().detect_dead_workers(_tell=True)

    @classmethod
    def default_name(cls):
        return 's:%s' % cls.__name__

    def _check_worker_in_blacklist(self, worker):
        if worker not in self._worker_blacklist_time:
            return False
        expire_time = time.time() - options.scheduler.worker_blacklist_time
        if expire_time > self._worker_blacklist_time[worker]:
            del self._worker_blacklist_time[worker]
            return False
        else:
            return True

    def detect_dead_workers(self):
        """
        Remove worker when it does not update its status for a long time
        """
        timeout = options.scheduler.status_timeout
        dead_workers = []

        check_time = time.time()
        for worker in list(self._meta_cache.keys()):
            worker_meta = self._meta_cache[worker]
            if 'update_time' not in worker_meta:
                continue
            try:
                if check_time - worker_meta['update_time'] > timeout:
                    dead_workers.append(worker)
            except (TypeError, ValueError):
                pass

        self.detach_dead_workers(dead_workers)
        self.ref().detect_dead_workers(_tell=True, _delay=1)

    def detach_dead_workers(self, workers):
        if not workers:
            return

        logger.warning('Workers %r dead, detaching from ResourceActor.', workers)
        workers = [w for w in workers if w in self._meta_cache and
                   not self._check_worker_in_blacklist(w)]
        for w in workers:
            del self._meta_cache[w]
            self._worker_blacklist_time[w] = time.time()
        if workers:
            self._broadcast_sessions(SessionActor.handle_worker_change, [], workers)

    def get_worker_count(self):
        return len(self._meta_cache)

    def get_workers_meta(self):
        return dict((k, v) for k, v in self._meta_cache.items()
                    if k not in self._worker_blacklist_time)

    def set_worker_meta(self, worker, worker_meta):
        if self._check_worker_in_blacklist(worker):
            return

        is_new = worker not in self._meta_cache

        self._meta_cache[worker] = worker_meta
        if self._kv_store_ref is not None:
            self._kv_store_ref.write('/workers/meta/%s' % worker, json.dumps(worker_meta),
                                     _tell=True, _wait=False)
            self._kv_store_ref.write('/workers/meta_timestamp', str(int(time.time())),
                                     _tell=True, _wait=False)
        if is_new:
            self._broadcast_sessions(SessionActor.handle_worker_change, [worker], [])

    def _broadcast_sessions(self, handler, *args, **kwargs):
        from .assigner import AssignerActor

        if not options.scheduler.enable_failover:  # pragma: no cover
            return

        if hasattr(handler, '__name__'):
            handler = handler.__name__

        futures = []
        for ep in self.get_schedulers():
            ref = self.ctx.actor_ref(AssignerActor.default_name(), address=ep)
            futures.append(ref.mark_metrics_expired(_tell=True, _wait=False))
        [f.result() for f in futures]

        futures = []
        for ep in self.get_schedulers():
            ref = self.ctx.actor_ref(SessionManagerActor.default_name(), address=ep)
            kwargs.update(dict(_tell=True, _wait=False))
            futures.append(ref.broadcast_sessions(handler, *args, **kwargs))
        [f.result() for f in futures]
