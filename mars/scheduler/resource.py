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
from ..utils import BlacklistSet


logger = logging.getLogger(__name__)


class ResourceHeartbeatActor(SchedulerActor):
    def __init__(self, resource_ref):
        super(ResourceHeartbeatActor, self).__init__()
        self._resource_ref = resource_ref

    def post_create(self):
        super(ResourceHeartbeatActor, self).post_create()
        self._resource_ref = self.ctx.actor_ref(self._resource_ref)
        self.ref().do_heartbeat(_tell=True)

    def do_heartbeat(self):
        self._resource_ref.heartbeat(_tell=True)
        self.ref().do_heartbeat(_tell=True, _delay=1)


class ResourceActor(SchedulerActor):
    """
    Actor managing free resources on workers
    """
    def __init__(self):
        super(ResourceActor, self).__init__()
        self._meta_cache = dict()
        self._worker_blacklist = BlacklistSet(options.scheduler.worker_blacklist_time)

        self._last_heartbeat_time = time.time()
        self._last_heartbeat_interval = 0

        self._kv_store_ref = None
        self._heartbeat_ref = None

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        super(ResourceActor, self).post_create()
        self.set_cluster_info_ref()

        self._kv_store_ref = self.ctx.actor_ref(KVStoreActor.default_uid())
        if not self.ctx.has_actor(self._kv_store_ref):
            self._kv_store_ref = None

        try:
            # we assign the heartbeat actor into another process
            # so it can sense whether the process is busy
            heartbeat_uid = self.ctx.distributor.make_same_process(
                ResourceHeartbeatActor.default_uid(), self.uid, delta=1)
        except AttributeError:
            heartbeat_uid = ResourceHeartbeatActor.default_uid()
        self._heartbeat_ref = self.ctx.create_actor(
            ResourceHeartbeatActor, self.ref(), uid=heartbeat_uid)

        self.ref().detect_dead_workers(_tell=True)

    def pre_destroy(self):
        self._heartbeat_ref.destroy()
        super(ResourceActor, self).pre_destroy()

    def heartbeat(self):
        t = time.time()
        self._last_heartbeat_interval = t - self._last_heartbeat_time
        self._last_heartbeat_time = t

    def detect_dead_workers(self):
        """
        Remove worker when it does not update its status for a long time
        """
        # take latency of scheduler into consideration by
        # include interval of last heartbeat
        timeout = options.scheduler.status_timeout + self._last_heartbeat_interval
        dead_workers = []

        check_time = self._last_heartbeat_time
        for worker in list(self._meta_cache.keys()):
            worker_meta = self._meta_cache[worker]
            try:
                if check_time - worker_meta['update_time'] > timeout:
                    dead_workers.append(worker)
            except (KeyError, TypeError, ValueError):
                pass

        self.detach_dead_workers(dead_workers)
        self.ref().detect_dead_workers(_tell=True, _delay=1)

    def detach_dead_workers(self, workers):
        from ..worker.execution import ExecutionActor

        workers = [w for w in workers if w in self._meta_cache and
                   w not in self._worker_blacklist]

        if not workers:
            return

        logger.warning('Workers %r dead, detaching from ResourceActor.', workers)
        for w in workers:
            del self._meta_cache[w]
            self._worker_blacklist.add(w)

        self._broadcast_sessions(SessionActor.handle_worker_change, [], workers)
        self._broadcast_workers(ExecutionActor.handle_worker_change, [], workers)

    def get_worker_count(self):
        return len(self._meta_cache)

    def get_workers_meta(self):
        return dict((k, v) for k, v in self._meta_cache.items()
                    if k not in self._worker_blacklist)

    def set_worker_meta(self, worker, worker_meta):
        from ..worker.execution import ExecutionActor

        if worker in self._worker_blacklist:
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
            self._broadcast_workers(ExecutionActor.handle_worker_change, [worker], [])

    def _broadcast_sessions(self, handler, *args, **kwargs):
        from .assigner import AssignerActor

        if not options.scheduler.enable_failover:  # pragma: no cover
            return

        if hasattr(handler, '__name__'):
            handler = handler.__name__

        futures = []
        for ep in self.get_schedulers():
            ref = self.ctx.actor_ref(AssignerActor.default_uid(), address=ep)
            futures.append(ref.mark_metrics_expired(_tell=True, _wait=False))
        [f.result() for f in futures]

        futures = []
        kwargs.update(dict(_tell=True, _wait=False))
        for ep in self.get_schedulers():
            ref = self.ctx.actor_ref(SessionManagerActor.default_uid(), address=ep)
            futures.append(ref.broadcast_sessions(handler, *args, **kwargs))
        [f.result() for f in futures]

    def _broadcast_workers(self, handler, *args, **kwargs):
        from ..worker.execution import ExecutionActor

        if not options.scheduler.enable_failover:  # pragma: no cover
            return

        if hasattr(handler, '__name__'):
            handler = handler.__name__

        kwargs.update(dict(_tell=True, _wait=False))
        for w in self._meta_cache.keys():
            ref = self.ctx.actor_ref(ExecutionActor.default_uid(), address=w)
            getattr(ref, handler)(*args, **kwargs)
