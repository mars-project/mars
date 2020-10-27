# -*- coding: utf-8 -*-
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

import os
import logging

from ...scheduler import ResourceActor
from ...scheduler.__main__ import SchedulerApplication
from ...scheduler.utils import SchedulerActor, SchedulerClusterInfoActor
from .core import K8SServiceMixin, ReadinessActor, K8SPodsIPWatcher

logger = logging.getLogger(__name__)


class WorkerWatcherActor(SchedulerActor):
    watcher_cls = K8SPodsIPWatcher
    _watcher_running = True

    def __init__(self):
        super().__init__()
        self._resource_ref = None

    def post_create(self):
        super().post_create()

        logger.debug('Actor %s running in process %d', self.uid, os.getpid())
        self._cluster_info_ref = self.ctx.actor_ref(SchedulerClusterInfoActor.default_uid())
        self.ref().watch_workers(_tell=True)

    def pre_destroy(self):
        super().pre_destroy()
        self._watcher_running = False

    def watch_workers(self):
        from kubernetes import client, config

        cls = type(self)

        if os.environ.get('KUBE_API_ADDRESS'):  # pragma: no cover
            k8s_config = client.Configuration()
            k8s_config.host = os.environ['KUBE_API_ADDRESS']
        else:
            k8s_config = config.load_incluster_config()

        watcher = self.watcher_cls(k8s_config, os.environ['MARS_K8S_POD_NAMESPACE'])

        for workers in watcher.watch_workers():  # pragma: no branch
            if not cls._watcher_running:  # pragma: no cover
                break

            if self._resource_ref is None:
                self.set_schedulers(self._cluster_info_ref.get_schedulers())
                self._resource_ref = self.get_actor_ref(ResourceActor.default_uid())

            if self._resource_ref:  # pragma: no branch
                self._resource_ref.mark_workers_alive(workers)


class K8SSchedulerApplication(K8SServiceMixin, SchedulerApplication):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._readiness_ref = None
        self._worker_watcher_ref = None

    def start(self):
        self.write_pid_file()
        super().start()
        self._worker_watcher_ref = self.pool.create_actor(
            WorkerWatcherActor, uid=WorkerWatcherActor.default_uid())
        self._readiness_ref = self.pool.create_actor(ReadinessActor, uid=ReadinessActor.default_uid())

    def stop(self):
        self._readiness_ref.destroy()
        self._worker_watcher_ref.destroy()
        super().stop()


main = K8SSchedulerApplication()

if __name__ == '__main__':   # pragma: no branch
    main()
