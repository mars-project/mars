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

import atexit
import logging
import os
import random
import threading
import time
import uuid

from ...actors import new_client
from ...utils import to_binary, to_str
from .config import MarsSchedulerConfig

try:
    from skein import ApplicationClient, Client as SkeinClient, \
        properties as skein_props, SkeinError
except ImportError:
    ApplicationClient, SkeinClient, skein_props, SkeinError = None, None, None, None

logger = logging.getLogger(__name__)


class YarnNodeWatcher(object):
    """
    Yarn node watcher class, compatible with SchedulerDiscoverer
    """
    dynamic = True

    def __init__(self, endpoint_file_name, expected_instances):
        self._expected_instances = expected_instances
        self._endpoint_file_name = endpoint_file_name
        logger.debug('Watching endpoint file at %s', self._endpoint_file_name)
        self._endpoint_last_modified = self._get_mtime()

    def __reduce__(self):
        return type(self), (self._endpoint_file_name, self._expected_instances)

    def _get_mtime(self):
        if not os.path.exists(self._endpoint_file_name):
            return 0
        else:
            return os.path.getmtime(self._endpoint_file_name)

    def get_schedulers(self):
        if os.path.exists(self._endpoint_file_name):
            with open(self._endpoint_file_name, 'r') as file_obj:
                eps = file_obj.read().splitlines(False)
            return sorted(eps)
        return []

    def is_all_schedulers_ready(self):
        return self._expected_instances == len(self.get_schedulers())

    def watch_schedulers(self):
        import gevent

        cur_endpoints = self.get_schedulers()
        yield cur_endpoints

        while True:
            if len(cur_endpoints) == self._expected_instances:
                linger = 10
            else:
                linger = 1

            gevent.sleep(linger)
            mtime = self._get_mtime()
            if mtime == self._endpoint_last_modified:
                continue

            self._endpoint_last_modified = mtime
            cur_endpoints = self.get_schedulers()
            yield cur_endpoints


def daemon_thread(mars_app, app_client, key_prefix, endpoint_file):
    last_eps = set()
    while True:
        try:
            cid_to_endpoint = dict()
            for val in app_client.kv.get_prefix(key_prefix).values():
                ep, cid = to_str(val).split('@', 1)
                cid_to_endpoint[cid] = ep

            containers = app_client.get_containers([MarsSchedulerConfig.service_name], states=['RUNNING'])
            eps = set()
            for container in containers:
                if container.yarn_container_id not in cid_to_endpoint:
                    continue
                eps.add(cid_to_endpoint[container.yarn_container_id])

            if eps != last_eps:
                logger.info('New endpoints retrieved: %r', eps)
                with open(endpoint_file, 'w') as file_obj:
                    file_obj.write('\n'.join(eps))
                last_eps = eps
            time.sleep(1)
        except SkeinError:
            mars_app._running = False
            break


class YarnServiceMixin(object):
    service_name = None

    @property
    def app_client(self):
        if not hasattr(self, '_app_client'):
            self._app_client = ApplicationClient.from_current()
        return self._app_client

    @property
    def endpoint_file_name(self):
        if not hasattr(self, '_endpoint_file_name'):
            self._endpoint_file_name = os.path.abspath(skein_props['yarn_container_id'] + '-scheduler-names')
            atexit.register(os.unlink, self._endpoint_file_name)
        return self._endpoint_file_name

    def get_container_ip(self):
        svc_containers = self.app_client.get_containers([self.service_name])
        container = next(c for c in svc_containers if c.yarn_container_id == skein_props['yarn_container_id'])
        return container.yarn_node_http_address.split(':')[0]

    def register_node(self):
        self._container_key = getattr(self, '_container_key', None) or \
                              self.service_name + '-' + str(uuid.uuid1())
        self.app_client.kv[self._container_key] = to_binary(
            f'{self.advertise_endpoint}@{skein_props["yarn_container_id"]}')

    def wait_all_schedulers_ready(self):
        """
        Wait till all containers are ready, both in yarn and in ClusterInfoActor
        """
        from ...scheduler.utils import SchedulerClusterInfoActor

        # check if all schedulers are ready using Kubernetes API
        sleep_fun = (getattr(self, 'pool', None) or time).sleep
        while not self.scheduler_discoverer.is_all_schedulers_ready():
            sleep_fun(1)
        yarn_schedulers = self.scheduler_discoverer.get_schedulers()

        logger.debug('Schedulers all ready in yarn, waiting ClusterInfoActor to be ready')
        # check if all schedulers are registered in ClusterInfoActor
        actor_client = new_client()
        while True:
            cluster_info = actor_client.actor_ref(
                SchedulerClusterInfoActor.default_uid(), address=random.choice(yarn_schedulers))
            cluster_info_schedulers = cluster_info.get_schedulers()
            if set(cluster_info_schedulers) == set(yarn_schedulers):
                break
            sleep_fun(1)  # pragma: no cover

    def start_daemon(self):
        thread_args = (self, self.app_client, MarsSchedulerConfig.service_name, self.endpoint_file_name)
        thread_obj = threading.Thread(target=daemon_thread, args=thread_args, daemon=True)
        thread_obj.start()

    def create_scheduler_discoverer(self):
        svc_spec = self.app_client.get_specification().services[MarsSchedulerConfig.service_name]
        self.scheduler_discoverer = YarnNodeWatcher(self.endpoint_file_name, svc_spec.instances)
