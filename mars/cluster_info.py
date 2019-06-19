#!/usr/bin/env python
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

import logging
import os

from . import kvstore
from .actors import FunctionActor
from .promise import PromiseActor
from .lib.uhashring import HashRing
from .utils import to_str


SCHEDULER_PATH = '/schedulers'
logger = logging.getLogger(__name__)


def create_hash_ring(schedulers):
    return HashRing(nodes=schedulers, hash_fn='ketama')


def get_scheduler(hash_ring, key, size=1):
    if size == 1:
        return hash_ring.get_node(key)

    return tuple(it['nodename'] for it in hash_ring.range(key, size=size))


class _ClusterInfoWatchActor(FunctionActor):
    def __init__(self, service_discover_addr, cluster_info_ref):
        self._client = kvstore.get(service_discover_addr)
        self._cluster_info_ref = cluster_info_ref

        if isinstance(self._client, kvstore.LocalKVStore):
            raise ValueError('etcd_addr should not be a local address, got {0}'.format(service_discover_addr))

    def _get_schedulers(self):
        schedulers = [s.key.rsplit('/', 1)[1] for s in self._client.read(SCHEDULER_PATH).children]
        logger.debug('Schedulers obtained. Results: %r', schedulers)
        return [to_str(s) for s in schedulers]

    def get_schedulers(self):
        try:
            return self._get_schedulers()
        except KeyError:
            self._client.watch(SCHEDULER_PATH)
            return self._get_schedulers()

    def watch(self):
        for new_schedulers in self._client.eternal_watch(SCHEDULER_PATH):
            self._cluster_info_ref.set_schedulers([to_str(s) for s in new_schedulers])


class ClusterInfoActor(FunctionActor):
    def __init__(self, schedulers=None, service_discover_addr=None):
        if (schedulers is None and service_discover_addr is None) or \
                (schedulers is not None and service_discover_addr is not None):
            raise ValueError('Either schedulers or etcd_addr should be provided')

        self._schedulers = schedulers
        self._hash_ring = None
        self._service_discover_addr = service_discover_addr
        self._watcher = None
        self._observer_refs = []

    @classmethod
    def default_uid(cls):
        raise NotImplementedError

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        if self._service_discover_addr:
            watcher = self._watcher = self.ctx.create_actor(
                _ClusterInfoWatchActor, self._service_discover_addr, self.ref())
            self._schedulers = watcher.get_schedulers()
            watcher.watch(_tell=True)

        self._hash_ring = create_hash_ring(self._schedulers)

    def pre_destroy(self):
        if self._service_discover_addr:
            self.ctx.destroy_actor(self._watcher)

    def register_observer(self, observer, fun_name):
        self._observer_refs.append((self.ctx.actor_ref(observer), fun_name))

    def get_schedulers(self):
        return self._schedulers

    def set_schedulers(self, schedulers):
        self._schedulers = schedulers
        self._hash_ring = create_hash_ring(self._schedulers)

        for observer_ref, fun_name in self._observer_refs:
            # notify the observers to update the new scheduler list
            getattr(observer_ref, fun_name)(schedulers)

    def get_scheduler(self, key, size=1):
        if len(self._schedulers) == 1 and size == 1:
            return self._schedulers[0]
        return get_scheduler(self._hash_ring, key, size=size)


class HasClusterInfoActor(PromiseActor):
    cluster_info_uid = None

    def __init__(self):
        super(HasClusterInfoActor, self).__init__()

        # the scheduler list
        self._schedulers = None
        self._hash_ring = None

        self._cluster_info_ref = None

    def set_schedulers(self, schedulers):
        self._schedulers = schedulers
        if len(schedulers) > 1:
            self._hash_ring = create_hash_ring(schedulers)

    def set_cluster_info_ref(self, set_schedulers_fun_name=None):
        set_schedulers_fun_name = set_schedulers_fun_name or self.set_schedulers.__name__

        # cluster_info_actor is created when scheduler initialized
        self._cluster_info_ref = self.ctx.actor_ref(self.cluster_info_uid)
        # when some schedulers lost, notification will be received
        self._cluster_info_ref.register_observer(self.ref(), set_schedulers_fun_name)
        self.set_schedulers(self._cluster_info_ref.get_schedulers())

    def get_actor_ref(self, key):
        addr = self.get_scheduler(key)
        return self.ctx.actor_ref(addr, key)

    def get_promise_ref(self, key):
        addr = self.get_scheduler(key)
        return self.promise_ref(addr, key)

    def get_scheduler(self, key, size=1):
        if len(self._schedulers) == 1:
            return self._schedulers[0]

        return get_scheduler(self._hash_ring, key, size=size)
