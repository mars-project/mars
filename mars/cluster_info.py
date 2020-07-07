#!/usr/bin/env python
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

import logging
import os
import sys

from . import kvstore
from .actors import FunctionActor
from .promise import PromiseActor
from .lib.uhashring import HashRing
from .utils import to_str


SCHEDULER_PATH = '/schedulers'
INITIAL_SCHEDULER_FILE = '/tmp/mars-initial-schedulers.tmp'
logger = logging.getLogger(__name__)


def create_hash_ring(schedulers):
    return HashRing(nodes=schedulers, hash_fn='ketama')


def get_scheduler(hash_ring, key, size=1):
    if size == 1:
        return hash_ring.get_node(key)

    return tuple(it['nodename'] for it in hash_ring.range(key, size=size))


class StaticSchedulerDiscoverer(object):
    dynamic = False

    def __init__(self, schedulers):
        self._schedulers = schedulers

    def __reduce__(self):
        return type(self), (self._schedulers,)

    def get(self):
        return self._schedulers

    def watch(self):
        raise NotImplementedError


class KVStoreSchedulerDiscoverer(object):
    dynamic = True

    def __init__(self, address):
        self._address = address
        self._client = kvstore.get(address)
        if isinstance(self._client, kvstore.LocalKVStore):
            raise ValueError('etcd_addr should not be a local address, got {0}'.format(address))

    def __reduce__(self):
        return type(self), (self._address,)

    @staticmethod
    def _resolve_schedulers(result):
        return [to_str(s.key.rsplit('/', 1)[1]) for s in result.children]

    def get(self):
        try:
            return self._resolve_schedulers(self._client.read(SCHEDULER_PATH))
        except KeyError:
            return next(self.watch())

    def watch(self):
        for result in self._client.eternal_watch(SCHEDULER_PATH):
            yield self._resolve_schedulers(result)


class _ClusterInfoWatchActor(FunctionActor):
    def __init__(self, discoverer, cluster_info_ref):
        self._discoverer = discoverer
        self._cluster_info_ref = cluster_info_ref

    def post_create(self):
        self._cluster_info_ref = self.ctx.actor_ref(self._cluster_info_ref)

    def get_schedulers(self):
        return self._discoverer.get()

    def watch(self):
        for schedulers in self._discoverer.watch():
            self._cluster_info_ref.set_schedulers(schedulers, _tell=True)


class ClusterInfoActor(FunctionActor):
    def __init__(self, discoverer, distributed=True):
        if isinstance(discoverer, list):
            discoverer = StaticSchedulerDiscoverer(discoverer)

        self._discoverer = discoverer
        self._distributed = distributed
        self._hash_ring = None
        self._watcher = None
        self._schedulers = []
        self._observer_refs = []

    @classmethod
    def default_uid(cls):
        raise NotImplementedError

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        if self._discoverer.dynamic:
            watcher = self._watcher = self.ctx.create_actor(
                _ClusterInfoWatchActor, self._discoverer, self.ref())
            watcher.watch(_tell=True)
        self._schedulers = self._discoverer.get()

        self._hash_ring = create_hash_ring(self._schedulers)

    def pre_destroy(self):
        if self._watcher:
            self.ctx.destroy_actor(self._watcher)

    def register_observer(self, observer, fun_name):
        self._observer_refs.append((self.ctx.actor_ref(observer), fun_name))

    def get_schedulers(self):
        return self._schedulers

    def set_schedulers(self, schedulers):
        logger.debug('Setting schedulers %r', schedulers)
        self._schedulers = schedulers
        self._hash_ring = create_hash_ring(self._schedulers)

        for observer_ref, fun_name in self._observer_refs:
            # notify the observers to update the new scheduler list
            getattr(observer_ref, fun_name)(schedulers, _tell=True)

    def get_scheduler(self, key, size=1):
        if len(self._schedulers) == 1 and size == 1:
            return self._schedulers[0]
        return get_scheduler(self._hash_ring, key, size=size)

    def is_distributed(self):
        return self._distributed


class HasClusterInfoActor(PromiseActor):
    cluster_info_uid = None

    def __init__(self):
        super().__init__()

        # the scheduler list
        if sys.platform.startswith('linux') and os.path.exists(INITIAL_SCHEDULER_FILE):
            # there is predefined scheduler list, read first
            with open(INITIAL_SCHEDULER_FILE, 'r') as scheduler_file:
                self._schedulers = scheduler_file.read().strip().split(',')
        else:
            self._schedulers = None
        self._hash_ring = None

        self._cluster_info_ref = None

    def get_schedulers(self):
        return self._schedulers

    def set_schedulers(self, schedulers):
        self._schedulers = schedulers
        if len(schedulers) > 1:
            self._hash_ring = create_hash_ring(schedulers)

    def set_cluster_info_ref(self, set_schedulers_fun_name=None):
        set_schedulers_fun_name = set_schedulers_fun_name or self.set_schedulers.__name__

        # cluster_info_actor is created when scheduler initialized
        self._cluster_info_ref = self.ctx.actor_ref(self.cluster_info_uid)
        # when some schedulers lost, notification will be received
        self._cluster_info_ref.register_observer(
            self.ref(), set_schedulers_fun_name, _tell=True, _wait=False)
        if not self._schedulers:
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
