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

import json

from .. import kvstore
from ..config import options
from ..cluster_info import ClusterInfoActor
from .session import SessionManagerActor
from .resource import ResourceActor
from .kvstore import KVStoreActor
from ..node_info import NodeInfoActor


class SchedulerService(object):
    def __init__(self):
        self._cluster_info_ref = None
        self._session_manager_ref = None
        self._resource_ref = None
        self._kv_store_ref = None
        self._node_info_ref = None

    def start(self, endpoint, pool):
        """
        there are two way to start a scheduler
        1) if options.kv_store is specified as an etcd address, the endpoint will be written
        into kv-storage to indicate that this scheduler is one the schedulers,
        and the etcd is used as a service discover.
        2) if options.kv_store is not an etcd address, there will be only one scheduler
        """
        kv_store = kvstore.get(options.kv_store)
        if isinstance(kv_store, kvstore.EtcdKVStore):
            # set etcd as service discover
            service_discover_addr = options.kv_store
            schedulers = None
            kv_store.write('/schedulers/%s' % endpoint, dir=True)
        else:
            # single scheduler
            service_discover_addr = None
            schedulers = [endpoint]

        # create ClusterInfoActor
        self._cluster_info_ref = pool.create_actor(
            ClusterInfoActor, schedulers, service_discover_addr, uid=ClusterInfoActor.default_name())
        # create SessionManagerActor
        self._session_manager_ref = pool.create_actor(SessionManagerActor,
            uid=SessionManagerActor.default_name())
        # create ResourceActor
        self._resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_name())
        # create KVStoreActor
        self._kv_store_ref = pool.create_actor(KVStoreActor, uid=KVStoreActor.default_name())
        # create NodeInfoActor
        self._node_info_ref = pool.create_actor(NodeInfoActor, uid=NodeInfoActor.default_name())
        kv_store.write('/schedulers/%s/meta' % endpoint,
                       json.dumps(self._resource_ref.get_workers_meta()))

    def stop(self, pool):
        pool.destroy_actor(self._resource_ref)
