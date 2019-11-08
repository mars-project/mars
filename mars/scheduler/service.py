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

import json
import logging

from .. import kvstore
from ..config import options
from .session import SessionManagerActor
from .resource import ResourceActor
from .chunkmeta import ChunkMetaActor
from .kvstore import KVStoreActor
from .node_info import NodeInfoActor
from .utils import SchedulerClusterInfoActor


logger = logging.getLogger(__name__)


class SchedulerService(object):
    def __init__(self):
        self._cluster_info_ref = None
        self._session_manager_ref = None
        self._assigner_ref = None
        self._resource_ref = None
        self._chunk_meta_ref = None
        self._kv_store_ref = None
        self._node_info_ref = None
        self._result_receiver_ref = None

    def start(self, endpoint, discoverer, pool, distributed=True):
        """
        there are two way to start a scheduler
        1) if options.kv_store is specified as an etcd address, the endpoint will be written
        into kv-storage to indicate that this scheduler is one the schedulers,
        and the etcd is used as a service discover.
        2) if options.kv_store is not an etcd address, there will be only one scheduler
        """
        kv_store = kvstore.get(options.kv_store)
        kv_store.write('/schedulers/%s' % endpoint, dir=True)

        if not isinstance(kv_store, kvstore.LocalKVStore):
            # set etcd as service discover
            logger.info('Mars Scheduler started with kv store %s.', options.kv_store)
            # create KVStoreActor when there is a distributed KV store
            self._kv_store_ref = pool.create_actor(KVStoreActor, uid=KVStoreActor.default_uid())
        else:
            # single scheduler
            logger.info('Mars Scheduler started in standalone mode.')

        # create ClusterInfoActor
        self._cluster_info_ref = pool.create_actor(
            SchedulerClusterInfoActor, discoverer, distributed=distributed,
            uid=SchedulerClusterInfoActor.default_uid())
        # create ChunkMetaActor
        self._chunk_meta_ref = pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())
        # create SessionManagerActor
        self._session_manager_ref = pool.create_actor(
            SessionManagerActor, uid=SessionManagerActor.default_uid())
        # create ResourceActor
        self._resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_uid())
        # create NodeInfoActor
        self._node_info_ref = pool.create_actor(NodeInfoActor, uid=NodeInfoActor.default_uid())
        kv_store.write('/schedulers/%s/meta' % endpoint,
                       json.dumps(self._resource_ref.get_workers_meta()))

    def stop(self, pool):
        pool.destroy_actor(self._resource_ref)
