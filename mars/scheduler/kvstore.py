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

from .utils import SchedulerActor
from .. import kvstore
from ..config import options

logger = logging.getLogger(__name__)


class _SyncEtcdActor(SchedulerActor):
    """
    Internally hold by KVStoreActor, sync the write option
    """

    def __init__(self):
        super(_SyncEtcdActor, self).__init__()

        self._store = kvstore.get(options.kv_store)

    def read(self, item, recursive=False, sort=False):
        return self._store.read(item, recursive=recursive, sort=sort)

    def write(self, key, value):
        return self._store.write(key, value)

    def write_batch(self, items):
        wrap = lambda x: (x,) if not isinstance(x, tuple) else x
        [self.write(*wrap(it)) for it in items]

    def delete(self, key, dir=False, recursive=False):
        return self._store.delete(key, dir=dir, recursive=recursive)


class KVStoreActor(SchedulerActor):
    def __init__(self, recover=False):
        super(KVStoreActor, self).__init__()

        self._store = kvstore.LocalKVStore()
        self._sync_ref = None

        # if recover, means the KVStoreActor is recovered by some fault tolerance mechanism,
        # at this particular moment, if the item does not exist,
        # we will try to read from etcd
        self._recover = recover

    @classmethod
    def default_name(cls):
        return 's:%s' % cls.__name__

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        if isinstance(kvstore.get(options.kv_store), kvstore.EtcdKVStore):
            self._sync_ref = self.ctx.create_actor(_SyncEtcdActor)

    def pre_destroy(self):
        self.ctx.destroy_actor(self._sync_ref)

    def read(self, item, recursive=False, sort=False, silent=False):
        try:
            # directly read data from the memory kvstore
            return self._store.read(item, recursive=recursive, sort=sort)
        except KeyError:
            if self._recover and self._sync_ref:
                result = self._sync_ref.read(item, recursive=recursive, sort=sort)
                if result.dir:
                    self._store.write(item, result.value)
                else:
                    for child in result.children:
                        self._store.write(child.key, child.value)
                return result
            elif not silent:
                raise

    def read_batch(self, items, silent=False):
        wrap = lambda x: (x,) if not isinstance(x, tuple) else x
        return [self.read(*wrap(it), **dict(silent=silent)) for it in items]

    def write(self, key, value=None, ttl=None, dir=False):
        # write into memory kvstore first
        self._store.write(key, value=value, ttl=ttl, dir=dir)
        # tell _SyncEtcdActor to sync data into etcd asynchronously
        if self._sync_ref:
            self._sync_ref.write(key, value, _tell=True, _wait=False)

    def write_batch(self, items):
        # write into memory kvstore first
        for it in items:
            args = (it,) if not isinstance(it, tuple) else it
            self._store.write(*args)
        # tell _SyncEtcdActor to sync data into etcd asynchronously
        if self._sync_ref:
            self._sync_ref.write_batch(items, _tell=True, _wait=False)

    def delete(self, key, dir=False, recursive=False):
        # delete memory kvstore first
        self._store.delete(key, dir=dir, recursive=recursive)
        # tell _SyncEtcdActor to delete data asynchronously
        if self._sync_ref:
            self._sync_ref.delete(key, dir=dir, recursive=recursive, _tell=True)
