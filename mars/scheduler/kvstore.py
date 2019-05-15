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


class KVStoreActor(SchedulerActor):
    """
    Actor handling reading and writing to an external KV store.
    """
    def __init__(self):
        super(KVStoreActor, self).__init__()
        self._store = kvstore.get(options.kv_store)

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())
        super(KVStoreActor, self).post_create()

    def read(self, item, recursive=False, sort=False):
        return self._store.read(item, recursive=recursive, sort=sort)

    def read_batch(self, items, recursive=False, sort=False):
        return [self._store.read(item, recursive=recursive, sort=sort) for item in items]

    def write(self, key, value):
        return self._store.write(key, value)

    def write_batch(self, items):
        wrap = lambda x: (x,) if not isinstance(x, tuple) else x
        [self.write(*wrap(it)) for it in items]

    def delete(self, key, dir=False, recursive=False, silent=False):
        try:
            return self._store.delete(key, dir=dir, recursive=recursive)
        except KeyError:
            if not silent:
                raise
