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

import functools

from ...actors import Distributor
from ...distributor import MarsDistributor


def gen_distributor(scheduler_n_process, worker_n_process):
    class LocalClusterDistributor(Distributor):
        def __init__(self, n_process):
            super().__init__(n_process)
            self._scheduler_distributor = MarsDistributor(scheduler_n_process, 's:h1:')
            self._worker_distributor = MarsDistributor(worker_n_process, 'w:0:')

        @staticmethod
        def _is_worker_uid(uid):
            return isinstance(uid, str) and uid.startswith('w:')

        @functools.lru_cache(100)
        def distribute(self, uid):
            if self._is_worker_uid(uid):
                return self._worker_distributor.distribute(uid) + scheduler_n_process

            return self._scheduler_distributor.distribute(uid)

        def make_same_process(self, uid, uid_rel, delta=0):
            if self._is_worker_uid(uid_rel):
                return self._worker_distributor.make_same_process(uid, uid_rel, delta=delta)
            return self._scheduler_distributor.make_same_process(uid, uid_rel, delta=delta)

    return LocalClusterDistributor(scheduler_n_process + worker_n_process)
