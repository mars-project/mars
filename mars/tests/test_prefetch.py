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

import threading
import time
import unittest

from mars.executor import Executor
from mars.tensor.datasource import ones, TensorOnes


def _slow_tensor_ones(ctx, chunk):
    TensorOnes.execute(ctx, chunk)
    time.sleep(.2)


class MockStorage(object):
    def __init__(self):
        self._data1 = {}
        self._data2 = {}
        self._lock = threading.Lock()
        self._set = threading.Event()

    def get(self, k):
        with self._lock:
            if k not in self._data1 and k in self._data2:
                self._data1[k] = self._data2[k]
                del self._data2[k]
        return self._data1.get(k)

    def __contains__(self, item):
        with self._lock:
            return item in self._data1

    def __getitem__(self, item):
        with self._lock:
            return self._data1[item]

    def __setitem__(self, key, value):
        with self._lock:
            if not self._set.is_set():
                self._data2[key] = value
                self._set.set()
            else:
                self._data1[key] = value

    def __delitem__(self, key):
        pass


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy', storage=MockStorage(), prefetch=True)
        self._raw_tensor_ones = TensorOnes.execute
        self.executor._op_runners[TensorOnes] = _slow_tensor_ones

    def tearDown(self):
        self.executor._op_runners[TensorOnes] = self._raw_tensor_ones

    def testPrefetch(self):
        t1 = ones((10, 8), chunk_size=10)
        t2 = ones((1, 8), chunk_size=10)
        t3 = t1 + t2

        self.executor.execute_tensor(t3)
