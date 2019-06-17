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

import sys
import unittest

import numpy as np

from mars.deploy.local.core import new_cluster

@unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
class Test(unittest.TestCase):
    def testMutableTensor(self):
        with new_cluster(scheduler_n_process=2, worker_n_process=2,
                         shared_memory='20M') as cluster:
            session = cluster.session

            mut = session.create_mutable_tensor("test", (4, 4), dtype=np.int32, chunk_size=3)
            mut[1:4, 2] = 8
            mut[2:4] = np.arange(8).reshape(2, 4)
            mut[1] = np.arange(4).reshape(4)
            result = session.fetch(mut.seal())

            expected = np.zeros((4, 4), dtype=np.int32)
            expected[1:4, 2] = 8
            expected[2:4] = np.arange(8).reshape(2, 4)
            expected[1] = np.arange(4).reshape(4)

            np.testing.assert_array_equal(result, expected)
