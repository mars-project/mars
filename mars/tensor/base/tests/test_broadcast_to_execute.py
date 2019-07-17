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


import unittest

import numpy as np

from mars.executor import Executor
from mars.tensor.datasource import tensor
from mars.tensor.base import broadcast_to


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')

    def testBroadcastToExecution(self):
        raw = np.random.random((10, 5, 1))
        arr = tensor(raw, chunk_size=2)
        arr2 = broadcast_to(arr, (5, 10, 5, 6))

        res = self.executor.execute_tensor(arr2, concat=True)

        self.assertTrue(np.array_equal(res[0], np.broadcast_to(raw, (5, 10, 5, 6))))