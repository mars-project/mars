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

import unittest

from mars.executor import Executor
from mars.tensor.fuse.cupy import _evaluate
from mars.tensor.datasource import ones
from mars.tensor import sqrt
from mars.optimizes.runtime.core import RuntimeOptimizer


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('cupy')

    def testElementwise(self):
        t1 = ones((10000, 5000), chunk_size=500, gpu=True)
        t2 = ones(5000, chunk_size=500, gpu=True)
        t = (t1 - t2) / sqrt(t2 * (1 - t2) * len(t2))

        g = t.build_graph(tiled=True)
        RuntimeOptimizer(g, self.executor._engine).optimize([], False)
        self.assertTrue(any(n.op.__class__.__name__ == 'TensorCpFuseChunk' for n in g))

        c = next(n for n in g if n.op.__class__.__name__ == 'TensorCpFuseChunk')
        self.assertGreater(len(_evaluate(c)), 1)
