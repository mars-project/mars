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
from numbers import Integral

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from mars.config import option_context
from mars.dataframe.expressions.utils import decide_chunk_sizes


@unittest.skipIf(pd is None, 'pandas not installed')
class Test(unittest.TestCase):
    def testDecideChunks(self):
        with option_context() as options:
            options.tensor.chunk_store_limit = 64

            memory_usage = pd.Series([8, 22.2, 4, 2, 11.2], index=list('abcde'))

            shape = (10, 5)
            nsplit = decide_chunk_sizes(shape, None, memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_chunk_sizes(shape, {0: 4}, memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_chunk_sizes(shape, (2, 3), memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_chunk_sizes(shape, (10, 3), memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            options.tensor.chunk_store_limit = 20

            shape = (10, 5)
            nsplit = decide_chunk_sizes(shape, None, memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_chunk_sizes(shape, {1: 3}, memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_chunk_sizes(shape, (2, 3), memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_chunk_sizes(shape, (10, 3), memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

