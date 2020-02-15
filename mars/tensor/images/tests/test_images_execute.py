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

import os
import unittest
import tempfile

import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None

from mars.tests.core import TestBase, ExecutorForTest
from mars.tensor.images import imread


@unittest.skipIf(not Image, 'Pillow not installed')
class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.executor = ExecutorForTest()

    def testImreadExecution(self):
        with tempfile.TemporaryDirectory() as tempdir:
            raws = []
            for i in range(10):
                array = np.random.randint(0, 256, 2500, dtype=np.uint8).reshape((50, 50))
                raws.append(array)
                im = Image.fromarray(array)
                im.save(os.path.join(tempdir, 'random_{}.png'.format(i)))
            # Single image
            t = imread(os.path.join(tempdir, 'random_0.png'))
            res = self.executor.execute_tensor(t, concat=True)[0]
            np.testing.assert_array_equal(res, raws[0])

            t2 = imread(os.path.join(tempdir, 'random_*.png'))
            res = self.executor.execute_tensor(t2, concat=True)[0]
            np.testing.assert_array_equal(np.sort(res, axis=0), np.sort(raws, axis=0))

            t3 = imread(os.path.join(tempdir, 'random_*.png'), chunk_size=4)
            res = self.executor.execute_tensor(t3, concat=True)[0]
            np.testing.assert_array_equal(np.sort(res, axis=0), np.sort(raws, axis=0))

            t4 = imread(os.path.join(tempdir, 'random_*.png'), chunk_size=4)
            res = self.executor.execute_tensor(t4, concat=True)[0]
            np.testing.assert_array_equal(np.sort(res, axis=0), np.sort(raws, axis=0))
