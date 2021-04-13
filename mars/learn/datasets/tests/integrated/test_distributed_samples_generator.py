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
import sys
import unittest

import numpy as np

try:
    import sklearn
except ImportError:
    sklearn = None

from mars.core import ExecutableTuple
from mars.learn.datasets import make_blobs
from mars.session import new_session
from mars.tests.integrated.base import IntegrationTestBase


@unittest.skipIf(sklearn is None, 'sklearn not installed')
@unittest.skipIf(sys.platform == 'win32', "plasma don't support windows")
class Test(IntegrationTestBase):
    def testDistributedMakeBlobs(self):
        service_ep = 'http://127.0.0.1:' + self.web_port
        timeout = 120 if 'CI' in os.environ else -1
        with new_session(service_ep) as sess:
            run_kwargs = {'timeout': timeout}

            X, y = make_blobs(
                n_samples=100000, n_features=3,
                centers=[[3, 3, 3], [0, 0, 0],
                         [1, 1, 1], [2, 2, 2]],
                cluster_std=[0.2, 0.1, 0.2, 0.2],
                random_state=9)

            X_res, y_res = ExecutableTuple([X, y]).execute(session=sess, **run_kwargs) \
                .fetch(session=sess)
            self.assertIsInstance(X_res, np.ndarray)
            self.assertIsInstance(y_res, np.ndarray)
