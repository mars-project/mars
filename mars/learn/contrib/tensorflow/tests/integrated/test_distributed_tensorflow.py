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
import os

from mars.learn.tests.integrated.base import LearnIntegrationTestBase
from mars.learn.contrib.tensorflow import run_tensorflow_script
from mars.session import new_session

try:
    import tensorflow
except ImportError:
    tensorflow = None


@unittest.skipIf(tensorflow is None, 'tensorflow not installed')
class Test(LearnIntegrationTestBase):
    def testDistributedRunTensorFlowScript(self):
        service_ep = 'http://127.0.0.1:' + self.web_port
        timeout = 120 if 'CI' in os.environ else -1
        with new_session(service_ep) as sess:
            run_kwargs = {'timeout': timeout}
            self.assertEqual(run_tensorflow_script(
                '../tf_test.py', n_workers=2, command_argv=['multiple'],
                port=3222, session=sess, run_kwargs=run_kwargs
            )['status'], 'ok')
