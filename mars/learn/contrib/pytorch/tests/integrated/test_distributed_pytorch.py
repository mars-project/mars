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
from mars.learn.contrib.pytorch import run_pytorch_script
from mars.session import new_session

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'pytorch not installed')
class Test(LearnIntegrationTestBase):
    def testDistributedRunPyTorchScript(self):
        service_ep = 'http://127.0.0.1:' + self.web_port
        timeout = 120 if 'CI' in os.environ else -1
        with new_session(service_ep) as sess:
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'pytorch_sample.py')
            run_kwargs = {'timeout': timeout}
            self.assertEqual(run_pytorch_script(
                path, n_workers=2, command_argv=['multiple'],
                port=9945, session=sess, run_kwargs=run_kwargs
            )['status'], 'ok')