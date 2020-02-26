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
from mars.tests.core import aio_case
from mars.utils import lazy_import

torch_installed = lazy_import('torch', globals=globals()) is not None


@unittest.skipIf(not torch_installed, 'pytorch not installed')
@aio_case
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

    def testDistributedRunPyTorchDataset(self):
        import mars.tensor as mt

        service_ep = 'http://127.0.0.1:' + self.web_port
        with new_session(service_ep) as sess:
            data = mt.random.rand(1000, 32, dtype='f4', chunk_size=100)
            labels = mt.random.randint(0, 1, (1000, 10), dtype='f4', chunk_size=100)
            data.execute(name='data', session=sess)
            labels.execute(name='labels', session=sess)

            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'dataset_sample.py')
            self.assertEqual(run_pytorch_script(
                path, n_workers=2, command_argv=['multiple'],
                port=9945, session=sess
            )['status'], 'ok')
