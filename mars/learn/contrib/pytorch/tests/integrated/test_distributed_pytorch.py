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
from mars.utils import lazy_import

torch_installed = lazy_import('torch', globals=globals()) is not None


@unittest.skipIf(not torch_installed, 'pytorch not installed')
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


class TestAPI(LearnIntegrationTestBase):
    def testDistributedRunPyTorchScript(self):
        import mars.tensor as mt
        import numpy as np
        from mars.context_ import Context

        scheduler_ep = '127.0.0.1:' + self.scheduler_port
        service_ep = 'http://127.0.0.1:' + self.web_port
        ctx = Context(scheduler_endpoint=scheduler_ep)
        with new_session(service_ep).as_default() as sess:
            session_id = sess.session_id
            t = mt.random.rand(10, 10, chunk_size=3)
            r = t.execute(name='test')
            self.assertEqual(t.key, ctx.get_tileable_key_by_name(session_id, 'test'))
            np.testing.assert_array_equal(r, ctx.get_tileable_data(session_id, t.key))
            np.testing.assert_array_equal(r[slice(4, 7), slice(4, 8)],
                                          ctx.get_tileable_data(session_id, t.key, indexes=[slice(4, 7), slice(4, 8)]))
