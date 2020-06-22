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

from io import BytesIO

from mars.tests.core import TestBase
from mars.remote import run_script


script1 = b"""
import os
assert os.environ['WORLD_SIZE'] == '2'
"""

script2 = b"""
assert session is not None
"""


class Test(TestBase):
    def testLocalRunScript(self):
        s = BytesIO(script1)
        self.assertEqual(run_script(
            s, n_workers=2, run_kwargs={'n_parallel': 2}
        )['status'], 'ok')

    def testLocalRunScriptWithExec(self):
        s = BytesIO(script2)
        self.assertEqual(run_script(
            s, n_workers=2, run_kwargs={'n_parallel': 2}, mode='exec'
        )['status'], 'ok')
