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
from io import BytesIO

from mars.remote import run_script
from mars.tests import setup


script1 = b"""
import os
assert os.environ['WORLD_SIZE'] == '2'
"""

script2 = b"""
assert session is not None
"""


setup = setup


def test_local_run_script(setup):
    s = BytesIO(script1)
    assert run_script(
        s, n_workers=2
    )['status'] == 'ok'


def test_local_run_script_with_exec(setup):
    s = BytesIO(script2)
    assert run_script(
        s, n_workers=2, mode='exec'
    )['status'] == 'ok'


def test_run_with_file(setup):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_script.py')
    assert run_script(
        path, n_workers=2
    )['status'] == 'ok'
