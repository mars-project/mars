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

import pytest

from mars.config import option_context
from mars.remote import run_script
from mars.tests import new_test_session


script1 = b"""
import os
assert os.environ['WORLD_SIZE'] == '2'
"""

script2 = b"""
assert session is not None
"""


@pytest.fixture(scope='module')
def setup():
    sess = new_test_session(default=True)
    with option_context({'show_progress': False}):
        try:
            yield sess
        finally:
            sess.stop_server()


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
