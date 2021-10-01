# Copyright 1999-2021 Alibaba Group Holding Ltd.
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

import pytest

from ... import tensor as mt
from ... import dataframe as md
from .. import run_script


script1 = b"""
import os
assert os.environ['WORLD_SIZE'] == '2'
"""

script2 = b"""
assert session is not None
"""

script3 = b"""
from mars.core.operand import Fetch
from mars.deploy.oscar.session import AbstractSession

assert AbstractSession.default is not None
assert isinstance(tensor.op, Fetch)
assert len(tensor.chunks) > 0
assert isinstance(tensor.chunks[0].op, Fetch)
tensor.fetch().sum() == df.fetch()['s'].sum()
"""


def test_local_run_script(setup_cluster):
    s = BytesIO(script1)
    assert run_script(s, n_workers=2).fetch()["status"] == "ok"


def test_local_run_script_with_exec(setup_cluster):
    s = BytesIO(script2)
    assert run_script(s, n_workers=2).fetch()["status"] == "ok"


def test_local_run_script_with_data(setup_cluster):
    s = BytesIO(script3)
    data = {"tensor": mt.arange(10), "df": md.DataFrame({"s": mt.arange(9, 0, -1)})}
    assert run_script(s, data=data, n_workers=1).fetch()["status"] == "ok"

    pytest.raises(TypeError, run_script, s, data=[])


def test_run_with_file(setup_cluster):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_script.py")
    assert run_script(path, n_workers=2).fetch()["status"] == "ok"
