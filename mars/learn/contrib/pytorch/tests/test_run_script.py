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

import pytest

from .....utils import lazy_import
from .. import run_pytorch_script


torch_installed = lazy_import("torch") is not None


@pytest.mark.skipif(not torch_installed, reason="pytorch not installed")
def test_distributed_run_py_torch_script(setup_cluster):
    sess = setup_cluster
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytorch_sample.py")
    assert (
        run_pytorch_script(
            path, n_workers=2, command_argv=["multiple"], port=9945, session=sess
        ).fetch()["status"]
        == "ok"
    )
