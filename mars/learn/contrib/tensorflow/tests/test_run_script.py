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

try:
    import tensorflow
except ImportError:
    tensorflow = None

from .. import run_tensorflow_script


@pytest.mark.skipif(tensorflow is None, reason="tensorflow not installed")
def test_local_run_tensor_flow_script(setup_cluster):
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tf_distributed_sample.py"
    )
    assert (
        run_tensorflow_script(path, n_workers=2, command_argv=["multiple"]).fetch()[
            "status"
        ]
        == "ok"
    )

    with pytest.raises(ValueError):
        run_tensorflow_script(path, n_workers=0)

    with pytest.raises(ValueError):
        run_tensorflow_script(path, 2, n_ps=-1)
