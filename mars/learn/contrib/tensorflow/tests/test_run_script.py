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

import pytest

from mars.learn.contrib.tensorflow import run_tensorflow_script
from mars.tests import setup

try:
    import tensorflow
except ImportError:
    tensorflow = None

setup = setup


@pytest.mark.skipif(tensorflow is None, reason='tensorflow not installed')
def test_local_run_tensor_flow_script(setup):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tf_distributed_sample.py')
    assert run_tensorflow_script(
        path, n_workers=1, command_argv=['multiple'],
        port=2222)['status'] == 'ok'

    with pytest.raises(ValueError):
        run_tensorflow_script(path, n_workers=0)

    with pytest.raises(ValueError):
        run_tensorflow_script(path, 2, n_ps=-1)
