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
import shutil
import pytest
import tempfile


@pytest.fixture
def cleanup_third_party_modules_output():
    output_dir = os.path.join(tempfile.gettempdir(), "test_inject_module_output")
    shutil.rmtree(output_dir, ignore_errors=True)
    yield
    shutil.rmtree(output_dir, ignore_errors=True)


def get_output_filenames():
    return os.listdir(os.path.join(tempfile.gettempdir(), "test_inject_module_output"))
