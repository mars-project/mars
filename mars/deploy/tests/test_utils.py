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

from mars.deploy.utils import load_service_config_file

_cwd = os.path.abspath(os.getcwd())


@pytest.mark.parametrize('cwd', [_cwd, os.path.dirname(_cwd)])
def test_load_service_config(cwd):
    old_cwd = os.getcwd()
    try:
        os.chdir(cwd)
        cfg = load_service_config_file(
            os.path.join(os.path.dirname(__file__), 'inherit_test_cfg2.yml'))

        assert 'services' in cfg
        assert cfg['test_list'] == ['item1', 'item2', 'item3']
        assert set(cfg['test_dict'].keys()) == {'key1', 'key2', 'key3'}
        assert set(cfg['test_dict']['key2'].values()) == {'val2_modified'}
    finally:
        os.chdir(old_cwd)
