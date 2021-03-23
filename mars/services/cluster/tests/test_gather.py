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
import time

from ..gather import gather_node_env, gather_node_resource, \
    gather_node_states


def test_gather_node_env():
    node_env = gather_node_env()
    band_data = node_env['bands']['cpu-0']
    assert band_data['resources']['cpu'] > 0
    assert band_data['resources']['memory'] > 0


def test_gather_node_resource():
    node_res = gather_node_resource()
    band_res = node_res['cpu-0']
    assert band_res['cpu_total'] >= band_res['cpu_avail']
    assert band_res['memory_total'] >= band_res['memory_avail']


def test_gather_node_states():
    gather_node_states()
    time.sleep(0.1)
    node_states = gather_node_states()
    assert not node_states['disk'].get('partitions')

    curdir = os.path.dirname(os.path.abspath(__file__))
    gather_node_states([curdir])
    time.sleep(0.1)
    node_states = gather_node_states([curdir])
    assert node_states['disk'].get('partitions')
