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
import time

from .. import DiskInfo
from ..gather import gather_node_env, gather_node_resource, gather_node_details


def test_gather_node_env():
    node_env = gather_node_env()
    band_data = node_env["bands"]["numa-0"]
    assert band_data["resources"]["cpu"] > 0
    assert band_data["resources"]["memory"] > 0


def test_gather_node_resource():
    node_res = gather_node_resource()
    band_res = node_res["numa-0"]
    assert band_res["cpu_total"] >= band_res["cpu_avail"]
    assert band_res["memory_total"] >= band_res["memory_avail"]


def test_gather_node_details():
    gather_node_details()
    time.sleep(0.1)
    node_details = gather_node_details()
    assert not node_details["disk"].get("partitions")

    curdir = os.path.dirname(os.path.abspath(__file__))
    gather_node_details(disk_infos=[DiskInfo(path=curdir)])
    time.sleep(0.1)
    node_details = gather_node_details(disk_infos=[DiskInfo(path=curdir)])
    assert node_details["disk"].get("partitions")
