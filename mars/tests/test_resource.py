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

import importlib
import os
import tempfile
import time

import pytest

from ..resource import Resource, ZeroResource

_v1_cpu_stat_first = "8678870951786"
_v1_cpu_stat_last = "8679429771672"

# just a fragment of real cpu.stat
_v2_cpu_stat_first = """
usage_usec 8678870951
"""
_v2_cpu_stat_last = """
usage_usec 8679429771
"""

# just a fragment of real memory.stat
_v1_memory_stat_content = """
cache 489275392
rss 218181632
mapped_file 486768640
swap 0
inactive_anon 486744064
active_anon 218103808
inactive_file 2457600
active_file 73728
hierarchical_memory_limit 1073741824
"""

_v2_memory_current_content = "218181632\n"
_v2_memory_max_content = "1073741824\n"


def test_stats():
    from mars import resource

    resource = importlib.reload(resource)
    resource.cpu_percent()

    mem_stats = resource.virtual_memory()
    assert mem_stats.available >= 0
    assert mem_stats.total >= 0
    assert mem_stats.percent >= 0
    assert mem_stats.used >= 0
    assert mem_stats.free >= 0

    cpu_usage = resource.cpu_percent()
    time.sleep(0.1)
    assert cpu_usage >= 0

    resource.disk_io_usage()
    time.sleep(0.1)
    recv_speed, send_speed = resource.disk_io_usage()
    assert recv_speed >= 0
    assert send_speed >= 0

    curdir = os.path.dirname(os.path.abspath(__file__))
    resource.disk_io_usage(curdir)
    time.sleep(0.1)
    usage = resource.disk_io_usage(curdir)
    if usage is not None:
        assert usage.reads >= 0
        assert usage.writes >= 0

    resource.net_io_usage()
    time.sleep(0.1)
    recv_speed, send_speed = resource.net_io_usage()
    assert recv_speed >= 0
    assert send_speed >= 0


def test_use_process_stats():
    from mars import resource

    cpu_total = resource.cpu_count()
    mem_total = resource.virtual_memory().total
    try:
        os.environ["MARS_USE_PROCESS_STAT"] = "1"
        os.environ["MARS_CPU_TOTAL"] = str(cpu_total)
        os.environ["MARS_MEMORY_TOTAL"] = str(mem_total)

        resource = importlib.reload(resource)
        resource.cpu_percent()
        time.sleep(0.5)

        mem_stats = resource.virtual_memory()
        assert mem_stats.available >= 0
        assert mem_stats.total >= 0
        assert mem_stats.percent >= 0
        assert mem_stats.used >= 0
        assert mem_stats.free >= 0

        cpu_usage = resource.cpu_percent()
        assert cpu_usage >= 0
        cpu_usage = resource.cpu_percent()
        assert cpu_usage >= 0
    finally:
        del os.environ["MARS_USE_PROCESS_STAT"]
        del os.environ["MARS_CPU_TOTAL"]
        del os.environ["MARS_MEMORY_TOTAL"]
        importlib.reload(resource)


@pytest.mark.parametrize("cgroup_ver", ["v1", "v2"])
def test_use_c_group_stats(cgroup_ver):
    from mars import resource

    def write_tmp_text_file(prefix, content):
        fd, file_name = tempfile.mkstemp(prefix)
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return file_name

    v1_cpu_acct_path = write_tmp_text_file(
        "test-mars-res-cgroup-v1-cpu-", _v1_cpu_stat_first
    )
    v1_mem_stat_path = write_tmp_text_file(
        "test-mars-res-cgroup-v1-mem-", _v1_memory_stat_content
    )
    v2_cpu_stat_path = write_tmp_text_file(
        "test-mars-res-cgroup-v2-cpu-", _v2_cpu_stat_first
    )
    v2_mem_cur_path = write_tmp_text_file(
        "test-mars-res-cgroup-v2-cpu-", _v2_memory_current_content
    )
    v2_mem_max_path = write_tmp_text_file(
        "test-mars-res-cgroup-v2-cpu-", _v2_memory_max_content
    )

    old_is_cgroup_v2 = resource._is_cgroup_v2
    old_v1_cpu_acct_file = resource.CGROUP_V1_CPU_ACCT_FILE
    old_v1_mem_stat_file = resource.CGROUP_V1_MEM_STAT_FILE
    old_v2_cpu_stat_file = resource.CGROUP_V2_CPU_STAT_FILE
    old_v2_mem_current_file = resource.CGROUP_V2_MEM_CURRENT_FILE
    old_v2_mem_max_file = resource.CGROUP_V2_MEM_MAX_FILE
    old_shm_path = resource._shm_path
    try:
        os.environ["MARS_USE_CGROUP_STAT"] = "1"

        resource = importlib.reload(resource)
        if cgroup_ver == "v1":
            resource.CGROUP_V1_CPU_ACCT_FILE = v1_cpu_acct_path
            resource.CGROUP_V1_MEM_STAT_FILE = v1_mem_stat_path
            resource._is_cgroup_v2 = False
        else:
            resource.CGROUP_V2_CPU_STAT_FILE = v2_cpu_stat_path
            resource.CGROUP_V2_MEM_CURRENT_FILE = v2_mem_cur_path
            resource.CGROUP_V2_MEM_MAX_FILE = v2_mem_max_path
            resource._is_cgroup_v2 = True
        resource._shm_path = None

        assert resource.cpu_percent() is None
        time.sleep(0.5)
        with open(v1_cpu_acct_path, "w") as f:
            f.write(_v1_cpu_stat_last)
        with open(v2_cpu_stat_path, "w") as f:
            f.write(_v2_cpu_stat_last)
        assert resource.cpu_percent() > 50
        assert resource.cpu_percent() < 150

        mem_stats = resource.virtual_memory()
        assert mem_stats.total == 1073741824
        assert mem_stats.used == 218181632
    finally:
        resource._is_cgroup_v2 = old_is_cgroup_v2
        resource._shm_path = old_shm_path
        resource.CGROUP_V1_CPU_ACCT_FILE = old_v1_cpu_acct_file
        resource.CGROUP_V1_MEM_STAT_FILE = old_v1_mem_stat_file
        resource.CGROUP_V2_CPU_STAT_FILE = old_v2_cpu_stat_file
        resource.CGROUP_V2_MEM_CURRENT_FILE = old_v2_mem_current_file
        resource.CGROUP_V2_MEM_MAX_FILE = old_v2_mem_max_file

        del os.environ["MARS_USE_CGROUP_STAT"]

        os.unlink(v1_cpu_acct_path)
        os.unlink(v1_mem_stat_path)
        os.unlink(v2_cpu_stat_path)
        os.unlink(v2_mem_cur_path)
        os.unlink(v2_mem_max_path)

        importlib.reload(resource)


def test_resource():
    assert Resource(num_cpus=1) + Resource(num_cpus=1) == Resource(num_cpus=2)
    assert Resource(num_cpus=1) + Resource(num_gpus=1) + Resource(
        mem_bytes=1024**3
    ) == Resource(num_cpus=1, num_gpus=1, mem_bytes=1024**3)
    assert -Resource(num_cpus=1, num_gpus=1, mem_bytes=1024**3) == Resource(
        num_cpus=-1, num_gpus=-1, mem_bytes=-(1024**3)
    )
    assert Resource(num_cpus=-1) < ZeroResource
    assert Resource(num_gpus=-1) < ZeroResource
    assert Resource(mem_bytes=-1) < ZeroResource
    assert Resource(num_cpus=1, num_gpus=1, mem_bytes=-(1024**3)) < ZeroResource
    assert Resource(num_cpus=1, num_gpus=1, mem_bytes=1024**3) > Resource(
        num_cpus=10, num_gpus=1, mem_bytes=1024
    )
    assert Resource(num_cpus=1, num_gpus=10, mem_bytes=1024**3) > Resource(
        num_cpus=10, num_gpus=1, mem_bytes=1024**3
    )
    assert Resource(num_cpus=100, num_gpus=10, mem_bytes=1024**3) > Resource(
        num_cpus=10, num_gpus=10, mem_bytes=1024**3
    )
    assert Resource(num_cpus=100, num_gpus=10, mem_bytes=1024) - Resource(
        num_cpus=10, num_gpus=20, mem_bytes=512
    ) == Resource(num_cpus=90, num_gpus=-10, mem_bytes=512)
