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

_cpu_stat_first = '8678870951786'
_cpu_stat_last = '8679429771672'

# just a fragment of real memory.stat
_memory_stat_content = """
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
        os.environ['MARS_USE_PROCESS_STAT'] = '1'
        os.environ['MARS_CPU_TOTAL'] = str(cpu_total)
        os.environ['MARS_MEMORY_TOTAL'] = str(mem_total)

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
        del os.environ['MARS_USE_PROCESS_STAT']
        del os.environ['MARS_CPU_TOTAL']
        del os.environ['MARS_MEMORY_TOTAL']
        importlib.reload(resource)


def test_use_c_group_stats():
    from mars import resource
    fd, cpu_stat_path = tempfile.mkstemp(prefix='test-mars-res-cpu-')
    with os.fdopen(fd, 'w') as f:
        f.write(_cpu_stat_first)
    fd, mem_stat_path = tempfile.mkstemp(prefix='test-mars-res-mem-')
    with os.fdopen(fd, 'w') as f:
        f.write(_memory_stat_content)

    old_cpu_stat_file = resource.CGROUP_CPU_STAT_FILE
    old_mem_stat_file = resource.CGROUP_MEM_STAT_FILE
    old_shm_path = resource._shm_path
    try:
        os.environ['MARS_USE_CGROUP_STAT'] = '1'

        resource = importlib.reload(resource)
        resource.CGROUP_CPU_STAT_FILE = cpu_stat_path
        resource.CGROUP_MEM_STAT_FILE = mem_stat_path
        resource._shm_path = None

        assert resource.cpu_percent() is None
        time.sleep(0.5)
        with open(cpu_stat_path, 'w') as f:
            f.write(_cpu_stat_last)
        assert resource.cpu_percent() > 50
        assert resource.cpu_percent() < 150

        mem_stats = resource.virtual_memory()
        assert mem_stats.total == 1073741824
        assert mem_stats.used == 218181632
    finally:
        resource.CGROUP_CPU_STAT_FILE = old_cpu_stat_file
        resource.CGROUP_MEM_STAT_FILE = old_mem_stat_file
        resource._shm_path = old_shm_path
        del os.environ['MARS_USE_CGROUP_STAT']
        os.unlink(cpu_stat_path)
        os.unlink(mem_stat_path)
        importlib.reload(resource)
