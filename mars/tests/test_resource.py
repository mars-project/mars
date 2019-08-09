# Copyright 1999-2018 Alibaba Group Holding Ltd.
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
import tempfile
import time
import unittest

from mars.compat import reload_module

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


class Test(unittest.TestCase):
    def testStats(self):
        from mars import resource
        resource = reload_module(resource)
        resource.cpu_percent()

        mem_stats = resource.virtual_memory()
        self.assertGreaterEqual(mem_stats.available, 0)
        self.assertGreaterEqual(mem_stats.total, 0)
        self.assertGreaterEqual(mem_stats.percent, 0)
        self.assertGreaterEqual(mem_stats.used, 0)
        self.assertGreaterEqual(mem_stats.free, 0)

        cpu_usage = resource.cpu_percent()
        time.sleep(0.1)
        self.assertGreaterEqual(cpu_usage, 0)

        resource.disk_io_usage()
        time.sleep(0.1)
        recv_speed, send_speed = resource.disk_io_usage()
        self.assertGreaterEqual(recv_speed, 0)
        self.assertGreaterEqual(send_speed, 0)

        resource.net_io_usage()
        time.sleep(0.1)
        recv_speed, send_speed = resource.net_io_usage()
        self.assertGreaterEqual(recv_speed, 0)
        self.assertGreaterEqual(send_speed, 0)

    def testUseProcessStats(self):
        from mars import resource

        cpu_total = resource.cpu_count()
        mem_total = resource.virtual_memory().total
        try:
            os.environ['MARS_USE_PROCESS_STAT'] = '1'
            os.environ['MARS_CPU_TOTAL'] = str(cpu_total)
            os.environ['MARS_MEMORY_TOTAL'] = str(mem_total)

            resource = reload_module(resource)
            resource.cpu_percent()

            mem_stats = resource.virtual_memory()
            self.assertGreaterEqual(mem_stats.available, 0)
            self.assertGreaterEqual(mem_stats.total, 0)
            self.assertGreaterEqual(mem_stats.percent, 0)
            self.assertGreaterEqual(mem_stats.used, 0)
            self.assertGreaterEqual(mem_stats.free, 0)

            cpu_usage = resource.cpu_percent()
            self.assertGreaterEqual(cpu_usage, 0)
        finally:
            del os.environ['MARS_USE_PROCESS_STAT']
            del os.environ['MARS_CPU_TOTAL']
            del os.environ['MARS_MEMORY_TOTAL']
            reload_module(resource)

    def testUseCGroupStats(self):
        from mars import resource
        fd, mem_stat_path = tempfile.mkstemp(prefix='test-mars-res-')
        with os.fdopen(fd, 'w') as f:
            f.write(_memory_stat_content)

        old_stat_file = resource.CGROUP_MEM_STAT_FILE
        try:
            os.environ['MARS_MEM_USE_CGROUP_STAT'] = '1'
            resource = reload_module(resource)
            resource.CGROUP_MEM_STAT_FILE = mem_stat_path

            mem_stats = resource.virtual_memory()
            self.assertEqual(mem_stats.total, 1073741824)
            self.assertEqual(mem_stats.used, 707457024)
        finally:
            resource.CGROUP_MEM_STAT_FILE = old_stat_file
            del os.environ['MARS_MEM_USE_CGROUP_STAT']
            os.unlink(mem_stat_path)
            reload_module(resource)
