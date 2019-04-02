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
import time
import unittest

from mars.compat import reload_module

_MOCK_NVIDIA_SMI_RESULT = b'''<?xml version="1.0" ?>
<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v10.dtd">
<nvidia_smi_log>
  <driver_version>410.79</driver_version>
  <cuda_version>10.0</cuda_version>
  <attached_gpus>1</attached_gpus>
  <gpu id="00000000:00:04.0">
    <product_name>Tesla K80</product_name>
    <fb_memory_usage>
      <total>1024 MiB</total>
      <used>512 MiB</used>
      <free>512 MiB</free>
    </fb_memory_usage>
    <utilization>
      <gpu_util>10 %</gpu_util>
    </utilization>
    <temperature>
      <gpu_temp>34 C</gpu_temp>
    </temperature>
  </gpu>
</nvidia_smi_log>'''


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

    def testCUDAInfo(self):
        from mars import resource
        from xml.etree import ElementTree

        self.assertTrue(
            (resource.cuda_info() is None and resource.cuda_card_stats() is None) or
            (resource.cuda_info() is not None and resource.cuda_card_stats() is not None)
        )

        try:
            resource._last_nvml_output = ElementTree.fromstring(_MOCK_NVIDIA_SMI_RESULT)
            resource._last_nvml_output_time = time.time()

            cuda_info = resource.cuda_info()
            self.assertEqual(cuda_info.driver_version, '410.79')
            self.assertEqual(cuda_info.cuda_version, '10.0')
            self.assertListEqual(cuda_info.products, ['Tesla K80'])
            self.assertEqual(cuda_info.gpu_count, 1)

            cuda_card_stats = resource.cuda_card_stats()
            self.assertEqual(len(cuda_card_stats), 1)
            self.assertEqual(cuda_card_stats[0].product_name, 'Tesla K80')
            self.assertAlmostEqual(cuda_card_stats[0].temperature, 34)
            self.assertAlmostEqual(cuda_card_stats[0].gpu_usage, 0.1)
            self.assertAlmostEqual(cuda_card_stats[0].fb_mem_info.total, 1024 ** 3)
            self.assertAlmostEqual(cuda_card_stats[0].fb_mem_info.used, 1024 ** 3 / 2)
            self.assertAlmostEqual(cuda_card_stats[0].fb_mem_info.free, 1024 ** 3 / 2)
            self.assertAlmostEqual(cuda_card_stats[0].fb_mem_info.percent, 0.5)
        finally:
            reload_module(resource)
