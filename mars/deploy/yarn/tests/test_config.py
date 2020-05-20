# -*- coding: utf-8 -*-
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
import unittest

import mars
from mars.deploy.yarn.config import SecurityConfig, AppFileConfig, AppMasterConfig, \
    MarsApplicationConfig, MarsSchedulerConfig, MarsWorkerConfig, MarsWebConfig


class Test(unittest.TestCase):
    def testSimpleObject(self):
        config = SecurityConfig('/path/to/cert.pem', '/path/to/key.pem').build()
        self.assertEqual(config['cert_file'], '/path/to/cert.pem')
        self.assertEqual(config['key_file'], '/path/to/key.pem')

        config = AppFileConfig(source='/path/to/file').build()
        self.assertEqual(config, '/path/to/file')
        config = AppFileConfig(source='/path/to/file', file_type='archive').build()
        self.assertEqual(config['source'], '/path/to/file')
        self.assertEqual(config['type'], 'archive')

        config = AppMasterConfig(security=SecurityConfig('/path/to/cert.pem', '/path/to/key.pem'),
                                 cpu=1, memory='512 MiB').build()
        self.assertEqual(config['security']['cert_file'], '/path/to/cert.pem')
        self.assertEqual(config['security']['key_file'], '/path/to/key.pem')
        self.assertEqual(config['resources']['vcores'], 1)

    def testSchedulerConfig(self):
        config = MarsSchedulerConfig('/path/to/packed.tar.gz', 'mars.test_mod', cpu=2, memory='10 GiB',
                                     env={'TEST_ENV': 'test_val'},
                                     extra_args='-Dscheduler.default_cpu_usage=0').build()
        self.assertEqual(config['files']['mars_env'], '/path/to/packed.tar.gz')
        self.assertIn('mars.deploy.yarn.scheduler', config['script'])

        config_envs = config['env']
        self.assertEqual(config_envs['TEST_ENV'], 'test_val')
        self.assertEqual(config_envs['MKL_NUM_THREADS'], '2')
        self.assertEqual(config_envs['MARS_CPU_TOTAL'], '2')
        self.assertEqual(int(config_envs['MARS_MEMORY_TOTAL']), 10 * 1024 ** 3)
        self.assertEqual(config_envs['MARS_LOAD_MODULES'], 'mars.test_mod')

        config = MarsSchedulerConfig('conda://path/to_env', 'mars.test_mod', cpu=2, memory='10 GiB',
                                     log_config='logging.conf', env={'TEST_ENV': 'test_val'},
                                     extra_args='-Dscheduler.default_cpu_usage=0').build()
        config_envs = config['env']
        self.assertEqual(config_envs['MARS_SOURCE_PATH'],
                         os.path.dirname(os.path.dirname(mars.__file__)))

        config = MarsSchedulerConfig('venv://path/to_env', 'mars.test_mod', cpu=2,
                                     log_config='logging.conf', env={'TEST_ENV': 'test_val'},
                                     extra_args='-Dscheduler.default_cpu_usage=0').build()
        config_envs = config['env']
        self.assertEqual(config_envs['MARS_SOURCE_PATH'],
                         os.path.dirname(os.path.dirname(mars.__file__)))

    def testWorkerConfig(self):
        config = MarsWorkerConfig('/path/to/packed.tar.gz').build()
        self.assertIn('mars.deploy.yarn.worker', config['script'])
        self.assertListEqual(config['depends'], [MarsSchedulerConfig.service_name])

        config = MarsWorkerConfig('/path/to/packed.tar.gz', worker_cache_mem='10g',
                                  spill_dirs=['/spill/dir1', '/spill/dir2']).build()
        config_envs = config['env']
        self.assertEqual(config_envs['MARS_CACHE_MEM_SIZE'], '10g')
        self.assertListEqual(config_envs['MARS_SPILL_DIRS'].split(':'), ['/spill/dir1', '/spill/dir2'])

    def testWebConfig(self):
        config = MarsWebConfig('/path/to/packed.tar.gz').build()
        self.assertIn('mars.deploy.yarn.web', config['script'])
        self.assertListEqual(config['depends'], [MarsSchedulerConfig.service_name])

    def testAppConfig(self):
        scheduler_config = MarsSchedulerConfig('/path/to/packed.tar.gz', 'mars.test_mod',
                                               cpu=2, memory='10 GiB', env={'TEST_ENV': 'test_val'},
                                               extra_args='-Dscheduler.default_cpu_usage=0')
        worker_config = MarsWorkerConfig('/path/to/packed.tar.gz', worker_cache_mem='10g',
                                         spill_dirs=['/spill/dir1', '/spill/dir2'])
        web_config = MarsWebConfig('/path/to/packed.tar.gz')

        config = MarsApplicationConfig(name='config-name', queue='default',
                                       scheduler_config=scheduler_config,
                                       worker_config=worker_config, web_config=web_config).build()
        self.assertEqual(config['name'], 'config-name')
        self.assertEqual(config['queue'], 'default')
