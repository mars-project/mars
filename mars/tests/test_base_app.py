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

import argparse
import unittest

from mars.base_app import BaseApplication
from mars.config import Config


class Test(unittest.TestCase):
    def testOptions(self):
        config = Config()
        config.register_option('config.str', None)
        config.register_option('config.int', 0)

        app = BaseApplication()
        handled_options = app._handle_options(
            ['-Dconfig.int=1024', '-Dconfig.str=config_val', '-s', '1.2.3.4'], config)
        self.assertListEqual(handled_options, ['-s', '1.2.3.4'])
        self.assertEqual(config.config.str, 'config_val')
        self.assertEqual(config.config.int, 1024)

    def testParseArgs(self):
        parser = argparse.ArgumentParser(description='TestService')
        app = BaseApplication()
        app.config_args(parser)

        task_detail = """
        {
          "cluster": {
            "scheduler": ["sch1", "sch2"],
            "worker": ["worker1", "worker2"],
            "web": ["web1"]
          },
          "task": {
            "type": "worker",
            "index": 0
          }
        }
        """

        env = {
            'MARS_LOAD_MODULES': 'extra.module',
            'MARS_TASK_DETAIL': task_detail,
        }
        args = app.parse_args(parser, [], env)
        self.assertEqual(args.advertise, 'worker1')
        self.assertEqual(args.schedulers, 'sch1,sch2')
        self.assertIn('extra.module', args.load_modules)
