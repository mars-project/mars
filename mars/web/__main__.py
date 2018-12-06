#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import gevent.monkey
gevent.monkey.patch_all(thread=False)

import logging
import time

from ..base_app import BaseApplication
from ..errors import StartArgumentError

logger = logging.getLogger(__name__)


class WebApplication(BaseApplication):
    def __init__(self):
        super(WebApplication, self).__init__()
        self.mars_web = None
        self.require_pool = False

    def config_args(self, parser):
        parser.add_argument('--ui-port', help='port of Mars UI')
        parser.add_argument('-s', '--schedulers', help='endpoint of scheduler, when single scheduler '
                            'and etcd is not available')

    def validate_arguments(self):
        if not self.args.schedulers and not self.args.kv_store:
            raise StartArgumentError('Either schedulers or url of kv store is required.')

    def main_loop(self):
        try:
            self.start_service()
            while True:
                time.sleep(0.1)
        finally:
            self.stop_service()

    def start_service(self):
        from .server import MarsWeb
        if MarsWeb is None:
            self.mars_web = None
            logger.warning('Mars UI cannot be loaded. Please check if necessary components are installed.')
        else:
            ui_port = int(self.args.ui_port) if self.args.ui_port else None
            scheduler_ip = self.args.schedulers or None
            self.mars_web = MarsWeb(port=ui_port, scheduler_ip=scheduler_ip)
            self.mars_web.start()

    def stop_service(self):
        if self.mars_web:
            self.mars_web.stop()


main = WebApplication()

if __name__ == '__main__':
    main()
