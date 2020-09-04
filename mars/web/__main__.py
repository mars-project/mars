#!/usr/bin/env python
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

import gevent.monkey
gevent.monkey.patch_all(thread=False, select=False)

import argparse  # noqa: E402
import logging   # noqa: E402
import random    # noqa: E402
import time      # noqa: E402

from ..base_app import BaseApplication, arg_deprecated_action  # noqa: E402
from ..errors import StartArgumentError                        # noqa: E402

logger = logging.getLogger(__name__)


class WebApplication(BaseApplication):
    service_description = 'Mars Web'

    def __init__(self):
        super().__init__()
        self.mars_web = None
        self.require_pool = False

    def config_args(self, parser):
        super().config_args(parser)
        parser.add_argument('--ui-port', help=argparse.SUPPRESS, action=arg_deprecated_action('-p'))

    def create_scheduler_discoverer(self):
        super().create_scheduler_discoverer()
        if self.scheduler_discoverer is None:
            raise StartArgumentError('Either schedulers or url of kv store is required.')

    def main_loop(self):
        try:
            self.start()
            self._running = True
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def start(self):
        from .server import MarsWeb
        if MarsWeb is None:
            self.mars_web = None
            logger.warning('Mars UI cannot be loaded. Please check if necessary components are installed.')
        else:
            host = self.args.host
            port_arg = self.args.ui_port or self.args.port
            ui_port = int(port_arg) if port_arg else None

            schedulers = self.scheduler_discoverer.get_schedulers()
            logger.debug('Obtained schedulers: %r', schedulers)
            if not schedulers:
                raise KeyError('No scheduler is available')
            scheduler_ep = random.choice(schedulers)
            self.mars_web = MarsWeb(host=host, port=ui_port, scheduler_ip=scheduler_ep)
            self.mars_web.start()
            if self.args.advertise is not None:
                self.advertise_endpoint = self.args.advertise.split(':', 1)[0] + ':' + str(self.mars_web.port)
            else:
                self.advertise_endpoint = host + ':' + str(self.mars_web.port)

    def stop(self):
        if self.mars_web:
            self.mars_web.stop()


main = WebApplication()

if __name__ == '__main__':
    main()
