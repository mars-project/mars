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

import argparse
import asyncio
import faulthandler
import glob
import importlib
import json
import os
import sys
import tempfile
from typing import List

import psutil

from ..utils import load_service_config_file

_is_windows: bool = sys.platform.startswith('win')

# make sure coverage is handled when starting with subprocess.Popen
if not _is_windows and 'COV_CORE_SOURCE' in os.environ:  # pragma: no cover
    try:
        from pytest_cov.embed import cleanup_on_sigterm
    except ImportError:
        pass
    else:
        cleanup_on_sigterm()


class OscarCommandRunner:
    command_description = None
    node_role = None
    _port_file_prefix = 'mars_service_process'

    def __init__(self):
        faulthandler.enable()

        self.args = None
        self.ports = None
        self.config = {}
        self.pool = None

        self.logging_conf = {}

        self._running = False

    def config_args(self, parser):
        parser.add_argument('-e', '--endpoint', help='endpoint of the service')
        parser.add_argument('-H', '--host', help='host name of the service')
        parser.add_argument('-p', '--ports', help='ports of the service, must equal to'
                                                  'num of processes')
        parser.add_argument('-c', '--config', help='service configuration')
        parser.add_argument('-f', '--config-file', help='configuration file of the service')
        parser.add_argument('-s', '--supervisors',
                            help='endpoint of supervisors, needed for workers and webs '
                                 'when kv-store argument is not available, or when you '
                                 'need to use multiple supervisors without kv-store')
        parser.add_argument('--log-level', help='log level')
        parser.add_argument('--log-format', help='log format')
        parser.add_argument('--log-conf', help='log config file, logging.conf by default')
        parser.add_argument('--load-modules', nargs='*', help='modules to import')
        parser.add_argument('--use-uvloop', help='use uvloop, auto by default')

    def config_logging(self):
        import logging.config
        import mars
        log_conf = self.args.log_conf or 'logging.conf'

        conf_file_paths = [
            '', os.path.abspath('.'), os.path.dirname(os.path.dirname(mars.__file__))
        ]
        for path in conf_file_paths:
            conf_path = os.path.join(path, log_conf) if path else log_conf
            if os.path.exists(conf_path):
                self.logging_conf['file'] = conf_path
                logging.config.fileConfig(conf_path, disable_existing_loggers=False)
                break
        else:
            log_level = self.args.log_level
            level = getattr(logging, log_level.upper()) if log_level else logging.INFO
            logging.getLogger('mars').setLevel(level)
            logging.basicConfig(format=self.args.log_format)
            self.logging_conf.update({'level': log_level, 'format': self.args.log_format})

    @classmethod
    def _build_endpoint_file_path(cls, pid: int = None, asterisk: bool = False):
        pid = pid or os.getpid()
        return os.path.join(
            tempfile.gettempdir(),
            f'{cls._port_file_prefix}.{"*" if asterisk else pid}'
        )

    def _write_supervisor_endpoint_file(self, args):
        file_name = self._build_endpoint_file_path()
        with open(file_name, 'w') as port_file:
            port_file.write(args.endpoint)
        return file_name

    def _collect_supervisors_from_dir(self):
        endpoints = []
        for fn in glob.glob(self._build_endpoint_file_path(asterisk=True)):
            _, pid_str = os.path.basename(fn).rsplit('.', 1)
            # detect if process exists
            if pid_str.isdigit() and not psutil.pid_exists(int(pid_str)):
                continue
            with open(fn, 'r') as ep_file:
                endpoints.append(ep_file.read().strip())
        return endpoints

    @classmethod
    def get_default_config_file(cls):
        mod_file_path = os.path.dirname(importlib.import_module(cls.__module__).__file__)
        return os.path.join(mod_file_path, 'config.yml')

    def parse_args(self, parser, argv, environ=None):
        environ = environ or os.environ
        args = parser.parse_args(argv)

        if args.endpoint is not None and args.host is not None:  # pragma: no cover
            raise ValueError('Cannot specify host and endpoint at the same time')

        if 'MARS_TASK_DETAIL' in environ:
            task_detail = json.loads(environ['MARS_TASK_DETAIL'])
            task_type, task_index = task_detail['task']['type'], task_detail['task']['index']

            args.host = args.host or task_detail['cluster'][task_type][task_index]
            args.supervisors = args.supervisors or ','.join(task_detail['cluster']['supervisor'])

        default_host = '0.0.0.0' if not _is_windows else '127.0.0.1'
        env_host = os.environ.get(
            'MARS_BIND_HOST', os.environ.get('MARS_CONTAINER_IP', default_host))
        args.host = args.host or env_host

        args.ports = args.ports or os.environ.get('MARS_BIND_PORT')
        if args.ports is not None:
            self.ports = [int(p) for p in args.ports.split(',')]

        if args.endpoint is None and len(self.ports or []) == 1:
            args.endpoint = f'{args.host}:{self.ports[0]}'
            self.ports = None

        load_modules = []
        for mods in tuple(args.load_modules or ()) + (environ.get('MARS_LOAD_MODULES'),):
            load_modules.extend(mods.split(',') if mods else [])
        args.load_modules = tuple(load_modules)

        args.use_uvloop = args.use_uvloop or 'auto'

        if args.config is not None:
            self.config = json.loads(args.config)
        else:
            if args.config_file is None:
                args.config_file = self.get_default_config_file()
            self.config = load_service_config_file(args.config_file)

        if args.supervisors is None:
            args.supervisors = ','.join(self._collect_supervisors_from_dir())

        return args

    async def _main(self, argv):
        self.config_logging()

        try:
            pool = self.pool = await self.create_actor_pool()

            await self.start_services()
            self._running = True
            await pool.join()
        except asyncio.CancelledError:
            if self._running:  # pragma: no branch
                await self.stop_services()
            if self.pool:  # pragma: no branch
                await self.pool.stop()

    async def create_actor_pool(self):
        raise NotImplementedError

    async def start_services(self):
        raise NotImplementedError

    async def stop_services(self):
        raise NotImplementedError

    def create_loop(self):
        use_uvloop = self.args.use_uvloop
        if not use_uvloop:
            loop = asyncio.get_event_loop()
        else:
            try:
                import uvloop
                loop = uvloop.new_event_loop()
                asyncio.set_event_loop(loop)
            except ImportError:
                if use_uvloop == 'auto':
                    loop = asyncio.get_event_loop()
                else:  # pragma: no cover
                    raise
        return loop

    def __call__(self, argv: List[str] = None):
        parser = argparse.ArgumentParser(description=self.command_description)
        self.config_args(parser)
        self.args = self.parse_args(parser, argv)

        loop = self.create_loop()
        task = loop.create_task(self._main(argv))

        try:
            loop.run_until_complete(task)
        except KeyboardInterrupt:
            task.cancel()
            loop.run_until_complete(task)
            # avoid displaying exception-unhandled warnings
            task.exception()
