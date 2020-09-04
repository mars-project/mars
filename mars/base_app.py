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
import json
import logging
import os
import sys

from .actors import create_actor_pool
from .config import options
from .errors import StartArgumentError
from .lib.tblib import pickling_support
from .utils import get_next_port, to_str

pickling_support.install()
logger = logging.getLogger(__name__)

try:
    from pytest_cov.embed import cleanup_on_sigterm
    cleanup_on_sigterm()
except ImportError:  # pragma: no cover
    cleanup_on_sigterm = None


def arg_deprecated_action(new_arg):  # pragma: no cover
    class ArgDeprecated(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            logger.warning('Argument %s deprecated. Use %s instead',
                           '|'.join(self.option_strings), new_arg)
            setattr(namespace, self.dest, values)
    return ArgDeprecated


class BaseApplication(object):
    service_description = ''
    service_logger = logger

    def __init__(self):
        self.args = None
        self.endpoint = None
        self.advertise_endpoint = None
        self.scheduler_discoverer = None
        self.pool = None
        self.n_process = None

        self._running = False

    @staticmethod
    def _handle_options(argv=None, config=None):
        argv = argv or sys.argv[1:]
        new_argv = []
        config = config or options
        for a in argv:
            if not a.startswith('-D'):
                new_argv.append(a)
                continue
            conf, val = a[2:].split('=', 1)
            conf_parts = conf.split('.')
            conf_obj = config
            for g in conf_parts[:-1]:
                conf_obj = getattr(conf_obj, g)
            try:
                setattr(conf_obj, conf_parts[-1], json.loads(val))
            except json.JSONDecodeError:
                setattr(conf_obj, conf_parts[-1], val)
        return new_argv

    def __call__(self, argv=None, config=None):
        return self._main(self._handle_options(argv, config))

    def _main(self, argv=None):
        parser = argparse.ArgumentParser(description=self.service_description)
        self.config_args(parser)
        args = self.args = self.parse_args(parser, argv)

        endpoint = args.endpoint
        if endpoint is not None:
            ep_parts = endpoint.split(':')
            host = args.host = ep_parts[0]
            port = args.port = int(ep_parts[1])
        else:
            host = args.host
            port = args.port

        options.kv_store = args.kv_store if args.kv_store else options.kv_store

        for m in args.load_modules:
            __import__(m, globals(), locals(), [])
        self.service_logger.info('Modules %s loaded', ','.join(args.load_modules))

        self.n_process = 1

        self.config_service()
        self.config_logging()

        host = host or '0.0.0.0'
        if not endpoint and port:
            endpoint = args.endpoint = host + ':' + port

        try:
            self.validate_arguments()
        except StartArgumentError as ex:
            parser.error(f'Failed to start application: {ex}')

        self.advertise_endpoint = None
        if getattr(self, 'require_pool', True):
            self.endpoint, self.pool, self.advertise_endpoint = self._try_create_pool(
                endpoint=endpoint, host=host, port=port, advertise_address=args.advertise)
            self.service_logger.info('%s started at %s.', self.service_description, self.endpoint)

        self.create_scheduler_discoverer()
        try:
            self.main_loop()
        except KeyboardInterrupt:
            pass

    def config_logging(self):
        import logging.config
        import mars
        log_conf = self.args.log_conf or 'logging.conf'

        conf_file_paths = [
            '', os.path.abspath('.'), os.path.dirname(os.path.dirname(mars.__file__))
        ]
        log_configured = False
        for path in conf_file_paths:
            conf_path = os.path.join(path, log_conf) if path else log_conf
            if os.path.exists(conf_path):
                logging.config.fileConfig(conf_path, disable_existing_loggers=False)
                log_configured = True
                break

        if not log_configured:
            log_level = self.args.log_level or self.args.level
            log_format = self.args.log_format or self.args.format
            level = getattr(logging, log_level.upper()) if log_level else logging.INFO
            logging.getLogger('mars').setLevel(level)
            logging.basicConfig(format=log_format)

    def validate_arguments(self):
        pass

    def create_scheduler_discoverer(self):
        from .cluster_info import StaticClusterDiscoverer, KVStoreClusterDiscoverer
        if self.args.kv_store:
            self.scheduler_discoverer = KVStoreClusterDiscoverer(self.args.kv_store)
        elif self.args.schedulers:
            schedulers = to_str(self.args.schedulers).split(',')
            self.scheduler_discoverer = StaticClusterDiscoverer(schedulers)

    def _try_create_pool(self, endpoint=None, host=None, port=None, advertise_address=None):
        pool = None
        if endpoint:
            pool = self.create_pool(address=endpoint, advertise_address=advertise_address)
        else:
            use_port = None
            retrial = 5
            while use_port is None:
                use_port = port or get_next_port()
                try:
                    endpoint = f'{host}:{use_port}'
                    pool = self.create_pool(address=endpoint, advertise_address=advertise_address)
                    break
                except:  # noqa: E722
                    retrial -= 1
                    if retrial == 0:
                        raise

                    if port is None:
                        use_port = None
                    else:
                        raise

        if advertise_address is not None:
            advertise_endpoint = advertise_address.split(':', 1)[0] + ':' + endpoint.rsplit(':', 1)[-1]
        else:
            advertise_endpoint = endpoint

        return endpoint, pool, advertise_endpoint

    def create_pool(self, *args, **kwargs):
        kwargs.update(dict(n_process=self.n_process, backend='gevent'))
        return create_actor_pool(*args, **kwargs)

    def main_loop(self):
        try:
            with self.pool:
                try:
                    self.start()
                    self._running = True
                    while self._running:
                        self.pool.join(1)
                        stopped = []
                        for idx, proc in enumerate(self.pool.processes):
                            if not proc.is_alive():
                                stopped.append(idx)
                        if stopped:
                            self.handle_process_down(stopped)
                except KeyboardInterrupt:
                    pass
                finally:
                    self.stop()
        except:
            logger.exception('Unexpected error occurred in loop')
            raise
        finally:
            self._running = False

    def handle_process_down(self, proc_indices):
        """
        Handle process down event, the default action is to quit
        the whole application. Applications can inherit this method
        to do customized process-level failover.

        :param proc_indices: indices of processes (not pids)
        """
        for idx in proc_indices:
            proc = self.pool.processes[idx]
            self.service_logger.fatal(
                'Process %d exited unpredictably. exitcode=%d', proc.pid, proc.exitcode)
        raise KeyboardInterrupt

    def config_service(self):
        pass

    def config_args(self, parser):
        parser.add_argument('-a', '--advertise', help='advertise ip exposed to other services')
        parser.add_argument('-e', '--endpoint', help='endpoint of the service')
        parser.add_argument('-k', '--kv-store', help='address of kv store service, '
                                                     'for instance, etcd://localhost:4001')
        parser.add_argument('-s', '--schedulers', help='endpoint of schedulers, needed for workers '
                                                       'and webs when kv-store argument is not available, '
                                                       'or when you need to use multiple schedulers '
                                                       'without kv-store')
        parser.add_argument('-H', '--host', help='host of the service, needed when `endpoint` is absent')
        parser.add_argument('-p', '--port', help='port of the service, needed when `endpoint` is absent')
        parser.add_argument('--log-level', help='log level')
        parser.add_argument('--level', help=argparse.SUPPRESS,
                            action=arg_deprecated_action('--log-level'))
        parser.add_argument('--log-format', help='log format')
        parser.add_argument('--format', help=argparse.SUPPRESS,
                            action=arg_deprecated_action('--log-format'))
        parser.add_argument('--log-conf', help='log config file, logging.conf by default')
        parser.add_argument('--log_conf', help=argparse.SUPPRESS,
                            action=arg_deprecated_action('--log-conf'))
        parser.add_argument('--load-modules', nargs='*', help='modules to import')

    def parse_args(self, parser, argv, environ=None):
        environ = environ or os.environ
        args = parser.parse_args(argv)

        args.host = args.host or environ.get('MARS_BIND_HOST')
        args.port = args.port or environ.get('MARS_BIND_PORT')
        args.endpoint = args.endpoint or environ.get('MARS_BIND_ENDPOINT')
        args.advertise = args.advertise or environ.get('MARS_CONTAINER_IP')
        load_modules = []
        for mods in tuple(args.load_modules or ()) + (environ.get('MARS_LOAD_MODULES'),):
            load_modules.extend(mods.split(',') if mods else [])
        load_modules.extend(['mars.executor', 'mars.serialize.protos'])
        args.load_modules = tuple(load_modules)

        if 'MARS_TASK_DETAIL' in environ:
            task_detail = json.loads(environ['MARS_TASK_DETAIL'])
            task_type, task_index = task_detail['task']['type'], task_detail['task']['index']

            args.advertise = args.advertise or task_detail['cluster'][task_type][task_index]
            args.schedulers = args.schedulers or ','.join(task_detail['cluster']['scheduler'])
        return args

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError
