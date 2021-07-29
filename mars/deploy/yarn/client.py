# -*- coding: utf-8 -*-
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

import logging
import random
import time
import uuid

from ...session import new_session
from ...utils import to_str
from ..utils import wait_services_ready
from .config import MarsApplicationConfig, MarsSupervisorConfig, MarsWorkerConfig

logger = logging.getLogger(__name__)


class YarnClusterClient:
    def __init__(self, skein_client, application_id, endpoint, is_client_managed=False):
        self._skein_client = skein_client
        self._is_client_managed = is_client_managed
        self._application_id = application_id
        self._endpoint = endpoint
        self._session = new_session(endpoint)

    @property
    def session(self):
        return self._session

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def application_id(self):
        return self._application_id

    def stop(self, status='SUCCEEDED'):
        import skein
        try:
            skein_client = skein.Client()
            app_client = skein_client.connect(self._application_id)
            app_client.shutdown(status=status)
            if self._is_client_managed:
                self._skein_client.close()
        except skein.ApplicationNotRunningError:
            pass


def _get_ready_container_count(app_client, svc):
    container_ids = set(c.yarn_container_id for c in app_client.get_containers([svc], ['RUNNING']))
    prefixes = app_client.kv.get_prefix(svc)
    registered_ids = set(to_str(v).rsplit('@', 1)[-1] for v in prefixes.values())
    return len(container_ids.intersection(registered_ids))


def new_cluster(environment=None, supervisor_num=1, supervisor_cpu=None, supervisor_mem=None,
                worker_num=1, worker_cpu=None, worker_mem=None, worker_spill_paths=None,
                worker_cache_mem=None, min_worker_num=None, timeout=None, log_config=None,
                skein_client=None, app_name=None, **kwargs):
    import skein
    from .supervisor import YarnSupervisorCommandRunner

    def _override_envs(src, updates):
        ret = src.copy()
        ret.update(updates)
        return ret

    app_name = app_name or f'mars-app-{uuid.uuid4()}'

    log_when_fail = kwargs.pop('log_when_fail', False)

    supervisor_extra_modules = kwargs.pop('supervisor_extra_modules', None)
    worker_extra_modules = kwargs.pop('worker_extra_modules', None)

    cmd_tmpl = kwargs.pop('cmd_tmpl', None)

    extra_envs = kwargs.pop('extra_env', dict())
    supervisor_extra_env = _override_envs(extra_envs, kwargs.pop('supervisor_extra_env', dict()))
    worker_extra_env = _override_envs(extra_envs, kwargs.pop('worker_extra_env', dict()))

    extra_args = kwargs.pop('extra_args', '')
    supervisor_extra_args = (extra_args + ' ' + kwargs.pop('supervisor_extra_args', '')).strip()
    worker_extra_args = (extra_args + ' ' + kwargs.pop('worker_extra_args', '')).strip()

    supervisor_log_config = kwargs.pop('supervisor_log_config', log_config)
    worker_log_config = kwargs.pop('worker_log_config', log_config)

    supervisor_config = MarsSupervisorConfig(
        instances=supervisor_num, environment=environment, cpu=supervisor_cpu, memory=supervisor_mem,
        modules=supervisor_extra_modules, env=supervisor_extra_env, log_config=supervisor_log_config,
        extra_args=supervisor_extra_args, cmd_tmpl=cmd_tmpl
    )
    worker_config = MarsWorkerConfig(
        instances=worker_num, environment=environment, cpu=worker_cpu, memory=worker_mem,
        spill_dirs=worker_spill_paths, worker_cache_mem=worker_cache_mem, modules=worker_extra_modules,
        env=worker_extra_env, log_config=worker_log_config, extra_args=worker_extra_args,
        cmd_tmpl=cmd_tmpl
    )
    app_config = MarsApplicationConfig(
        app_name, supervisor_config=supervisor_config, worker_config=worker_config)

    skein_client = skein_client or skein.Client()
    app_id = None
    try:
        is_client_managed = skein_client is not None
        app_id = skein_client.submit(app_config.build())

        check_start_time = time.time()
        while True:
            try:
                app_client = skein_client.connect(app_id)
                break
            except skein.ApplicationNotRunningError:  # pragma: no cover
                time.sleep(0.5)
                if timeout and time.time() - check_start_time > timeout:
                    raise

        logger.debug('Application client for %s at %s retrieved', app_id, app_client.address)

        # wait until supervisors and expected num of workers are ready
        min_worker_num = int(min_worker_num or worker_num)
        limits = [supervisor_num, min_worker_num]
        services = [MarsSupervisorConfig.service_name, MarsWorkerConfig.service_name]

        wait_services_ready(services, limits,
                            lambda svc: _get_ready_container_count(app_client, svc),
                            timeout=None if not timeout else timeout - (time.time() - check_start_time))
        web_endpoint_kv = app_client.kv.get_prefix(YarnSupervisorCommandRunner.web_service_name)
        web_endpoint = random.choice([to_str(v).split('@', 1)[0] for v in web_endpoint_kv.values()])
        return YarnClusterClient(skein_client, app_client.id, web_endpoint,
                                 is_client_managed=is_client_managed)
    except:  # noqa: E722
        skein_client = skein.Client()
        try:
            if log_when_fail:
                if app_id is not None:
                    try:
                        app_client = skein_client.connect(app_id)
                        app_client.shutdown(status='FAILED')
                    except skein.ApplicationNotRunningError:
                        pass

                    try:
                        logs = skein_client.application_logs(app_id)
                        logger.error('Error when creating cluster:\n%s', logs.dumps())
                    except ValueError:
                        logger.error('Error when creating cluster and failed to get logs')
                else:
                    logger.error('Error when creating cluster and no logs from cluster')
        finally:
            if app_id is not None:
                skein_client.kill_application(app_id)
        raise
