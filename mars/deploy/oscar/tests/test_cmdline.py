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
import os
import subprocess
import sys
import time
from typing import List

import numpy as np
import pytest

import mars.tensor as mt
from mars.core.session import new_session
from mars.deploy.oscar.cmdline import OscarCommandRunner
from mars.deploy.oscar.worker import WorkerCommandRunner
from mars.services import NodeRole
from mars.services.cluster import ClusterAPI
from mars.tests import flaky
from mars.utils import get_next_port, kill_process_tree


class _ProcessExitedException(Exception):
    pass


def _wait_supervisor_ready(supervisor_proc: subprocess.Popen, timeout=120):
    start_time = time.time()
    supervisor_pid = supervisor_proc.pid
    while True:
        if supervisor_proc.poll() is not None:
            raise _ProcessExitedException

        try:
            ep_file_name = OscarCommandRunner._build_endpoint_file_path(pid=supervisor_pid)
            with open(ep_file_name, 'r') as ep_file:
                return ep_file.read().strip()
        except:  # noqa: E722  # pylint: disable=bare-except
            if time.time() - start_time > timeout:
                raise
            pass
        finally:
            time.sleep(0.1)


def _wait_worker_ready(supervisor_addr, worker_procs: List[subprocess.Popen],
                       n_supervisors=1, timeout=120):
    async def wait_for_workers():
        start_time = time.time()
        while True:
            if any(proc.poll() is not None for proc in worker_procs):
                raise _ProcessExitedException

            try:
                cluster_api = await ClusterAPI.create(supervisor_addr)
                sv_info = await cluster_api.get_nodes_info(role=NodeRole.SUPERVISOR, resource=True)
                worker_info = await cluster_api.get_nodes_info(role=NodeRole.WORKER, resource=True)
                if len(sv_info) >= n_supervisors and len(worker_info) >= len(worker_procs):
                    break
            except:  # noqa: E722  # pylint: disable=bare-except
                if time.time() - start_time > timeout:
                    raise
                pass
            finally:
                await asyncio.sleep(0.1)

    loop = asyncio.new_event_loop()
    loop.run_until_complete(wait_for_workers())


_test_port_cache = dict()


def _get_labelled_port(label=None, create=True):
    test_name = os.environ['PYTEST_CURRENT_TEST']
    if (test_name, label) not in _test_port_cache:
        if create:
            _test_port_cache[(test_name, label)] = get_next_port()
        else:
            return None
    return _test_port_cache[(test_name, label)]


supervisor_cmd_start = [sys.executable, '-m', 'mars.deploy.oscar.supervisor']
worker_cmd_start = [sys.executable, '-m', 'mars.deploy.oscar.worker']
start_params = {
    'bare_start': [
        supervisor_cmd_start,
        worker_cmd_start + [
            '--config-file', os.path.join(os.path.dirname(__file__), 'local_test_config.yml'),
        ],
        False
    ],
    'with_supervisors': [
        supervisor_cmd_start + [
            '-e', lambda: f'127.0.0.1:{_get_labelled_port("supervisor")}',
            '-w', lambda: str(_get_labelled_port("web"))
        ],
        worker_cmd_start + [
            '-e', lambda: f'127.0.0.1:{get_next_port(occupy=True)}',
            '-s', lambda: f'127.0.0.1:{_get_labelled_port("supervisor")}',
            '--config-file', os.path.join(os.path.dirname(__file__), 'local_test_config.yml')
        ],
        True
    ],
}


def _reload_args(args):
    return [arg if not callable(arg) else arg() for arg in args]


@pytest.mark.parametrize('supervisor_args,worker_args,use_web_addr',
                         list(start_params.values()), ids=list(start_params.keys()))
@flaky(rerun_filter=lambda *args: issubclass(args[0][0], _ProcessExitedException))
def test_cmdline_run(supervisor_args, worker_args, use_web_addr):
    sv_proc = w_procs = None
    try:
        sv_args = _reload_args(supervisor_args)
        sv_proc = subprocess.Popen(sv_args, env=os.environ.copy())

        oscar_port = _get_labelled_port('supervisor', create=False)
        if not oscar_port:
            oscar_ep = _wait_supervisor_ready(sv_proc)
        else:
            oscar_ep = f'127.0.0.1:{oscar_port}'

        if use_web_addr:
            host = oscar_ep.rsplit(':', 1)[0]
            api_ep = f'http://{host}:{_get_labelled_port("web", create=False)}'
        else:
            api_ep = oscar_ep

        w_procs = [subprocess.Popen(
            _reload_args(worker_args), env=os.environ.copy()) for _ in range(2)]
        _wait_worker_ready(oscar_ep, w_procs)

        new_session(api_ep, default=True)
        data = np.random.rand(10, 10)
        res = mt.tensor(data, chunk_size=5).sum().execute().fetch()
        np.testing.assert_almost_equal(res, data.sum())
    finally:
        ep_file_name = OscarCommandRunner._build_endpoint_file_path(pid=sv_proc.pid)
        try:
            os.unlink(ep_file_name)
        except OSError:
            pass

        w_procs = w_procs or []
        for proc in w_procs + [sv_proc]:
            if not proc:
                continue
            proc.terminate()
            try:
                proc.wait(3)
            except subprocess.TimeoutExpired:
                kill_process_tree(proc.pid)


def test_parse_args():
    parser = argparse.ArgumentParser(description='TestService')
    app = WorkerCommandRunner()
    app.config_args(parser)

    task_detail = """
    {
      "cluster": {
        "supervisor": ["sv1", "sv2"],
        "worker": ["worker1", "worker2"]
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
        'MARS_CACHE_MEM_SIZE': '20M',
        'MARS_PLASMA_DIRS': '/dev/shm',
        'MARS_SPILL_DIRS': '/tmp'
    }
    args = app.parse_args(parser, ['-p', '10324'], env)
    assert args.host == 'worker1'
    assert args.endpoint == 'worker1:10324'
    assert args.supervisors == 'sv1,sv2'
    assert 'extra.module' in args.load_modules
    assert app.config['storage']['plasma'] == {
        'store_memory': '20M',
        'plasma_directory': '/dev/shm',
    }
    assert app.config['storage']['filesystem'] == {
            'root_dirs': '/tmp',
            'level': 'DISK',
    }
