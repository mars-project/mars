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

import asyncio
import os
import subprocess
import sys
import time

import numpy as np
import pytest

import mars.tensor as mt
from mars.core.session import new_session
from mars.deploy.oscar.cmdline import OscarCommandRunner
from mars.services import NodeRole
from mars.services.cluster import ClusterAPI
from mars.utils import get_next_port, kill_process_tree


def _wait_supervisor_ready(supervisor_pid, timeout=120):
    async def wait_proc():
        start_time = time.time()
        while True:
            try:
                ep_file_name = OscarCommandRunner._build_endpoint_file_path(pid=supervisor_pid)
                with open(ep_file_name, 'r') as ep_file:
                    return ep_file.read().strip()
            except:  # noqa: E722  # pylint: disable=bare-except
                if time.time() - start_time > timeout:
                    raise
                pass
            finally:
                await asyncio.sleep(0.1)

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(wait_proc())


def _wait_worker_ready(supervisor_addr, n_supervisors=1, n_workers=1, timeout=120):
    async def wait_for_workers():
        start_time = time.time()
        while True:
            try:
                cluster_api = await ClusterAPI.create(supervisor_addr)
                sv_info = await cluster_api.get_nodes_info(role=NodeRole.SUPERVISOR, resource=True)
                worker_info = await cluster_api.get_nodes_info(role=NodeRole.WORKER, resource=True)
                if len(sv_info) >= n_supervisors and len(worker_info) >= n_workers:
                    break
            except:  # noqa: E722  # pylint: disable=bare-except
                if time.time() - start_time > timeout:
                    raise
                pass
            finally:
                await asyncio.sleep(0.1)

    loop = asyncio.get_event_loop()
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
start_params = [
    [
        supervisor_cmd_start,
        worker_cmd_start + [
            '--config-file', os.path.join(os.path.dirname(__file__), 'local_test_config.yml')
        ],
        False
    ],
    [
        supervisor_cmd_start + [
            '-e', lambda: f'127.0.0.1:{_get_labelled_port("supervisor")}',
            '-w', lambda: str(_get_labelled_port("web"))
        ],
        worker_cmd_start + [
            '-e', lambda: f'127.0.0.1:{_get_labelled_port("worker")}',
            '-s', lambda: f'127.0.0.1:{_get_labelled_port("supervisor")}',
            '--config-file', os.path.join(os.path.dirname(__file__), 'local_test_config.yml')
        ],
        True
    ],
]
start_labels = [
    'bare_start',
    'with_supervisors',
]


def _reload_args(args):
    return [arg if not callable(arg) else arg() for arg in args]


@pytest.mark.parametrize('supervisor_args,worker_args,use_web_addr',
                         start_params, ids=start_labels)
def test_cmdline_run(supervisor_args, worker_args, use_web_addr):
    sv_proc = w_proc = None
    try:
        sv_args = _reload_args(supervisor_args)
        sv_proc = subprocess.Popen(sv_args, env=os.environ.copy())

        oscar_port = _get_labelled_port('supervisor', create=False)
        if not oscar_port:
            oscar_ep = _wait_supervisor_ready(sv_proc.pid)
        else:
            oscar_ep = f'127.0.0.1:{oscar_port}'

        if use_web_addr:
            host = oscar_ep.rsplit(':', 1)[0]
            api_ep = f'http://{host}:{_get_labelled_port("web", create=False)}'
        else:
            api_ep = oscar_ep

        w_proc = subprocess.Popen(
            _reload_args(worker_args), env=os.environ.copy())
        _wait_worker_ready(oscar_ep)

        new_session(api_ep, default=True)
        data = np.random.rand(10, 10)
        res = mt.tensor(data, chunk_size=5).sum().execute().fetch()
        np.testing.assert_almost_equal(res, data.sum())
    finally:
        for proc in [w_proc, sv_proc]:
            if not proc:
                continue
            proc.terminate()
            proc.wait(3)
            if proc.returncode is None:
                kill_process_tree(proc.pid)
