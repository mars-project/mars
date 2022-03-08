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

import concurrent.futures
import os
import subprocess

import psutil
import pytest

from mars.config import option_context
from mars.core.mode import is_kernel_mode, is_build_mode
from mars.lib.aio import stop_isolation
from mars.oscar.backends.router import Router
from mars.oscar.backends.ray.communication import RayServer
from mars.serialization.ray import register_ray_serializers, unregister_ray_serializers
from mars.utils import lazy_import

ray = lazy_import("ray")


@pytest.fixture(scope="module")
def ray_start_regular_shared(request):  # pragma: no cover
    yield from _ray_start_regular(request)


@pytest.fixture
def ray_start_regular(request):  # pragma: no cover
    yield from _ray_start_regular(request)


def _ray_start_regular(request):  # pragma: no cover
    param = getattr(request, "param", {})
    if not param.get("enable", True):
        yield
    else:
        num_cpus = param.get("num_cpus", 64)
        total_memory_mb = num_cpus * 2 * 1024**2
        try:
            register_ray_serializers()
            try:
                job_config = ray.job_config.JobConfig(total_memory_mb=total_memory_mb)
            except TypeError:
                job_config = None
            yield ray.init(num_cpus=num_cpus, job_config=job_config)
        finally:
            ray.shutdown()
            unregister_ray_serializers()
            Router.set_instance(None)
            RayServer.clear()
            if "COV_CORE_SOURCE" in os.environ:
                # Remove this when https://github.com/ray-project/ray/issues/16802 got fixed
                subprocess.check_call(["ray", "stop", "--force"])


@pytest.fixture(scope="module")
def ray_large_cluster_shared(request):  # pragma: no cover
    yield from _ray_large_cluster(request)


@pytest.fixture
def ray_large_cluster(request):  # pragma: no cover
    yield from _ray_large_cluster(request)


def _ray_large_cluster(request):  # pragma: no cover
    param = getattr(request, "param", {})
    num_nodes = param.get("num_nodes", 3)
    num_cpus = param.get("num_cpus", 16)
    try:
        from ray.cluster_utils import Cluster
    except ModuleNotFoundError:
        from ray._private.cluster_utils import Cluster
    cluster = Cluster()
    remote_nodes = []
    for i in range(num_nodes):
        remote_nodes.append(
            cluster.add_node(num_cpus=num_cpus, memory=num_cpus * 2 * 1024**3)
        )
        if len(remote_nodes) == 1:
            ray.init(address=cluster.address)
    register_ray_serializers()
    try:
        yield
    finally:
        unregister_ray_serializers()
        Router.set_instance(None)
        RayServer.clear()
        ray.shutdown()
        cluster.shutdown()
        if "COV_CORE_SOURCE" in os.environ:
            # Remove this when https://github.com/ray-project/ray/issues/16802 got fixed
            subprocess.check_call(["ray", "stop", "--force"])


@pytest.fixture
def stop_ray(request):  # pragma: no cover
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
async def ray_create_mars_cluster(request):
    from mars.deploy.oscar.ray import new_cluster, _load_config

    ray_config = _load_config()
    param = getattr(request, "param", {})
    supervisor_mem = param.get("supervisor_mem", 1 * 1024**3)
    worker_num = param.get("worker_num", 2)
    worker_cpu = param.get("worker_cpu", 2)
    worker_mem = param.get("worker_mem", 256 * 1024**2)
    ray_config.update(param.get("config", {}))
    client = await new_cluster(
        "test_cluster",
        supervisor_mem=supervisor_mem,
        worker_num=worker_num,
        worker_cpu=worker_cpu,
        worker_mem=worker_mem,
        config=ray_config,
    )
    async with client:
        yield client


@pytest.fixture(scope="module")
def _stop_isolation():
    yield
    stop_isolation()


@pytest.fixture(scope="module")
def _new_test_session(_stop_isolation):
    from .deploy.oscar.tests.session import new_test_session

    sess = new_test_session(
        address="test://127.0.0.1", init_local=True, default=True, timeout=300
    )
    with option_context({"show_progress": False}):
        try:
            yield sess
        finally:
            sess.stop_server(isolation=False)


@pytest.fixture(scope="module")
def _new_integrated_test_session(_stop_isolation):
    from .deploy.oscar.tests.session import new_test_session

    sess = new_test_session(
        address="127.0.0.1", init_local=True, n_worker=2, default=True, timeout=300
    )
    with option_context({"show_progress": False}):
        try:
            yield sess
        finally:
            try:
                sess.stop_server(isolation=False)
            except concurrent.futures.TimeoutError:
                subprocesses = psutil.Process().children(recursive=True)
                for proc in subprocesses:
                    proc.terminate()
                for proc in subprocesses:
                    try:
                        proc.wait(1)
                    except (psutil.TimeoutExpired, psutil.NoSuchProcess):
                        pass
                    try:
                        proc.kill()
                    except psutil.NoSuchProcess:
                        pass


@pytest.fixture(scope="module")
def _new_gpu_test_session(_stop_isolation):  # pragma: no cover
    from .deploy.oscar.tests.session import new_test_session
    from .resource import cuda_count

    cuda_devices = list(range(min(cuda_count(), 2)))

    sess = new_test_session(
        address="127.0.0.1",
        init_local=True,
        n_worker=1,
        n_cpu=1,
        cuda_devices=cuda_devices,
        default=True,
        timeout=300,
    )
    with option_context({"show_progress": False}):
        try:
            yield sess
        finally:
            sess.stop_server(isolation=False)


@pytest.fixture
def setup(_new_test_session):
    _new_test_session.as_default()
    yield _new_test_session
    assert not (is_build_mode() or is_kernel_mode())


@pytest.fixture
def setup_cluster(_new_integrated_test_session):
    _new_integrated_test_session.as_default()
    yield _new_integrated_test_session


@pytest.fixture
def setup_gpu(_new_gpu_test_session):  # pragma: no cover
    _new_gpu_test_session.as_default()
    yield _new_test_session
