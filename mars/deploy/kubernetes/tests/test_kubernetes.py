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

import glob
import os
import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from distutils.spawn import find_executable

import numpy as np
import pytest

from .... import tensor as mt
from ....tests.core import mock
from .. import new_cluster
from ..config import HostPathVolumeConfig

try:
    from kubernetes import config as k8s_config, client as k8s_client
except ImportError:
    k8s_client = k8s_config = None

MARS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(mt.__file__)))
TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
DOCKER_ROOT = os.path.join(os.path.dirname(TEST_ROOT), "docker")

kube_available = (
    find_executable("kubectl") is not None
    and find_executable("docker") is not None
    and k8s_config is not None
)


def _collect_coverage():
    dist_coverage_path = os.path.join(MARS_ROOT, ".dist-coverage")
    if os.path.exists(dist_coverage_path):
        # change ownership of coverage files
        if find_executable("sudo"):
            proc = subprocess.Popen(
                [
                    "sudo",
                    "-n",
                    "chown",
                    "-R",
                    f"{os.geteuid()}:{os.getegid()}",
                    dist_coverage_path,
                ],
                shell=False,
            )
            proc.wait()

        # rewrite paths in coverage result files
        for fn in glob.glob(os.path.join(dist_coverage_path, ".coverage.*")):
            if "COVERAGE_FILE" in os.environ:
                new_cov_file = os.environ["COVERAGE_FILE"] + os.path.basename(
                    fn
                ).replace(".coverage", "")
            else:
                new_cov_file = fn.replace(".dist-coverage" + os.sep, "")
            shutil.copyfile(fn, new_cov_file)
        shutil.rmtree(dist_coverage_path)


def _build_docker_images(use_test_docker_file=True):
    image_name = "mars-test-image:" + uuid.uuid1().hex
    try:
        if use_test_docker_file:
            proc = subprocess.Popen(
                ["docker", "build", "-f", "Dockerfile.test", "-t", image_name, "."],
                cwd=TEST_ROOT,
            )
        else:
            proc = subprocess.Popen(
                [
                    "docker",
                    "build",
                    "-f",
                    os.path.join(DOCKER_ROOT, "Dockerfile"),
                    "-t",
                    image_name,
                    ".",
                ],
                cwd=MARS_ROOT,
            )
        if proc.wait() != 0:
            raise SystemError("Executing docker build failed.")

        if use_test_docker_file:
            proc = subprocess.Popen(
                [
                    "docker",
                    "run",
                    "-v",
                    MARS_ROOT + ":/mnt/mars",
                    image_name,
                    "/srv/build_ext.sh",
                ]
            )
            if proc.wait() != 0:
                raise SystemError("Executing docker run failed.")
    except:  # noqa: E722
        _remove_docker_image(image_name)
        raise
    return image_name


def _remove_docker_image(image_name, raises=True):
    if "CI" not in os.environ:
        # delete image iff in CI environment
        return
    proc = subprocess.Popen(["docker", "rmi", "-f", image_name])
    if proc.wait() != 0 and raises:
        raise SystemError("Executing docker rmi failed.")


def _load_docker_env():
    if os.path.exists("/var/run/docker.sock") or not shutil.which("minikube"):
        return

    proc = subprocess.Popen(["minikube", "docker-env"], stdout=subprocess.PIPE)
    proc.wait(30)
    for line in proc.stdout:
        line = line.decode().split("#", 1)[0]
        line = line.strip()  # type: str | bytes
        export_pos = line.find("export")
        if export_pos < 0:
            continue
        line = line[export_pos + 6 :].strip()
        var, value = line.split("=", 1)
        os.environ[var] = value.strip('"')


@contextmanager
def _start_kube_cluster(use_test_docker_file=True, **kwargs):
    _load_docker_env()
    image_name = _build_docker_images(use_test_docker_file=use_test_docker_file)

    temp_spill_dir = tempfile.mkdtemp(prefix="test-mars-k8s-")
    api_client = k8s_config.new_client_from_config()
    kube_api = k8s_client.CoreV1Api(api_client)

    cluster_client = None
    try:
        if use_test_docker_file:
            extra_volumes = [
                HostPathVolumeConfig("mars-src-path", "/mnt/mars", MARS_ROOT)
            ]
            pre_stop_command = ["rm", "/tmp/stopping.tmp"]
        else:
            extra_volumes = []
            pre_stop_command = None

        cluster_client = new_cluster(
            api_client,
            image=image_name,
            worker_spill_paths=[temp_spill_dir],
            extra_volumes=extra_volumes,
            pre_stop_command=pre_stop_command,
            timeout=600,
            log_when_fail=True,
            **kwargs,
        )

        assert cluster_client.endpoint is not None

        pod_items = kube_api.list_namespaced_pod(cluster_client.namespace).to_dict()

        log_processes = []
        for item in pod_items["items"]:
            log_processes.append(
                subprocess.Popen(
                    [
                        "kubectl",
                        "logs",
                        "-f",
                        "-n",
                        cluster_client.namespace,
                        item["metadata"]["name"],
                    ]
                )
            )

        yield

        if use_test_docker_file:
            # turn off service processes with grace to get coverage data
            procs = []
            pod_items = kube_api.list_namespaced_pod(cluster_client.namespace).to_dict()
            for item in pod_items["items"]:
                p = subprocess.Popen(
                    [
                        "kubectl",
                        "exec",
                        "-n",
                        cluster_client.namespace,
                        item["metadata"]["name"],
                        "--",
                        "/srv/graceful_stop.sh",
                    ]
                )
                procs.append(p)
            for p in procs:
                p.wait()

        [p.terminate() for p in log_processes]
    finally:
        shutil.rmtree(temp_spill_dir)
        if cluster_client:
            try:
                cluster_client.stop(wait=True, timeout=20)
            except TimeoutError:
                pass
        _collect_coverage()
        _remove_docker_image(image_name, False)


@pytest.mark.parametrize("use_test_docker_file", [True, False])
@pytest.mark.skipif(not kube_available, reason="Cannot run without kubernetes")
def test_run_in_kubernetes(use_test_docker_file):
    with _start_kube_cluster(
        supervisor_cpu=0.5,
        supervisor_mem="1G",
        worker_cpu=0.5,
        worker_mem="1G",
        worker_cache_mem="64m",
        extra_labels={"mars-test/group": "test-label-name"},
        extra_env={"MARS_K8S_GROUP_LABELS": "mars-test/group"},
        use_test_docker_file=use_test_docker_file,
    ):
        a = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
        b = mt.ones((100, 100), chunk_size=20) * 2 * 1 + 1
        c = (a * b * 2 + 1).sum()
        r = c.execute().fetch()

        expected = (np.ones(a.shape) * 2 * 1 + 1) ** 2 * 2 + 1
        np.testing.assert_array_equal(r, expected.sum())


@pytest.mark.skipif(not kube_available, reason="Cannot run without kubernetes")
@mock.patch(
    "kubernetes.client.CoreV1Api.create_namespaced_replication_controller",
    new=lambda *_, **__: None,
)
@mock.patch(
    "kubernetes.client.AppsV1Api.create_namespaced_deployment",
    new=lambda *_, **__: None,
)
def test_create_timeout():
    _load_docker_env()
    api_client = k8s_config.new_client_from_config()

    cluster = None
    try:
        extra_vol_config = HostPathVolumeConfig("mars-src-path", "/mnt/mars", MARS_ROOT)
        with pytest.raises(TimeoutError):
            cluster = new_cluster(
                api_client,
                image="pseudo_image",
                supervisor_cpu=0.5,
                supervisor_mem="1G",
                worker_cpu=0.5,
                worker_mem="1G",
                extra_volumes=[extra_vol_config],
                timeout=1,
            )
    finally:
        if cluster:
            cluster.stop(wait=True)
        _collect_coverage()
