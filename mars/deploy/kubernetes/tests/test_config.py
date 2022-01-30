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

from ..config import (
    NamespaceConfig,
    RoleConfig,
    RoleBindingConfig,
    ServiceConfig,
    EmptyDirVolumeConfig,
    MarsSupervisorsConfig,
    MarsWorkersConfig,
)


def test_simple_objects():
    ns_config_dict = NamespaceConfig("ns_name").build()
    assert ns_config_dict["metadata"]["name"] == "ns_name"

    role_config_dict = RoleConfig(
        "mars-pod-reader", "ns_name", "", "pods", "get,watch,list"
    ).build()
    assert role_config_dict["metadata"]["name"] == "mars-pod-reader"
    assert "get" in role_config_dict["rules"][0]["verbs"]

    role_binding_config_dict = RoleBindingConfig(
        "mars-pod-reader-binding", "ns_name", "mars-pod-reader", "default"
    ).build()
    assert role_binding_config_dict["metadata"]["name"] == "mars-pod-reader-binding"

    service_config_dict = ServiceConfig(
        "mars-test-service", "NodePort", "mars/service-type=marssupervisor", 7103, 7103
    ).build()
    assert service_config_dict["metadata"]["name"] == "mars-test-service"


def test_supervisor_object():
    supervisor_config = MarsSupervisorsConfig(
        1, cpu=2, memory="10g", limit_resources=False, modules=["mars.test_mod"]
    )
    supervisor_config.add_simple_envs(dict(TEST_ENV="test_val"))

    supervisor_config_dict = supervisor_config.build()
    assert supervisor_config_dict["metadata"]["name"] == "marssupervisor"
    assert supervisor_config_dict["spec"]["replicas"] == 1

    container_dict = supervisor_config_dict["spec"]["template"]["spec"]["containers"][0]
    assert int(container_dict["resources"]["requests"]["memory"]) == 10 * 1024**3

    container_envs = dict((p["name"], p) for p in container_dict["env"])
    assert container_envs["TEST_ENV"]["value"] == "test_val"
    assert container_envs["MKL_NUM_THREADS"]["value"] == "2"
    assert container_envs["MARS_CPU_TOTAL"]["value"] == "2"
    assert int(container_envs["MARS_MEMORY_TOTAL"]["value"]) == 10 * 1024**3
    assert container_envs["MARS_LOAD_MODULES"]["value"] == "mars.test_mod"


def test_worker_object():
    worker_config_dict = MarsWorkersConfig(
        4,
        cpu=2,
        memory=10 * 1024**3,
        limit_resources=True,
        memory_limit_ratio=2,
        spill_volumes=[
            "/tmp/spill_vol",
            EmptyDirVolumeConfig("empty-dir", "/tmp/empty"),
        ],
        worker_cache_mem="20%",
        min_cache_mem="10%",
        modules="mars.test_mod",
        mount_shm=True,
    ).build()
    assert worker_config_dict["metadata"]["name"] == "marsworker"
    assert worker_config_dict["spec"]["replicas"] == 4

    container_dict = worker_config_dict["spec"]["template"]["spec"]["containers"][0]
    assert int(container_dict["resources"]["requests"]["memory"]) == 10 * 1024**3
    assert int(container_dict["resources"]["limits"]["memory"]) == 20 * 1024**3

    container_envs = dict((p["name"], p) for p in container_dict["env"])
    assert container_envs["MKL_NUM_THREADS"]["value"] == "2"
    assert container_envs["MARS_CPU_TOTAL"]["value"] == "2"
    assert int(container_envs["MARS_MEMORY_TOTAL"]["value"]) == 10 * 1024**3
    assert container_envs["MARS_LOAD_MODULES"]["value"] == "mars.test_mod"
    assert set(container_envs["MARS_SPILL_DIRS"]["value"].split(":")) == {
        "/tmp/empty",
        "/mnt/hostpath0",
    }
    assert container_envs["MARS_CACHE_MEM_SIZE"]["value"] == "20%"

    volume_list = worker_config_dict["spec"]["template"]["spec"]["volumes"]
    volume_envs = dict((v["name"], v) for v in volume_list)
    assert "empty-dir" in volume_envs
    assert volume_envs["host-path-vol-0"]["hostPath"]["path"] == "/tmp/spill_vol"

    volume_mounts = dict((v["name"], v) for v in container_dict["volumeMounts"])
    assert volume_mounts["empty-dir"]["mountPath"] == "/tmp/empty"
    assert volume_mounts["host-path-vol-0"]["mountPath"] == "/mnt/hostpath0"

    worker_config_dict = MarsWorkersConfig(
        4,
        cpu=2,
        memory=10 * 1024**3,
        limit_resources=False,
        spill_volumes=[
            "/tmp/spill_vol",
            EmptyDirVolumeConfig("empty-dir", "/tmp/empty"),
        ],
        modules="mars.test_mod",
        mount_shm=False,
    ).build()

    volume_list = worker_config_dict["spec"]["template"]["spec"]["volumes"]
    assert "shm-volume" not in volume_list

    container_dict = worker_config_dict["spec"]["template"]["spec"]["containers"][0]
    volume_mounts = dict((v["name"], v) for v in container_dict["volumeMounts"])
    assert "shm-volume" not in volume_mounts
