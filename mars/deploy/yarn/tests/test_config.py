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

import os

from .... import __file__ as mars_file
from ..config import (
    SecurityConfig,
    AppFileConfig,
    AppMasterConfig,
    MarsApplicationConfig,
    MarsSupervisorConfig,
    MarsWorkerConfig,
)


def test_simple_object():
    config = SecurityConfig("/path/to/cert.pem", "/path/to/key.pem").build()
    assert config["cert_file"] == "/path/to/cert.pem"
    assert config["key_file"] == "/path/to/key.pem"

    config = AppFileConfig(source="/path/to/file").build()
    assert config == "/path/to/file"
    config = AppFileConfig(source="/path/to/file", file_type="archive").build()
    assert config["source"] == "/path/to/file"
    assert config["type"] == "archive"

    config = AppMasterConfig(
        security=SecurityConfig("/path/to/cert.pem", "/path/to/key.pem"),
        cpu=1,
        memory="512 MiB",
    ).build()
    assert config["security"]["cert_file"] == "/path/to/cert.pem"
    assert config["security"]["key_file"] == "/path/to/key.pem"
    assert config["resources"]["vcores"] == 1


def test_supervisor_config():
    config = MarsSupervisorConfig(
        "/path/to/packed.tar.gz",
        "mars.test_mod",
        cpu=2,
        memory="10 GiB",
        env={"TEST_ENV": "test_val"},
        extra_args="-Dsupervisor.default_cpu_usage=0",
    ).build()
    assert config["files"]["mars_env"] == "/path/to/packed.tar.gz"
    assert "mars.deploy.yarn.supervisor" in config["script"]

    config_envs = config["env"]
    assert config_envs["TEST_ENV"] == "test_val"
    assert config_envs["MKL_NUM_THREADS"] == "2"
    assert config_envs["MARS_CPU_TOTAL"] == "2"
    assert int(config_envs["MARS_MEMORY_TOTAL"]) == 10 * 1024**3
    assert config_envs["MARS_LOAD_MODULES"] == "mars.test_mod"

    config = MarsSupervisorConfig(
        "conda://path/to_env",
        "mars.test_mod",
        cpu=2,
        memory="10 GiB",
        log_config="logging.conf",
        env={"TEST_ENV": "test_val"},
        extra_args="-Dsupervisor.default_cpu_usage=0",
    ).build()
    config_envs = config["env"]
    assert config_envs["MARS_SOURCE_PATH"] == os.path.dirname(
        os.path.dirname(mars_file)
    )

    config = MarsSupervisorConfig(
        "venv://path/to_env",
        "mars.test_mod",
        cpu=2,
        log_config="logging.conf",
        env={"TEST_ENV": "test_val"},
        extra_args="-Dsupervisor.default_cpu_usage=0",
    ).build()
    config_envs = config["env"]
    assert config_envs["MARS_SOURCE_PATH"] == os.path.dirname(
        os.path.dirname(mars_file)
    )


def test_worker_config():
    config = MarsWorkerConfig("/path/to/packed.tar.gz").build()
    assert "mars.deploy.yarn.worker" in config["script"]
    assert config["depends"] == [MarsSupervisorConfig.service_name]

    config = MarsWorkerConfig(
        "/path/to/packed.tar.gz",
        worker_cache_mem="10g",
        spill_dirs=["/spill/dir1", "/spill/dir2"],
    ).build()
    config_envs = config["env"]
    assert config_envs["MARS_CACHE_MEM_SIZE"] == "10g"
    assert config_envs["MARS_SPILL_DIRS"].split(":") == ["/spill/dir1", "/spill/dir2"]


def test_app_config():
    supervisor_config = MarsSupervisorConfig(
        "/path/to/packed.tar.gz",
        "mars.test_mod",
        cpu=2,
        memory="10 GiB",
        env={"TEST_ENV": "test_val"},
        extra_args="-Dsupervisor.default_cpu_usage=0",
    )
    worker_config = MarsWorkerConfig(
        "/path/to/packed.tar.gz",
        worker_cache_mem="10g",
        spill_dirs=["/spill/dir1", "/spill/dir2"],
    )

    config = MarsApplicationConfig(
        name="config-name",
        queue="default",
        supervisor_config=supervisor_config,
        worker_config=worker_config,
    ).build()
    assert config["name"] == "config-name"
    assert config["queue"] == "default"
