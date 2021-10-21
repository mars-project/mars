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
import logging
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from distutils.spawn import find_executable

import numpy as np
import pytest

from .... import tensor as mt
from ....tests.core import flaky, require_hadoop
from ...yarn import new_cluster

logger = logging.getLogger(__name__)
MARS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(mt.__file__)))


def _collect_coverage():
    time.sleep(5)
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
            cov_db = sqlite3.connect(fn)
            c = cov_db.cursor()
            c.execute(
                f"UPDATE file SET path=REPLACE(path, '{MARS_ROOT + os.path.sep}', '')"
            )
            cov_db.commit()
            cov_db.close()

            if "COVERAGE_FILE" in os.environ:
                new_cov_file = os.environ["COVERAGE_FILE"] + os.path.basename(
                    fn
                ).replace(".coverage", "")
            else:
                new_cov_file = fn.replace(".dist-coverage" + os.sep, "")
            shutil.copyfile(fn, new_cov_file)
        shutil.rmtree(dist_coverage_path)


def _run_yarn_test_with_env(env_path, timeout):
    cluster = None

    coverage_result = os.path.join(MARS_ROOT, ".dist-coverage", ".coverage")
    cov_dir = os.path.join(MARS_ROOT, ".dist-coverage")
    os.makedirs(cov_dir, exist_ok=True)
    os.chmod(cov_dir, 0o777)
    try:
        log_config_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "yarn-logging.conf"
        )

        cmd_tmpl = (
            '"{executable}" -m coverage run --source=%s/mars --rcfile=%s/setup.cfg'
            % (MARS_ROOT, MARS_ROOT)
        )
        extra_env = {
            "COVERAGE_FILE": coverage_result,
            "COVERAGE_PROCESS_START": f"{MARS_ROOT}/setup.cfg",
        }
        cluster = new_cluster(
            env_path,
            timeout=timeout,
            worker_cpu=1,
            worker_mem="1G",
            extra_env=extra_env,
            log_config=log_config_file,
            extra_args=f"--config-file {MARS_ROOT}/mars/deploy/yarn/tests/test_yarn_config.yml",
            log_when_fail=True,
            cmd_tmpl=cmd_tmpl,
        )
        assert cluster.endpoint is not None

        check_time = time.time()
        while cluster.session.get_total_n_cpu() == 0:
            time.sleep(1)
            if time.time() - check_time > 5:
                raise SystemError("Worker not ready")

        a = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
        b = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
        r = (a * b * 2 + 1).sum().execute().fetch()

        expected = (np.ones(a.shape) * 2 * 1 + 1) ** 2 * 2 + 1
        np.testing.assert_array_equal(r, expected.sum())
    finally:
        if cluster is not None:
            cluster.stop()
        _collect_coverage()


@require_hadoop
@flaky(max_runs=3)
def test_run_with_conda_env():
    _run_yarn_test_with_env("conda://" + os.environ["CONDA_PREFIX"], 600)


@require_hadoop
@flaky(max_runs=3)
def test_run_with_packed_env():
    import conda_pack

    temp_dir = os.environ.get("MARS_YARN_TEST_DIR")
    clean_after_test = False
    if temp_dir is None:
        clean_after_test = True
        temp_dir = tempfile.mkdtemp(prefix="test-mars-yarn-")
    else:
        os.makedirs(temp_dir, exist_ok=True)

    packed_env_file = os.path.join(temp_dir, "mars-test-env.tar.gz")
    if not os.path.exists(packed_env_file):
        try:
            conda_pack.pack(output=packed_env_file, ignore_editable_packages=True)
        except conda_pack.CondaPackException:
            logger.exception("Failed to pack environment, this test will be skipped")
            return

    try:
        _run_yarn_test_with_env(packed_env_file, 1200)
    finally:
        if clean_after_test:
            shutil.rmtree(temp_dir)


@require_hadoop
@flaky(max_runs=3)
def test_create_timeout():
    cluster = None
    try:
        env_path = "conda://" + os.environ["CONDA_PREFIX"]
        log_config_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "yarn-logging.conf"
        )

        with pytest.raises(TimeoutError):
            cluster = new_cluster(
                env_path,
                log_config=log_config_file,
                worker_cpu=1,
                worker_mem="1G",
                worker_cache_mem="64m",
                log_when_fail=True,
                timeout=1,
            )
    finally:
        if cluster is not None:
            cluster.stop()
        _collect_coverage()
