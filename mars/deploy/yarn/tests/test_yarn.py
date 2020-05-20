# -*- coding: utf-8 -*-
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

import glob
import logging
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
import unittest
from distutils.spawn import find_executable

import numpy as np
from numpy.testing import assert_array_equal

import mars.tensor as mt
from mars.deploy.yarn import new_cluster

logger = logging.getLogger(__name__)
MARS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(mt.__file__)))


@unittest.skipIf(not os.environ.get('HADOOP_HOME'), 'Only run when hadoop is installed')
class Test(unittest.TestCase):
    def tearDown(self):
        time.sleep(5)
        dist_coverage_path = os.path.join(MARS_ROOT, '.dist-coverage')
        if os.path.exists(dist_coverage_path):
            # change ownership of coverage files
            if find_executable('sudo'):
                proc = subprocess.Popen(['sudo', '-n', 'chown', '-R', '%d:%d' % (os.geteuid(), os.getegid()),
                                         dist_coverage_path], shell=False)
                proc.wait()

            # rewrite paths in coverage result files
            for fn in glob.glob(os.path.join(dist_coverage_path, '.coverage.*')):
                cov_db = sqlite3.connect(fn)
                c = cov_db.cursor()
                c.execute('UPDATE file SET path=REPLACE(path, \'%s\', \'\')' % (MARS_ROOT + os.path.sep))
                cov_db.commit()
                cov_db.close()

                if 'COVERAGE_FILE' in os.environ:
                    new_cov_file = os.environ['COVERAGE_FILE'] \
                                   + os.path.basename(fn).replace('.coverage', '')
                else:
                    new_cov_file = fn.replace('.dist-coverage' + os.sep, '')
                shutil.copyfile(fn, new_cov_file)
            shutil.rmtree(dist_coverage_path)

    def _runYarnTestWithEnv(self, env_path, timeout):
        cluster = None

        coverage_result = os.path.join(MARS_ROOT, '.dist-coverage', '.coverage')
        cov_dir = os.path.join(MARS_ROOT, '.dist-coverage')
        os.makedirs(cov_dir, exist_ok=True)
        os.chmod(cov_dir, 0o777)
        try:
            log_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yarn-logging.conf')

            cmd_tmpl = '"{executable}" -m coverage run --source=%s/mars --rcfile=%s/.coveragerc' \
                % (MARS_ROOT, MARS_ROOT)
            extra_env = {'COVERAGE_FILE': coverage_result, 'COVERAGE_PROCESS_START': '%s/.coveragerc' % MARS_ROOT}
            cluster = new_cluster(env_path, timeout=timeout, extra_env=extra_env, log_config=log_config_file,
                                  scheduler_extra_args='-Dscheduler.default_cpu_usage=0',
                                  worker_extra_args='--ignore-avail-mem',
                                  worker_cache_mem='64m', log_when_fail=True, cmd_tmpl=cmd_tmpl)
            self.assertIsNotNone(cluster.endpoint)

            check_time = time.time()
            while cluster.session.count_workers() == 0:
                time.sleep(1)
                if time.time() - check_time > 5:
                    raise SystemError('Worker not ready')

            a = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
            b = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
            c = (a * b * 2 + 1).sum()
            r = cluster.session.run(c, timeout=600)

            expected = (np.ones(a.shape) * 2 * 1 + 1) ** 2 * 2 + 1
            assert_array_equal(r, expected.sum())
        finally:
            if cluster is not None:
                cluster.stop()

    def testRunWithCondaEnv(self):
        self._runYarnTestWithEnv('conda://' + os.environ['CONDA_PREFIX'], 600)

    def testRunWithPackedEnv(self):
        import conda_pack
        temp_dir = os.environ.get('MARS_YARN_TEST_DIR')
        clean_after_test = False
        if temp_dir is None:
            clean_after_test = True
            temp_dir = tempfile.mkdtemp(prefix='test-mars-yarn-')
        else:
            os.makedirs(temp_dir, exist_ok=True)

        packed_env_file = os.path.join(temp_dir, 'mars-test-env.tar.gz')
        if not os.path.exists(packed_env_file):
            try:
                conda_pack.pack(output=packed_env_file, ignore_editable_packages=True)
            except conda_pack.CondaPackException:
                logger.exception('Failed to pack environment, this test will be skipped')
                return

        try:
            self._runYarnTestWithEnv(packed_env_file, 1200)
        finally:
            if clean_after_test:
                shutil.rmtree(temp_dir)

    def testCreateTimeout(self):
        cluster = None
        try:
            env_path = 'conda://' + os.environ['CONDA_PREFIX']
            log_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yarn-logging.conf')

            with self.assertRaises(TimeoutError):
                cluster = new_cluster(env_path, log_config=log_config_file, worker_cache_mem='64m',
                                      log_when_fail=True, timeout=1)
        finally:
            if cluster is not None:
                cluster.stop()
