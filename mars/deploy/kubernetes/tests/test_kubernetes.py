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
import os
import shutil
import subprocess
import tempfile
import unittest
import uuid
from distutils.spawn import find_executable

import numpy as np
from numpy.testing import assert_array_equal

import mars.tensor as mt
from mars.deploy.kubernetes import new_cluster
from mars.deploy.kubernetes.config import HostPathVolumeConfig
from mars.tests.core import mock

try:
    from kubernetes import config as k8s_config, client as k8s_client
except ImportError:
    k8s_client = k8s_config = None

MARS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(mt.__file__)))
TEST_ROOT = os.path.dirname(os.path.abspath(__file__))


def _rewrite_json_coverage(file_name, new_file_name):
    with open(file_name, 'rb') as f:
        content = f.read()
    with open(new_file_name, 'wb') as f:
        f.write(content.replace(b'/mnt/mars', MARS_ROOT.encode()))


def _rewrite_sqlite_coverage(file_name, new_file_name):
    import sqlite3
    shutil.copyfile(file_name, new_file_name)
    conn = None
    try:
        conn = sqlite3.connect(new_file_name)
        cursor = conn.cursor()
        cursor.execute('SELECT id, path FROM file')

        updates = []
        for file_id, file_path in cursor:
            updates.append((file_path.replace('/mnt/mars', MARS_ROOT), file_id))
        cursor = conn.cursor()
        cursor.executemany('UPDATE file SET path=? WHERE id=?', updates)
        conn.commit()
    finally:
        if conn is not None:
            conn.close()


@unittest.skipIf(
    find_executable('kubectl') is None or find_executable('docker') is None
    or k8s_config is None,
    reason='Cannot run without kubernetes')
class Test(unittest.TestCase):
    def tearDown(self):
        kube_coverage_path = os.path.join(MARS_ROOT, '.kube-coverage')
        if os.path.exists(kube_coverage_path):
            # change ownership of coverage files
            if find_executable('sudo'):
                proc = subprocess.Popen(['sudo', '-n', 'chown', '-R', '%d:%d' % (os.geteuid(), os.getegid()),
                                         kube_coverage_path], shell=False)
                proc.wait()

            # rewrite paths in coverage result files
            for fn in glob.glob(os.path.join(kube_coverage_path, '.coverage.*')):
                with open(fn, 'rb') as f:
                    sqlite_header = f.read(6)

                if 'COVERAGE_FILE' in os.environ:
                    new_cov_file = os.environ['COVERAGE_FILE'] \
                                   + os.path.basename(fn).replace('.coverage', '')
                else:
                    new_cov_file = fn.replace('.kube-coverage' + os.sep, '')

                if sqlite_header == b'SQLite':
                    _rewrite_sqlite_coverage(fn, new_cov_file)
                else:
                    _rewrite_json_coverage(fn, new_cov_file)
            shutil.rmtree(kube_coverage_path)

    @classmethod
    def _build_docker_images(cls):
        try:
            cls._docker_image = 'mars-test-image:' + uuid.uuid4().hex
            proc = subprocess.Popen(['docker', 'build',
                                     '-f', 'Dockerfile.test',
                                     '-t', cls._docker_image,
                                     '.'], cwd=TEST_ROOT)
            if proc.wait() != 0:
                raise SystemError('Executing docker build failed.')
            proc = subprocess.Popen(['docker', 'run',
                                     '-v', MARS_ROOT + ':/mnt/mars',
                                     cls._docker_image, '/srv/build_ext.sh'])
            if proc.wait() != 0:
                raise SystemError('Executing docker run failed.')
        except:  # noqa: E722
            cls._remove_docker_image()
            raise

    @classmethod
    def _remove_docker_image(cls, raises=True):
        proc = subprocess.Popen(['docker', 'rmi', '-f', cls._docker_image])
        if proc.wait() != 0 and raises:
            raise SystemError('Executing docker rmi failed.')

    def testRunInKubernetes(self):
        self._build_docker_images()

        temp_spill_dir = tempfile.mkdtemp(prefix='test-mars-k8s-')
        api_client = k8s_config.new_client_from_config()
        kube_api = k8s_client.CoreV1Api(api_client)

        cluster = None
        try:
            extra_vol_config = HostPathVolumeConfig('mars-src-path', '/mnt/mars', MARS_ROOT)
            cluster = new_cluster(api_client, image=self._docker_image,
                                  worker_spill_paths=[temp_spill_dir],
                                  extra_volumes=[extra_vol_config],
                                  pre_stop_command=['rm', '/tmp/stopping.tmp'],
                                  timeout=120, log_when_fail=True)
            self.assertIsNotNone(cluster.endpoint)

            pod_items = kube_api.list_namespaced_pod(cluster.namespace).to_dict()

            log_processes = []
            for item in pod_items['items']:
                log_processes.append(subprocess.Popen(['kubectl', 'logs', '-f', '-n', cluster.namespace,
                                                      item['metadata']['name']]))

            a = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
            b = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
            c = (a * b * 2 + 1).sum()
            r = cluster.session.run(c, timeout=120)

            expected = (np.ones(a.shape) * 2 * 1 + 1) ** 2 * 2 + 1
            assert_array_equal(r, expected.sum())

            # turn off service processes with grace to get coverage data
            procs = []
            for item in pod_items['items']:
                p = subprocess.Popen(['kubectl', 'exec', '-n', cluster.namespace,
                                     item['metadata']['name'], '/srv/graceful_stop.sh'])
                procs.append(p)
            for p in procs:
                p.wait()

            [p.terminate() for p in log_processes]
        finally:
            shutil.rmtree(temp_spill_dir)
            if cluster:
                try:
                    cluster.stop(wait=True, timeout=20)
                except TimeoutError:
                    pass
            self._remove_docker_image(False)

    @mock.patch('kubernetes.client.CoreV1Api.create_namespaced_replication_controller',
                new=lambda *_, **__: None)
    def testCreateTimeout(self):
        api_client = k8s_config.new_client_from_config()

        cluster = None
        self._docker_image = 'pseudo_image'
        try:
            extra_vol_config = HostPathVolumeConfig('mars-src-path', '/mnt/mars', MARS_ROOT)
            with self.assertRaises(TimeoutError):
                cluster = new_cluster(api_client, image=self._docker_image,
                                      extra_volumes=[extra_vol_config], timeout=1)
        finally:
            if cluster:
                cluster.stop(wait=True)
