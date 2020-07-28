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

import os
import sys
import unittest

import numpy as np
try:
    from sklearn.cluster import KMeans as SK_KMEANS
    from sklearn.datasets import make_blobs
except ImportError:
    pass

from mars import tensor as mt
from mars.learn.cluster import KMeans
from mars.learn.tests.integrated.base import LearnIntegrationTestBase
from mars.session import new_session


@unittest.skipIf(KMeans is None, 'scikit-learn not installed')
@unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
class Test(LearnIntegrationTestBase):
    def testDistributedKMeans(self):
        service_ep = 'http://127.0.0.1:' + self.web_port
        timeout = 120 if 'CI' in os.environ else -1

        with new_session(service_ep) as sess:
            run_kwargs = {'timeout': timeout}

            rnd = np.random.RandomState(0)
            X, _ = make_blobs(random_state=rnd)
            raw = X
            X = mt.tensor(X, chunk_size=50)

            km_elkan = KMeans(algorithm='elkan', n_clusters=5,
                              random_state=0, n_init=1, tol=1e-4,
                              init='k-means++')
            sk_km_elkan = SK_KMEANS(algorithm='elkan', n_clusters=5,
                                    random_state=0, n_init=1, tol=1e-4,
                                    init='k-means++')

            km_elkan.fit(X, session=sess, run_kwargs=run_kwargs)
            sk_km_elkan.fit(raw)

            np.testing.assert_allclose(km_elkan.cluster_centers_, sk_km_elkan.cluster_centers_)
            np.testing.assert_array_equal(km_elkan.labels_, sk_km_elkan.labels_)

            self.assertEqual(km_elkan.n_iter_, sk_km_elkan.n_iter_)

        with new_session(service_ep) as sess2:
            run_kwargs = {'timeout': timeout}

            rnd = np.random.RandomState(0)
            X, _ = make_blobs(random_state=rnd)
            X = mt.tensor(X, chunk_size=50)

            kmeans = KMeans(n_clusters=5, random_state=0, n_init=1,
                            tol=1e-4, init='k-means||')
            kmeans.fit(X, session=sess2, run_kwargs=run_kwargs)
