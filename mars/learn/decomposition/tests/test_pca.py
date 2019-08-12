# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import unittest

import numpy as np

import mars.tensor as mt
from mars.session import new_session

try:
    import sklearn
    from sklearn import datasets
    from sklearn.utils.testing import assert_array_almost_equal

    from ..pca import PCA
except ImportError:
    sklearn = None


@unittest.skipIf(sklearn is None, 'scikit-learn not installed')
class Test(unittest.TestCase):
    def setUp(self):
        self.iris = mt.tensor(datasets.load_iris().data)
        self.session = new_session().as_default()

    def testPCA(self):
        X = self.iris

        for n_comp in np.arange(X.shape[1]):
            pca = PCA(n_components=n_comp, svd_solver='full')

            X_r = pca.fit(X).transform(X).fetch()
            np.testing.assert_equal(X_r.shape[1], n_comp)

            X_r2 = pca.fit_transform(X)
            assert_array_almost_equal(X_r, X_r2)

            X_r = pca.transform(X)
            X_r2 = pca.fit_transform(X)
            assert_array_almost_equal(X_r, X_r2)

            # Test get_covariance and get_precision
            cov = pca.get_covariance()
            precision = pca.get_precision()
            assert_array_almost_equal(mt.dot(cov, precision).execute(),
                                      mt.eye(X.shape[1]).execute(), 12)

        # test explained_variance_ratio_ == 1 with all components
        pca = PCA(svd_solver='full')
        pca.fit(X)
        np.testing.assert_allclose(pca.explained_variance_ratio_.sum(), 1.0, 3)
