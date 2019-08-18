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
from itertools import product

import numpy as np

import mars.tensor as mt
from mars.session import new_session

try:
    import sklearn
    from sklearn import datasets
    from sklearn.utils.testing import assert_array_almost_equal, assert_almost_equal

    from ..pca import PCA
except ImportError:
    sklearn = None


@unittest.skipIf(sklearn is None, 'scikit-learn not installed')
class Test(unittest.TestCase):
    def setUp(self):
        self.iris = mt.tensor(datasets.load_iris().data)
        # solver_list not includes arpack
        self.solver_list = ['full', 'randomized', 'auto']
        self.session = new_session().as_default()

    def testPCA(self):
        X = self.iris

        for n_comp in np.arange(X.shape[1]):
            pca = PCA(n_components=n_comp, svd_solver='full')

            X_r = pca.fit(X).transform(X).fetch()
            np.testing.assert_equal(X_r.shape[1], n_comp)

            X_r2 = pca.fit_transform(X).fetch()
            assert_array_almost_equal(X_r, X_r2)

            X_r = pca.transform(X).fetch()
            X_r2 = pca.fit_transform(X).fetch()
            assert_array_almost_equal(X_r, X_r2)

            # Test get_covariance and get_precision
            cov = pca.get_covariance()
            precision = pca.get_precision()
            assert_array_almost_equal(mt.dot(cov, precision).execute(),
                                      np.eye(X.shape[1]), 12)

        # test explained_variance_ratio_ == 1 with all components
        pca = PCA(svd_solver='full')
        pca.fit(X)
        np.testing.assert_allclose(pca.explained_variance_ratio_.sum().execute(), 1.0, 3)

    def testPCARandomizedSolver(self):
        # PCA on dense arrays
        X = self.iris

        # Loop excluding the 0, invalid for randomized
        for n_comp in np.arange(1, X.shape[1]):
            pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=0)

            X_r = pca.fit(X).transform(X)
            np.testing.assert_equal(X_r.shape[1], n_comp)

            X_r2 = pca.fit_transform(X)
            assert_array_almost_equal(X_r.fetch(), X_r2.fetch())

            X_r = pca.transform(X)
            assert_array_almost_equal(X_r.fetch(), X_r2.fetch())

            # Test get_covariance and get_precision
            cov = pca.get_covariance()
            precision = pca.get_precision()
            assert_array_almost_equal(mt.dot(cov, precision).execute(),
                                      mt.eye(X.shape[1]).execute(), 12)

        pca = PCA(n_components=0, svd_solver='randomized', random_state=0)
        with self.assertRaises(ValueError):
            pca.fit(X)

        pca = PCA(n_components=0, svd_solver='randomized', random_state=0)
        with self.assertRaises(ValueError):
            pca.fit(X)
        # Check internal state
        self.assertEqual(pca.n_components,
                         PCA(n_components=0,
                             svd_solver='randomized', random_state=0).n_components)
        self.assertEqual(pca.svd_solver,
                         PCA(n_components=0,
                             svd_solver='randomized', random_state=0).svd_solver)

    def testWhitening(self):
        # Check that PCA output has unit-variance
        rng = np.random.RandomState(0)
        n_samples = 100
        n_features = 80
        n_components = 30
        rank = 50

        # some low rank data with correlated features
        X = mt.dot(rng.randn(n_samples, rank),
                   mt.dot(mt.diag(mt.linspace(10.0, 1.0, rank)),
                          rng.randn(rank, n_features)))
        # the component-wise variance of the first 50 features is 3 times the
        # mean component-wise variance of the remaining 30 features
        X[:, :50] *= 3

        self.assertEqual(X.shape, (n_samples, n_features))

        # the component-wise variance is thus highly varying:
        self.assertGreater(X.std(axis=0).std().execute(), 43.8)

        for solver, copy in product(self.solver_list, (True, False)):
            # whiten the data while projecting to the lower dim subspace
            X_ = X.copy()  # make sure we keep an original across iterations.
            pca = PCA(n_components=n_components, whiten=True, copy=copy,
                      svd_solver=solver, random_state=0, iterated_power=7)
            # test fit_transform
            X_whitened = pca.fit_transform(X_.copy())
            self.assertEqual(X_whitened.shape, (n_samples, n_components))
            X_whitened2 = pca.transform(X_)
            assert_array_almost_equal(X_whitened.fetch(), X_whitened2.fetch())

            assert_almost_equal(X_whitened.std(ddof=1, axis=0).execute(),
                                np.ones(n_components),
                                decimal=6)
            assert_almost_equal(X_whitened.mean(axis=0).execute(), np.zeros(n_components))

            X_ = X.copy()
            pca = PCA(n_components=n_components, whiten=False, copy=copy,
                      svd_solver=solver).fit(X_)
            X_unwhitened = pca.transform(X_)
            self.assertEqual(X_unwhitened.shape, (n_samples, n_components))

            # in that case the output components still have varying variances
            assert_almost_equal(X_unwhitened.std(axis=0).std().execute(), 74.1, 1)
            # we always center, so no test for non-centering.

    def testExplainedVariance(self):
        # Check that PCA output has unit-variance
        rng = np.random.RandomState(0)
        n_samples = 100
        n_features = 80

        X = mt.tensor(rng.randn(n_samples, n_features))

        pca = PCA(n_components=2, svd_solver='full').fit(X)
        rpca = PCA(n_components=2, svd_solver='randomized', random_state=42).fit(X)
        assert_array_almost_equal(pca.explained_variance_.execute(),
                                  rpca.explained_variance_.execute(), 1)
        assert_array_almost_equal(pca.explained_variance_ratio_.execute(),
                                  rpca.explained_variance_ratio_.execute(), 1)

        # compare to empirical variances
        expected_result = np.linalg.eig(np.cov(X.execute(), rowvar=False))[0]
        expected_result = sorted(expected_result, reverse=True)[:2]

        X_pca = pca.transform(X)
        assert_array_almost_equal(pca.explained_variance_.execute(),
                                  mt.var(X_pca, ddof=1, axis=0).execute())
        assert_array_almost_equal(pca.explained_variance_.execute(), expected_result)

        X_rpca = rpca.transform(X)
        assert_array_almost_equal(rpca.explained_variance_.execute(),
                                  mt.var(X_rpca, ddof=1, axis=0).execute(),
                                  decimal=1)
        assert_array_almost_equal(rpca.explained_variance_.execute(),
                                  expected_result, decimal=1)

        # Same with correlated data
        X = datasets.make_classification(n_samples, n_features,
                                         n_informative=n_features-2,
                                         random_state=rng)[0]
        X = mt.tensor(X)

        pca = PCA(n_components=2).fit(X)
        rpca = PCA(n_components=2, svd_solver='randomized',
                   random_state=rng).fit(X)
        assert_array_almost_equal(pca.explained_variance_ratio_.execute(),
                                  rpca.explained_variance_ratio_.execute(), 5)

    def test_singular_values(self):
        # Check that the PCA output has the correct singular values

        rng = np.random.RandomState(0)
        n_samples = 100
        n_features = 80

        X = mt.tensor(rng.randn(n_samples, n_features))

        pca = PCA(n_components=2, svd_solver='full',
                  random_state=rng).fit(X)
        rpca = PCA(n_components=2, svd_solver='randomized',
                   random_state=rng).fit(X)
        assert_array_almost_equal(pca.singular_values_.fetch(), rpca.singular_values_.fetch(), 1)

        # Compare to the Frobenius norm
        X_pca = pca.transform(X)
        X_rpca = rpca.transform(X)
        assert_array_almost_equal(mt.sum(pca.singular_values_**2.0).execute(),
                                  (mt.linalg.norm(X_pca, "fro")**2.0).execute(), 12)
        assert_array_almost_equal(mt.sum(rpca.singular_values_**2.0).execute(),
                                  (mt.linalg.norm(X_rpca, "fro")**2.0).execute(), 0)

        # Compare to the 2-norms of the score vectors
        assert_array_almost_equal(pca.singular_values_.fetch(),
                                  mt.sqrt(mt.sum(X_pca**2.0, axis=0)).execute(), 12)
        assert_array_almost_equal(rpca.singular_values_.fetch(),
                                  mt.sqrt(mt.sum(X_rpca**2.0, axis=0)).execute(), 2)

        # Set the singular values and see what we get back
        rng = np.random.RandomState(0)
        n_samples = 100
        n_features = 110

        X = mt.tensor(rng.randn(n_samples, n_features))

        pca = PCA(n_components=3, svd_solver='full', random_state=rng)
        rpca = PCA(n_components=3, svd_solver='randomized', random_state=rng)
        X_pca = pca.fit_transform(X)

        X_pca /= np.sqrt(np.sum(X_pca**2.0, axis=0))
        X_pca[:, 0] *= 3.142
        X_pca[:, 1] *= 2.718

        X_hat = mt.dot(X_pca, pca.components_)
        pca.fit(X_hat)
        rpca.fit(X_hat)
        assert_array_almost_equal(pca.singular_values_.fetch(), [3.142, 2.718, 1.0], 14)
        assert_array_almost_equal(rpca.singular_values_.fetch(), [3.142, 2.718, 1.0], 14)
