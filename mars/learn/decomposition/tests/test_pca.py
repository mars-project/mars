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
    import scipy as sp
    import sklearn
    from sklearn import datasets
    from sklearn.utils.testing import assert_array_almost_equal, \
        assert_almost_equal, assert_raises_regex, assert_raise_message, assert_raises

    from mars.learn.decomposition.pca import PCA, _assess_dimension_, _infer_dimension_
except ImportError:
    sklearn = None

try:
    from jax.config import config

    JAX_INSTALLED = True
except ImportError:
    JAX_INSTALLED = False


@unittest.skipIf(sklearn is None or not JAX_INSTALLED, 'scikit-learn or jax not installed')
class Test(unittest.TestCase):
    def setUp(self):
        self.iris = mt.tensor(datasets.load_iris().data)
        # solver_list not includes arpack
        self.solver_list = ['full', 'randomized', 'auto']
        self.session = new_session().as_default()
        # Note: for jax offers 32bit by default,
        # but we need 64bit to pass these test case.
        config.update("jax_enable_x64", True)

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

        X_pca /= mt.sqrt(mt.sum(X_pca**2.0, axis=0))
        X_pca[:, 0] *= 3.142
        X_pca[:, 1] *= 2.718

        X_hat = mt.dot(X_pca, pca.components_)
        pca.fit(X_hat)
        rpca.fit(X_hat)
        assert_array_almost_equal(pca.singular_values_.fetch(), [3.142, 2.718, 1.0], 14)
        assert_array_almost_equal(rpca.singular_values_.fetch(), [3.142, 2.718, 1.0], 14)

    def test_pca_check_projection(self):
        # Test that the projection of data is correct
        rng = np.random.RandomState(0)
        n, p = 100, 3
        X = mt.tensor(rng.randn(n, p) * .1)
        X[:10] += mt.array([3, 4, 5])
        Xt = 0.1 * mt.tensor(rng.randn(1, p)) + mt.array([3, 4, 5])

        for solver in self.solver_list:
            Yt = PCA(n_components=2, svd_solver=solver).fit(X).transform(Xt)
            Yt /= mt.sqrt((Yt ** 2).sum())

            assert_almost_equal(mt.abs(Yt[0][0]).execute(), 1., 1)

    def test_pca_inverse(self):
        # Test that the projection of data can be inverted
        rng = np.random.RandomState(0)
        n, p = 50, 3
        X = mt.tensor(rng.randn(n, p))  # spherical data
        X[:, 1] *= .00001  # make middle component relatively small
        X += [5, 4, 3]  # make a large mean

        # same check that we can find the original data from the transformed
        # signal (since the data is almost of rank n_components)
        pca = PCA(n_components=2, svd_solver='full').fit(X)
        Y = pca.transform(X)
        Y_inverse = pca.inverse_transform(Y)
        assert_almost_equal(X.execute(), Y_inverse.execute(), decimal=3)

        # same as above with whitening (approximate reconstruction)
        for solver in self.solver_list:
            pca = PCA(n_components=2, whiten=True, svd_solver=solver)
            pca.fit(X)
            Y = pca.transform(X)
            Y_inverse = pca.inverse_transform(Y)
            assert_almost_equal(X.execute(), Y_inverse.execute(), decimal=3)

    def test_pca_validation(self):
        for solver in self.solver_list:
            # Ensures that solver-specific extreme inputs for the n_components
            # parameter raise errors
            X = mt.array([[0, 1, 0], [1, 0, 0]])
            smallest_d = 2  # The smallest dimension
            lower_limit = {'randomized': 1, 'full': 0, 'auto': 0}

            # We conduct the same test on X.T so that it is invariant to axis.
            for data in [X, X.T]:
                for n_components in [-1, 3]:

                    if solver == 'auto':
                        solver_reported = 'full'
                    else:
                        solver_reported = solver

                    assert_raises_regex(ValueError,
                                        "n_components={}L? must be between "
                                        r"{}L? and min\(n_samples, n_features\)="
                                        "{}L? with svd_solver=\'{}\'"
                                        .format(n_components,
                                                lower_limit[solver],
                                                smallest_d,
                                                solver_reported),
                                        PCA(n_components,
                                            svd_solver=solver).fit, data)

            n_components = 1.0
            type_ncom = type(n_components)
            assert_raise_message(ValueError,
                                 "n_components={} must be of type int "
                                 "when greater than or equal to 1, was of type={}"
                                 .format(n_components, type_ncom),
                                 PCA(n_components, svd_solver=solver).fit, data)

    def test_n_components_none(self):
        for solver in self.solver_list:
            # Ensures that n_components == None is handled correctly
            X = self.iris
            # We conduct the same test on X.T so that it is invariant to axis.
            for data in [X, X.T]:
                pca = PCA(svd_solver=solver)
                pca.fit(data)
                self.assertEqual(pca.n_components_, min(data.shape))

    def test_randomized_pca_check_projection(self):
        # Test that the projection by randomized PCA on dense data is correct
        rng = np.random.RandomState(0)
        n, p = 100, 3
        X = mt.tensor(rng.randn(n, p) * .1)
        X[:10] += mt.array([3, 4, 5])
        Xt = 0.1 * mt.tensor(rng.randn(1, p)) + mt.array([3, 4, 5])

        Yt = PCA(n_components=2, svd_solver='randomized',
                 random_state=0).fit(X).transform(Xt)
        Yt /= np.sqrt((Yt ** 2).sum())

        assert_almost_equal(mt.abs(Yt[0][0]).execute(), 1., 1)

    def test_randomized_pca_check_list(self):
        # Test that the projection by randomized PCA on list data is correct
        X = mt.tensor([[1.0, 0.0], [0.0, 1.0]])
        X_transformed = PCA(n_components=1, svd_solver='randomized',
                            random_state=0).fit(X).transform(X)
        self.assertEqual(X_transformed.shape, (2, 1))
        assert_almost_equal(X_transformed.mean().execute(), 0.00, 2)
        assert_almost_equal(X_transformed.std().execute(), 0.71, 2)

    def test_randomized_pca_inverse(self):
        # Test that randomized PCA is inversible on dense data
        rng = np.random.RandomState(0)
        n, p = 50, 3
        X = mt.tensor(rng.randn(n, p))  # spherical data
        X[:, 1] *= .00001  # make middle component relatively small
        X += [5, 4, 3]  # make a large mean

        # same check that we can find the original data from the transformed signal
        # (since the data is almost of rank n_components)
        pca = PCA(n_components=2, svd_solver='randomized', random_state=0).fit(X)
        Y = pca.transform(X)
        Y_inverse = pca.inverse_transform(Y)
        assert_almost_equal(X.execute(), Y_inverse.execute(), decimal=2)

        # same as above with whitening (approximate reconstruction)
        pca = PCA(n_components=2, whiten=True, svd_solver='randomized',
                  random_state=0).fit(X)
        Y = pca.transform(X)
        Y_inverse = pca.inverse_transform(Y)
        relative_max_delta = (mt.abs(X - Y_inverse) / mt.abs(X).mean()).max()
        self.assertLess(relative_max_delta.execute(), 1e-5)

    def test_n_components_mle(self):
        # Ensure that n_components == 'mle' doesn't raise error for auto/full
        # svd_solver and raises error for arpack/randomized svd_solver
        rng = np.random.RandomState(0)
        n_samples = 600
        n_features = 10
        X = mt.tensor(rng.randn(n_samples, n_features))
        n_components_dict = {}
        for solver in self.solver_list:
            pca = PCA(n_components='mle', svd_solver=solver)
            if solver in ['auto', 'full']:
                pca.fit(X)
                n_components_dict[solver] = pca.n_components_
            else:  # arpack/randomized solver
                error_message = ("n_components='mle' cannot be a string with "
                                 "svd_solver='{}'".format(solver))
                assert_raise_message(ValueError, error_message, pca.fit, X)
        self.assertEqual(n_components_dict['auto'], n_components_dict['full'])

    def test_pca_dim(self):
        # Check automated dimensionality setting
        rng = np.random.RandomState(0)
        n, p = 100, 5
        X = mt.tensor(rng.randn(n, p) * .1)
        X[:10] += mt.array([3, 4, 5, 1, 2])
        pca = PCA(n_components='mle', svd_solver='full').fit(X)
        self.assertEqual(pca.n_components, 'mle')
        self.assertEqual(pca.n_components_, 1)

    def test_infer_dim_1(self):
        # TODO: explain what this is testing
        # Or at least use explicit variable names...
        n, p = 1000, 5
        rng = np.random.RandomState(0)
        X = (mt.tensor(rng.randn(n, p)) * .1 + mt.tensor(rng.randn(n, 1)) * mt.array([3, 4, 5, 1, 2]) +
             mt.array([1, 0, 7, 4, 6]))
        pca = PCA(n_components=p, svd_solver='full')
        pca.fit(X)
        spect = pca.explained_variance_
        ll = mt.array([_assess_dimension_(spect, k, n, p) for k in range(p)]).execute()
        self.assertGreater(ll[1], ll.max() - .01 * n)

    def test_infer_dim_2(self):
        # TODO: explain what this is testing
        # Or at least use explicit variable names...
        n, p = 1000, 5
        rng = np.random.RandomState(0)
        X = mt.tensor(rng.randn(n, p) * .1)
        X[:10] += mt.array([3, 4, 5, 1, 2])
        X[10:20] += mt.array([6, 0, 7, 2, -1])
        pca = PCA(n_components=p, svd_solver='full')
        pca.fit(X)
        spect = pca.explained_variance_
        self.assertGreater(_infer_dimension_(spect, n, p).execute(), 1)

    def test_infer_dim_3(self):
        n, p = 100, 5
        rng = np.random.RandomState(0)
        X = mt.tensor(rng.randn(n, p) * .1)
        X[:10] += mt.array([3, 4, 5, 1, 2])
        X[10:20] += mt.array([6, 0, 7, 2, -1])
        X[30:40] += 2 * mt.array([-1, 1, -1, 1, -1])
        pca = PCA(n_components=p, svd_solver='full')
        pca.fit(X)
        spect = pca.explained_variance_
        self.assertGreater(_infer_dimension_(spect, n, p).execute(), 2)

    def test_infer_dim_by_explained_variance(self):
        X = self.iris
        pca = PCA(n_components=0.95, svd_solver='full')
        pca.fit(X)
        self.assertEqual(pca.n_components, 0.95)
        self.assertEqual(pca.n_components_, 2)

        pca = PCA(n_components=0.01, svd_solver='full')
        pca.fit(X)
        self.assertEqual(pca.n_components, 0.01)
        self.assertEqual(pca.n_components_, 1)

        rng = np.random.RandomState(0)
        # more features than samples
        X = mt.tensor(rng.rand(5, 20))
        pca = PCA(n_components=.5, svd_solver='full').fit(X)
        self.assertEqual(pca.n_components, 0.5)
        self.assertEqual(pca.n_components_, 2)

    def test_pca_score(self):
        # Test that probabilistic PCA scoring yields a reasonable score
        n, p = 1000, 3
        rng = np.random.RandomState(0)
        X = mt.tensor(rng.randn(n, p) * .1) + mt.array([3, 4, 5])
        for solver in self.solver_list:
            pca = PCA(n_components=2, svd_solver=solver)
            pca.fit(X)
            ll1 = pca.score(X)
            h = -0.5 * mt.log(2 * mt.pi * mt.exp(1) * 0.1 ** 2) * p
            np.testing.assert_almost_equal((ll1 / h).execute(), 1, 0)

    def test_pca_score2(self):
        # Test that probabilistic PCA correctly separated different datasets
        n, p = 100, 3
        rng = np.random.RandomState(0)
        X = mt.tensor(rng.randn(n, p) * .1) + mt.array([3, 4, 5])
        for solver in self.solver_list:
            pca = PCA(n_components=2, svd_solver=solver)
            pca.fit(X)
            ll1 = pca.score(X)
            ll2 = pca.score(mt.tensor(rng.randn(n, p) * .2) + mt.array([3, 4, 5]))
            self.assertGreater(ll1.fetch(), ll2.fetch())

            # Test that it gives different scores if whiten=True
            pca = PCA(n_components=2, whiten=True, svd_solver=solver)
            pca.fit(X)
            ll2 = pca.score(X)
            assert ll1.fetch() > ll2.fetch()

    def test_pca_score3(self):
        # Check that probabilistic PCA selects the right model
        n, p = 200, 3
        rng = np.random.RandomState(0)
        Xl = mt.tensor(rng.randn(n, p) + rng.randn(n, 1) * np.array([3, 4, 5]) +
                       np.array([1, 0, 7]))
        Xt = mt.tensor(rng.randn(n, p) + rng.randn(n, 1) * np.array([3, 4, 5]) +
                       np.array([1, 0, 7]))
        ll = mt.zeros(p)
        for k in range(p):
            pca = PCA(n_components=k, svd_solver='full')
            pca.fit(Xl)
            ll[k] = pca.score(Xt)

        assert ll.argmax().execute() == 1

    def test_pca_score_with_different_solvers(self):
        digits = datasets.load_digits()
        X_digits = mt.tensor(digits.data)

        pca_dict = {svd_solver: PCA(n_components=30, svd_solver=svd_solver,
                                    random_state=0)
                    for svd_solver in self.solver_list}

        for pca in pca_dict.values():
            pca.fit(X_digits)
            # Sanity check for the noise_variance_. For more details see
            # https://github.com/scikit-learn/scikit-learn/issues/7568
            # https://github.com/scikit-learn/scikit-learn/issues/8541
            # https://github.com/scikit-learn/scikit-learn/issues/8544
            assert mt.all((pca.explained_variance_ - pca.noise_variance_) >= 0).execute()

        # Compare scores with different svd_solvers
        score_dict = {svd_solver: pca.score(X_digits).execute()
                      for svd_solver, pca in pca_dict.items()}
        assert_almost_equal(score_dict['full'], score_dict['randomized'],
                            decimal=3)

    def test_pca_zero_noise_variance_edge_cases(self):
        # ensure that noise_variance_ is 0 in edge cases
        # when n_components == min(n_samples, n_features)
        n, p = 100, 3

        rng = np.random.RandomState(0)
        X = mt.tensor(rng.randn(n, p) * .1) + mt.array([3, 4, 5])
        # arpack raises ValueError for n_components == min(n_samples,
        # n_features)
        svd_solvers = ['full', 'randomized']

        for svd_solver in svd_solvers:
            pca = PCA(svd_solver=svd_solver, n_components=p)
            pca.fit(X)
            self.assertEqual(pca.noise_variance_, 0)

            pca.fit(X.T)
            self.assertEqual(pca.noise_variance_, 0)

    def test_svd_solver_auto(self):
        rng = np.random.RandomState(0)
        X = mt.tensor(rng.uniform(size=(1000, 50)))

        # case: n_components in (0,1) => 'full'
        pca = PCA(n_components=.5)
        pca.fit(X)
        pca_test = PCA(n_components=.5, svd_solver='full')
        pca_test.fit(X)
        assert_array_almost_equal(pca.components_.execute(), pca_test.components_.execute())

        # case: max(X.shape) <= 500 => 'full'
        pca = PCA(n_components=5, random_state=0)
        Y = X[:10, :]
        pca.fit(Y)
        pca_test = PCA(n_components=5, svd_solver='full', random_state=0)
        pca_test.fit(Y)
        assert_array_almost_equal(pca.components_.execute(), pca_test.components_.execute())

        # case: n_components >= .8 * min(X.shape) => 'full'
        pca = PCA(n_components=50)
        pca.fit(X)
        pca_test = PCA(n_components=50, svd_solver='full')
        pca_test.fit(X)
        assert_array_almost_equal(pca.components_.execute(), pca_test.components_.execute())

        # n_components >= 1 and n_components < .8 * min(X.shape) => 'randomized'
        pca = PCA(n_components=10, random_state=0)
        pca.fit(X)
        pca_test = PCA(n_components=10, svd_solver='randomized', random_state=0)
        pca_test.fit(X)
        assert_array_almost_equal(pca.components_.execute(), pca_test.components_.execute())

    def test_pca_sparse_input(self):
        for svd_solver in self.solver_list:
            X = np.random.RandomState(0).rand(5, 4)
            X = mt.tensor(sp.sparse.csr_matrix(X))
            self.assertTrue(X.issparse())

            pca = PCA(n_components=3, svd_solver=svd_solver)

            assert_raises(TypeError, pca.fit, X)

    def test_pca_bad_solver(self):
        X = mt.tensor(np.random.RandomState(0).rand(5, 4))
        pca = PCA(n_components=3, svd_solver='bad_argument')
        with self.assertRaises(ValueError):
            pca.fit(X)

    def test_pca_dtype_preservation(self):
        for svd_solver in self.solver_list:
            self._check_pca_float_dtype_preservation(svd_solver)
            self._check_pca_int_dtype_upcast_to_double(svd_solver)

    def _check_pca_float_dtype_preservation(self, svd_solver):
        # Ensure that PCA does not upscale the dtype when input is float32
        X_64 = mt.tensor(np.random.RandomState(0).rand(1000, 4).astype(np.float64,
                                                                       copy=False))
        X_32 = X_64.astype(np.float32)

        pca_64 = PCA(n_components=3, svd_solver=svd_solver,
                     random_state=0).fit(X_64)
        pca_32 = PCA(n_components=3, svd_solver=svd_solver,
                     random_state=0).fit(X_32)

        self.assertEqual(pca_64.components_.dtype, np.float64)
        self.assertEqual(pca_32.components_.dtype, np.float32)
        self.assertEqual(pca_64.transform(X_64).dtype, np.float64)
        self.assertEqual(pca_32.transform(X_32).dtype, np.float32)

        # decimal=5 fails on mac with scipy = 1.1.0
        assert_array_almost_equal(pca_64.components_.execute(), pca_32.components_.execute(),
                                  decimal=4)

    def _check_pca_int_dtype_upcast_to_double(self, svd_solver):
        # Ensure that all int types will be upcast to float64
        X_i64 = mt.tensor(np.random.RandomState(0).randint(0, 1000, (1000, 4)))
        X_i64 = X_i64.astype(np.int64, copy=False)
        X_i32 = X_i64.astype(np.int32, copy=False)

        pca_64 = PCA(n_components=3, svd_solver=svd_solver,
                     random_state=0).fit(X_i64)
        pca_32 = PCA(n_components=3, svd_solver=svd_solver,
                     random_state=0).fit(X_i32)

        self.assertEqual(pca_64.components_.dtype, np.float64)
        self.assertEqual(pca_32.components_.dtype, np.float64)
        self.assertEqual(pca_64.transform(X_i64).dtype, np.float64)
        self.assertEqual(pca_32.transform(X_i32).dtype, np.float64)

        assert_array_almost_equal(pca_64.components_.execute(), pca_32.components_.execute(),
                                  decimal=5)

    def test_pca_deterministic_output(self):
        rng = np.random.RandomState(0)
        X = mt.tensor(rng.rand(10, 10))

        for solver in self.solver_list:
            transformed_X = mt.zeros((20, 2))
            for i in range(20):
                pca = PCA(n_components=2, svd_solver=solver, random_state=rng)
                transformed_X[i, :] = pca.fit_transform(X)[0]
            np.testing.assert_allclose(
                transformed_X.execute(), mt.tile(transformed_X[0, :], 20).reshape(20, 2).execute())
