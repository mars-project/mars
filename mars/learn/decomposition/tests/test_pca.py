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

from itertools import product

import numpy as np
import pytest
import scipy as sp
from sklearn import datasets
from sklearn.utils._testing import (
    assert_array_almost_equal,
    assert_almost_equal,
    assert_raises_regex,
    assert_raise_message,
    assert_raises,
)

from .... import tensor as mt
from .._pca import PCA, _assess_dimension, _infer_dimension


iris = mt.tensor(datasets.load_iris().data)
# solver_list not includes arpack
solver_list = ["full", "randomized", "auto"]


def test_pca(setup):
    X = iris

    for n_comp in np.arange(X.shape[1]):
        pca = PCA(n_components=n_comp, svd_solver="full")

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
        assert_array_almost_equal(
            mt.dot(cov, precision).to_numpy(), np.eye(X.shape[1]), 12
        )

    # test explained_variance_ratio_ == 1 with all components
    pca = PCA(svd_solver="full")
    pca.fit(X)
    np.testing.assert_allclose(pca.explained_variance_ratio_.sum().to_numpy(), 1.0, 3)


def test_pca_randomized_solver(setup):
    # PCA on dense arrays
    X = iris

    # Loop excluding the 0, invalid for randomized
    for n_comp in np.arange(1, X.shape[1]):
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=0)

        X_r = pca.fit(X).transform(X)
        np.testing.assert_equal(X_r.shape[1], n_comp)

        X_r2 = pca.fit_transform(X)
        assert_array_almost_equal(X_r.fetch(), X_r2.fetch())

        X_r = pca.transform(X)
        assert_array_almost_equal(X_r.fetch(), X_r2.fetch())

        # Test get_covariance and get_precision
        cov = pca.get_covariance()
        precision = pca.get_precision()
        assert_array_almost_equal(
            mt.dot(cov, precision).to_numpy(), mt.eye(X.shape[1]).to_numpy(), 12
        )

    pca = PCA(n_components=0, svd_solver="randomized", random_state=0)
    with pytest.raises(ValueError):
        pca.fit(X)

    pca = PCA(n_components=0, svd_solver="randomized", random_state=0)
    with pytest.raises(ValueError):
        pca.fit(X)
    # Check internal state
    assert (
        pca.n_components
        == PCA(n_components=0, svd_solver="randomized", random_state=0).n_components
    )
    assert (
        pca.svd_solver
        == PCA(n_components=0, svd_solver="randomized", random_state=0).svd_solver
    )


def test_whitening(setup):
    # Check that PCA output has unit-variance
    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 80
    n_components = 30
    rank = 50

    # some low rank data with correlated features
    X = mt.dot(
        rng.randn(n_samples, rank),
        mt.dot(mt.diag(mt.linspace(10.0, 1.0, rank)), rng.randn(rank, n_features)),
    )
    # the component-wise variance of the first 50 features is 3 times the
    # mean component-wise variance of the remaining 30 features
    X[:, :50] *= 3

    assert X.shape == (n_samples, n_features)

    # the component-wise variance is thus highly varying:
    assert X.std(axis=0).std().to_numpy() > 43.8

    for solver, copy in product(solver_list, (True, False)):
        # whiten the data while projecting to the lower dim subspace
        X_ = X.copy()  # make sure we keep an original across iterations.
        pca = PCA(
            n_components=n_components,
            whiten=True,
            copy=copy,
            svd_solver=solver,
            random_state=0,
            iterated_power=7,
        )
        # test fit_transform
        X_whitened = pca.fit_transform(X_.copy())
        assert X_whitened.shape == (n_samples, n_components)
        X_whitened2 = pca.transform(X_)
        assert_array_almost_equal(X_whitened.fetch(), X_whitened2.fetch())

        assert_almost_equal(
            X_whitened.std(ddof=1, axis=0).to_numpy(), np.ones(n_components), decimal=6
        )
        assert_almost_equal(X_whitened.mean(axis=0).to_numpy(), np.zeros(n_components))

        X_ = X.copy()
        pca = PCA(
            n_components=n_components, whiten=False, copy=copy, svd_solver=solver
        ).fit(X_)
        X_unwhitened = pca.transform(X_)
        assert X_unwhitened.shape == (n_samples, n_components)

        # in that case the output components still have varying variances
        assert_almost_equal(X_unwhitened.std(axis=0).std().to_numpy(), 74.1, 1)
        # we always center, so no test for non-centering.


def test_explained_variance(setup):
    # Check that PCA output has unit-variance
    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 80

    X = mt.tensor(rng.randn(n_samples, n_features))

    pca = PCA(n_components=2, svd_solver="full").fit(X)
    rpca = PCA(n_components=2, svd_solver="randomized", random_state=42).fit(X)
    assert_array_almost_equal(
        pca.explained_variance_.to_numpy(), rpca.explained_variance_.to_numpy(), 1
    )
    assert_array_almost_equal(
        pca.explained_variance_ratio_.to_numpy(),
        rpca.explained_variance_ratio_.to_numpy(),
        1,
    )

    # compare to empirical variances
    expected_result = np.linalg.eig(np.cov(X.to_numpy(), rowvar=False))[0]
    expected_result = sorted(expected_result, reverse=True)[:2]

    X_pca = pca.transform(X)
    assert_array_almost_equal(
        pca.explained_variance_.to_numpy(), mt.var(X_pca, ddof=1, axis=0).to_numpy()
    )
    assert_array_almost_equal(pca.explained_variance_.to_numpy(), expected_result)

    X_rpca = rpca.transform(X)
    assert_array_almost_equal(
        rpca.explained_variance_.to_numpy(),
        mt.var(X_rpca, ddof=1, axis=0).to_numpy(),
        decimal=1,
    )
    assert_array_almost_equal(
        rpca.explained_variance_.to_numpy(), expected_result, decimal=1
    )

    # Same with correlated data
    X = datasets.make_classification(
        n_samples, n_features, n_informative=n_features - 2, random_state=rng
    )[0]
    X = mt.tensor(X)

    pca = PCA(n_components=2).fit(X)
    rpca = PCA(n_components=2, svd_solver="randomized", random_state=rng).fit(X)
    assert_array_almost_equal(
        pca.explained_variance_ratio_.to_numpy(),
        rpca.explained_variance_ratio_.to_numpy(),
        5,
    )


def test_singular_values(setup):
    # Check that the PCA output has the correct singular values

    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 80

    X = mt.tensor(rng.randn(n_samples, n_features))

    pca = PCA(n_components=2, svd_solver="full", random_state=rng).fit(X)
    rpca = PCA(n_components=2, svd_solver="randomized", random_state=rng).fit(X)
    assert_array_almost_equal(
        pca.singular_values_.fetch(), rpca.singular_values_.fetch(), 1
    )

    # Compare to the Frobenius norm
    X_pca = pca.transform(X)
    X_rpca = rpca.transform(X)
    assert_array_almost_equal(
        mt.sum(pca.singular_values_**2.0).to_numpy(),
        (mt.linalg.norm(X_pca, "fro") ** 2.0).to_numpy(),
        12,
    )
    assert_array_almost_equal(
        mt.sum(rpca.singular_values_**2.0).to_numpy(),
        (mt.linalg.norm(X_rpca, "fro") ** 2.0).to_numpy(),
        0,
    )

    # Compare to the 2-norms of the score vectors
    assert_array_almost_equal(
        pca.singular_values_.fetch(),
        mt.sqrt(mt.sum(X_pca**2.0, axis=0)).to_numpy(),
        12,
    )
    assert_array_almost_equal(
        rpca.singular_values_.fetch(),
        mt.sqrt(mt.sum(X_rpca**2.0, axis=0)).to_numpy(),
        2,
    )

    # Set the singular values and see what we get back
    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 110

    X = mt.tensor(rng.randn(n_samples, n_features))

    pca = PCA(n_components=3, svd_solver="full", random_state=rng)
    rpca = PCA(n_components=3, svd_solver="randomized", random_state=rng)
    X_pca = pca.fit_transform(X)

    X_pca /= mt.sqrt(mt.sum(X_pca**2.0, axis=0))
    X_pca[:, 0] *= 3.142
    X_pca[:, 1] *= 2.718

    X_hat = mt.dot(X_pca, pca.components_)
    pca.fit(X_hat)
    rpca.fit(X_hat)
    assert_array_almost_equal(pca.singular_values_.fetch(), [3.142, 2.718, 1.0], 14)
    assert_array_almost_equal(rpca.singular_values_.fetch(), [3.142, 2.718, 1.0], 14)


def test_pca_check_projection(setup):
    # Test that the projection of data is correct
    rng = np.random.RandomState(0)
    n, p = 100, 3
    X = mt.tensor(rng.randn(n, p) * 0.1)
    X[:10] += mt.array([3, 4, 5])
    Xt = 0.1 * mt.tensor(rng.randn(1, p)) + mt.array([3, 4, 5])

    for solver in solver_list:
        Yt = PCA(n_components=2, svd_solver=solver).fit(X).transform(Xt)
        Yt /= mt.sqrt((Yt**2).sum())

        assert_almost_equal(mt.abs(Yt[0][0]).to_numpy(), 1.0, 1)


def test_pca_inverse(setup):
    # Test that the projection of data can be inverted
    rng = np.random.RandomState(0)
    n, p = 50, 3
    X = mt.tensor(rng.randn(n, p))  # spherical data
    X[:, 1] *= 0.00001  # make middle component relatively small
    X += [5, 4, 3]  # make a large mean

    # same check that we can find the original data from the transformed
    # signal (since the data is almost of rank n_components)
    pca = PCA(n_components=2, svd_solver="full").fit(X)
    Y = pca.transform(X)
    Y_inverse = pca.inverse_transform(Y)
    assert_almost_equal(X.to_numpy(), Y_inverse.to_numpy(), decimal=3)

    # same as above with whitening (approximate reconstruction)
    for solver in solver_list:
        pca = PCA(n_components=2, whiten=True, svd_solver=solver)
        pca.fit(X)
        Y = pca.transform(X)
        Y_inverse = pca.inverse_transform(Y)
        assert_almost_equal(X.to_numpy(), Y_inverse.to_numpy(), decimal=3)


def test_pca_validation(setup):
    for solver in solver_list:
        # Ensures that solver-specific extreme inputs for the n_components
        # parameter raise errors
        X = mt.array([[0, 1, 0], [1, 0, 0]])
        smallest_d = 2  # The smallest dimension
        lower_limit = {"randomized": 1, "full": 0, "auto": 0}

        # We conduct the same test on X.T so that it is invariant to axis.
        for data in [X, X.T]:
            for n_components in [-1, 3]:

                if solver == "auto":
                    solver_reported = "full"
                else:
                    solver_reported = solver

                assert_raises_regex(
                    ValueError,
                    f"n_components={n_components}L? must be between "
                    rf"{lower_limit[solver]}L? and min\(n_samples, n_features\)="
                    f"{smallest_d}L? with svd_solver='{solver_reported}'",
                    PCA(n_components, svd_solver=solver).fit,
                    data,
                )

        n_components = 1.0
        type_ncom = type(n_components)
        assert_raise_message(
            ValueError,
            f"n_components={n_components} must be of type int "
            f"when greater than or equal to 1, was of type={type_ncom}",
            PCA(n_components, svd_solver=solver).fit,
            data,
        )


def test_n_components_none(setup):
    for solver in solver_list:
        # Ensures that n_components == None is handled correctly
        X = iris
        # We conduct the same test on X.T so that it is invariant to axis.
        for data in [X, X.T]:
            pca = PCA(svd_solver=solver)
            pca.fit(data)
            assert pca.n_components_ == min(data.shape)


def test_randomized_pca_check_projection(setup):
    # Test that the projection by randomized PCA on dense data is correct
    rng = np.random.RandomState(0)
    n, p = 100, 3
    X = mt.tensor(rng.randn(n, p) * 0.1)
    X[:10] += mt.array([3, 4, 5])
    Xt = 0.1 * mt.tensor(rng.randn(1, p)) + mt.array([3, 4, 5])

    Yt = (
        PCA(n_components=2, svd_solver="randomized", random_state=0)
        .fit(X)
        .transform(Xt)
    )
    Yt /= np.sqrt((Yt**2).sum())

    assert_almost_equal(mt.abs(Yt[0][0]).to_numpy(), 1.0, 1)


def test_randomized_pca_check_list(setup):
    # Test that the projection by randomized PCA on list data is correct
    X = mt.tensor([[1.0, 0.0], [0.0, 1.0]])
    X_transformed = (
        PCA(n_components=1, svd_solver="randomized", random_state=0).fit(X).transform(X)
    )
    assert X_transformed.shape == (2, 1)
    assert_almost_equal(X_transformed.mean().to_numpy(), 0.00, 2)
    assert_almost_equal(X_transformed.std().to_numpy(), 0.71, 2)


def test_randomized_pca_inverse(setup):
    # Test that randomized PCA is inversible on dense data
    rng = np.random.RandomState(0)
    n, p = 50, 3
    X = mt.tensor(rng.randn(n, p))  # spherical data
    X[:, 1] *= 0.00001  # make middle component relatively small
    X += [5, 4, 3]  # make a large mean

    # same check that we can find the original data from the transformed signal
    # (since the data is almost of rank n_components)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0).fit(X)
    Y = pca.transform(X)
    Y_inverse = pca.inverse_transform(Y)
    assert_almost_equal(X.to_numpy(), Y_inverse.to_numpy(), decimal=2)

    # same as above with whitening (approximate reconstruction)
    pca = PCA(n_components=2, whiten=True, svd_solver="randomized", random_state=0).fit(
        X
    )
    Y = pca.transform(X)
    Y_inverse = pca.inverse_transform(Y)
    relative_max_delta = (mt.abs(X - Y_inverse) / mt.abs(X).mean()).max()
    assert relative_max_delta.to_numpy() < 1e-5


def test_n_components_mle(setup):
    # Ensure that n_components == 'mle' doesn't raise error for auto/full
    # svd_solver and raises error for arpack/randomized svd_solver
    rng = np.random.RandomState(0)
    n_samples = 600
    n_features = 10
    X = mt.tensor(rng.randn(n_samples, n_features))
    n_components_dict = {}
    for solver in solver_list:
        pca = PCA(n_components="mle", svd_solver=solver)
        if solver in ["auto", "full"]:
            pca.fit(X)
            n_components_dict[solver] = pca.n_components_
        else:  # arpack/randomized solver
            error_message = (
                "n_components='mle' cannot be a string with " f"svd_solver='{solver}'"
            )
            assert_raise_message(ValueError, error_message, pca.fit, X)
    assert n_components_dict["auto"] == n_components_dict["full"]


def test_pca_dim(setup):
    # Check automated dimensionality setting
    rng = np.random.RandomState(0)
    n, p = 100, 5
    X = mt.tensor(rng.randn(n, p) * 0.1)
    X[:10] += mt.array([3, 4, 5, 1, 2])
    pca = PCA(n_components="mle", svd_solver="full").fit(X)
    assert pca.n_components == "mle"
    assert pca.n_components_ == 1


def test_infer_dim_1(setup):
    # TODO: explain what this is testing
    # Or at least use explicit variable names...
    n, p = 1000, 5
    rng = np.random.RandomState(0)
    X = (
        mt.tensor(rng.randn(n, p)) * 0.1
        + mt.tensor(rng.randn(n, 1)) * mt.array([3, 4, 5, 1, 2])
        + mt.array([1, 0, 7, 4, 6])
    )
    pca = PCA(n_components=p, svd_solver="full")
    pca.fit(X)
    spect = pca.explained_variance_.to_numpy()
    ll = np.array([_assess_dimension(spect, k, n) for k in range(1, p)])
    assert ll[1] > ll.max() - 0.01 * n


def test_infer_dim_2(setup):
    # TODO: explain what this is testing
    # Or at least use explicit variable names...
    n, p = 1000, 5
    rng = np.random.RandomState(0)
    X = mt.tensor(rng.randn(n, p) * 0.1)
    X[:10] += mt.array([3, 4, 5, 1, 2])
    X[10:20] += mt.array([6, 0, 7, 2, -1])
    pca = PCA(n_components=p, svd_solver="full")
    pca.fit(X)
    spect = pca.explained_variance_.fetch()
    assert _infer_dimension(spect, n) > 1


def test_infer_dim_3(setup):
    n, p = 100, 5
    rng = np.random.RandomState(0)
    X = mt.tensor(rng.randn(n, p) * 0.1)
    X[:10] += mt.array([3, 4, 5, 1, 2])
    X[10:20] += mt.array([6, 0, 7, 2, -1])
    X[30:40] += 2 * mt.array([-1, 1, -1, 1, -1])
    pca = PCA(n_components=p, svd_solver="full")
    pca.fit(X)
    spect = pca.explained_variance_.fetch()
    assert _infer_dimension(spect, n) > 2


def test_infer_dim_by_explained_variance(setup):
    X = iris
    pca = PCA(n_components=0.95, svd_solver="full")
    pca.fit(X)
    assert pca.n_components == 0.95
    assert pca.n_components_ == 2

    pca = PCA(n_components=0.01, svd_solver="full")
    pca.fit(X)
    assert pca.n_components == 0.01
    assert pca.n_components_ == 1

    rng = np.random.RandomState(0)
    # more features than samples
    X = mt.tensor(rng.rand(5, 20))
    pca = PCA(n_components=0.5, svd_solver="full").fit(X)
    assert pca.n_components == 0.5
    assert pca.n_components_ == 2


def test_pca_score(setup):
    # Test that probabilistic PCA scoring yields a reasonable score
    n, p = 1000, 3
    rng = np.random.RandomState(0)
    X = mt.tensor(rng.randn(n, p) * 0.1) + mt.array([3, 4, 5])
    for solver in solver_list:
        pca = PCA(n_components=2, svd_solver=solver)
        pca.fit(X)
        ll1 = pca.score(X)
        h = -0.5 * mt.log(2 * mt.pi * mt.exp(1) * 0.1**2) * p
        np.testing.assert_almost_equal((ll1 / h).to_numpy(), 1, 0)


def test_pca_score2(setup):
    # Test that probabilistic PCA correctly separated different datasets
    n, p = 100, 3
    rng = np.random.RandomState(0)
    X = mt.tensor(rng.randn(n, p) * 0.1) + mt.array([3, 4, 5])
    for solver in solver_list:
        pca = PCA(n_components=2, svd_solver=solver)
        pca.fit(X)
        ll1 = pca.score(X)
        ll2 = pca.score(mt.tensor(rng.randn(n, p) * 0.2) + mt.array([3, 4, 5]))
        assert ll1.fetch() > ll2.fetch()

        # Test that it gives different scores if whiten=True
        pca = PCA(n_components=2, whiten=True, svd_solver=solver)
        pca.fit(X)
        ll2 = pca.score(X)
        assert ll1.fetch() > ll2.fetch()


def test_pca_score3(setup):
    # Check that probabilistic PCA selects the right model
    n, p = 200, 3
    rng = np.random.RandomState(0)
    Xl = mt.tensor(
        rng.randn(n, p) + rng.randn(n, 1) * np.array([3, 4, 5]) + np.array([1, 0, 7])
    )
    Xt = mt.tensor(
        rng.randn(n, p) + rng.randn(n, 1) * np.array([3, 4, 5]) + np.array([1, 0, 7])
    )
    ll = mt.zeros(p)
    for k in range(p):
        pca = PCA(n_components=k, svd_solver="full")
        pca.fit(Xl)
        ll[k] = pca.score(Xt)

    assert ll.argmax().to_numpy() == 1


def test_pca_score_with_different_solvers(setup):
    digits = datasets.load_digits()
    X_digits = mt.tensor(digits.data)

    pca_dict = {
        svd_solver: PCA(n_components=30, svd_solver=svd_solver, random_state=0)
        for svd_solver in solver_list
    }

    for pca in pca_dict.values():
        pca.fit(X_digits)
        # Sanity check for the noise_variance_. For more details see
        # https://github.com/scikit-learn/scikit-learn/issues/7568
        # https://github.com/scikit-learn/scikit-learn/issues/8541
        # https://github.com/scikit-learn/scikit-learn/issues/8544
        assert mt.all((pca.explained_variance_ - pca.noise_variance_) >= 0).to_numpy()

    # Compare scores with different svd_solvers
    score_dict = {
        svd_solver: pca.score(X_digits).to_numpy()
        for svd_solver, pca in pca_dict.items()
    }
    assert_almost_equal(score_dict["full"], score_dict["randomized"], decimal=3)


def test_pca_zero_noise_variance_edge_cases(setup):
    # ensure that noise_variance_ is 0 in edge cases
    # when n_components == min(n_samples, n_features)
    n, p = 100, 3

    rng = np.random.RandomState(0)
    X = mt.tensor(rng.randn(n, p) * 0.1) + mt.array([3, 4, 5])
    # arpack raises ValueError for n_components == min(n_samples,
    # n_features)
    svd_solvers = ["full", "randomized"]

    for svd_solver in svd_solvers:
        pca = PCA(svd_solver=svd_solver, n_components=p)
        pca.fit(X)
        assert pca.noise_variance_ == 0

        pca.fit(X.T)
        assert pca.noise_variance_ == 0


def test_svd_solver_auto(setup):
    rng = np.random.RandomState(0)
    X = mt.tensor(rng.uniform(size=(1000, 50)))

    # case: n_components in (0,1) => 'full'
    pca = PCA(n_components=0.5)
    pca.fit(X)
    pca_test = PCA(n_components=0.5, svd_solver="full")
    pca_test.fit(X)
    assert_array_almost_equal(
        pca.components_.to_numpy(), pca_test.components_.to_numpy()
    )

    # case: max(X.shape) <= 500 => 'full'
    pca = PCA(n_components=5, random_state=0)
    Y = X[:10, :]
    pca.fit(Y)
    pca_test = PCA(n_components=5, svd_solver="full", random_state=0)
    pca_test.fit(Y)
    assert_array_almost_equal(
        pca.components_.to_numpy(), pca_test.components_.to_numpy()
    )

    # case: n_components >= .8 * min(X.shape) => 'full'
    pca = PCA(n_components=50)
    pca.fit(X)
    pca_test = PCA(n_components=50, svd_solver="full")
    pca_test.fit(X)
    assert_array_almost_equal(
        pca.components_.to_numpy(), pca_test.components_.to_numpy()
    )

    # n_components >= 1 and n_components < .8 * min(X.shape) => 'randomized'
    pca = PCA(n_components=10, random_state=0)
    pca.fit(X)
    pca_test = PCA(n_components=10, svd_solver="randomized", random_state=0)
    pca_test.fit(X)
    assert_array_almost_equal(
        pca.components_.to_numpy(), pca_test.components_.to_numpy()
    )


def test_pca_sparse_input(setup):
    for svd_solver in solver_list:
        X = np.random.RandomState(0).rand(5, 4)
        X = mt.tensor(sp.sparse.csr_matrix(X))
        assert X.issparse() is True

        pca = PCA(n_components=3, svd_solver=svd_solver)

        assert_raises(TypeError, pca.fit, X)


def test_pca_bad_solver(setup):
    X = mt.tensor(np.random.RandomState(0).rand(5, 4))
    pca = PCA(n_components=3, svd_solver="bad_argument")
    with pytest.raises(ValueError):
        pca.fit(X)


def test_pca_dtype_preservation(setup):
    for svd_solver in solver_list:
        _check_pca_float_dtype_preservation(svd_solver)
        _check_pca_int_dtype_upcast_to_double(svd_solver)


def _check_pca_float_dtype_preservation(svd_solver):
    # Ensure that PCA does not upscale the dtype when input is float32
    X_64 = mt.tensor(
        np.random.RandomState(0).rand(1000, 4).astype(np.float64, copy=False)
    )
    X_32 = X_64.astype(np.float32)

    pca_64 = PCA(n_components=3, svd_solver=svd_solver, random_state=0).fit(X_64)
    pca_32 = PCA(n_components=3, svd_solver=svd_solver, random_state=0).fit(X_32)

    assert pca_64.components_.dtype == np.float64
    assert pca_32.components_.dtype == np.float32
    assert pca_64.transform(X_64).dtype == np.float64
    assert pca_32.transform(X_32).dtype == np.float32

    # decimal=5 fails on mac with scipy = 1.1.0
    assert_array_almost_equal(
        pca_64.components_.to_numpy(), pca_32.components_.to_numpy(), decimal=4
    )


def _check_pca_int_dtype_upcast_to_double(svd_solver):
    # Ensure that all int types will be upcast to float64
    X_i64 = mt.tensor(np.random.RandomState(0).randint(0, 1000, (1000, 4)))
    X_i64 = X_i64.astype(np.int64, copy=False)
    X_i32 = X_i64.astype(np.int32, copy=False)

    pca_64 = PCA(n_components=3, svd_solver=svd_solver, random_state=0).fit(X_i64)
    pca_32 = PCA(n_components=3, svd_solver=svd_solver, random_state=0).fit(X_i32)

    assert pca_64.components_.dtype == np.float64
    assert pca_32.components_.dtype == np.float64
    assert pca_64.transform(X_i64).dtype == np.float64
    assert pca_32.transform(X_i32).dtype == np.float64

    assert_array_almost_equal(
        pca_64.components_.to_numpy(), pca_32.components_.to_numpy(), decimal=5
    )


def test_pca_deterministic_output(setup):
    rng = np.random.RandomState(0)
    X = mt.tensor(rng.rand(10, 10))

    for solver in solver_list:
        transformed_X = np.zeros((20, 2))
        for i in range(20):
            pca = PCA(n_components=2, svd_solver=solver, random_state=rng)
            transformed_X[i, :] = pca.fit_transform(X)[0].fetch()
        np.testing.assert_allclose(
            transformed_X, np.tile(transformed_X[0, :], 20).reshape(20, 2)
        )
