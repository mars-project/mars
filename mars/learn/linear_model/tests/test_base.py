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

import re

import numpy as np
import pytest
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_almost_equal,
    assert_allclose,
)
from scipy import sparse
from scipy import linalg
from sklearn.datasets import make_sparse_uncorrelated, make_regression, load_iris
from sklearn.linear_model import LinearRegression as sklearn_LR
from sklearn.linear_model._base import make_dataset
from sklearn.utils import check_random_state

from .. import LinearRegression
from .._base import _preprocess_data, _rescale_data


rng = np.random.RandomState(0)
rtol = 1e-6


def test_linear_regression(setup):
    # Regular model fitting, #samples > 2, #features >= 2
    X = [[1, 1.5], [1.8, 2], [4, 5]]
    Y = [1, 2, 3]

    reg = LinearRegression()
    reg.fit(X, Y)

    model = sklearn_LR()
    model.fit(X, Y)

    assert_array_almost_equal(reg.coef_, model.coef_)
    assert_array_almost_equal(reg.intercept_, model.intercept_)
    assert_array_almost_equal(reg.predict(X), model.predict(X))

    # Regular model fitting, #samples <= 2, # features < 2
    error_msg = re.escape("Does not support sigular matrix!")

    X = [[1], [2]]
    Y = [1, 2]

    reg = LinearRegression()
    reg.fit(X, Y)

    model = sklearn_LR()
    model.fit(X, Y)

    assert_array_almost_equal(reg.coef_, model.coef_)
    assert_array_almost_equal(reg.intercept_, model.intercept_)
    assert_array_almost_equal(reg.predict(X), model.predict(X))

    # Extra case #1: singular matrix, degenerate input
    error_msg = re.escape("Does not support sigular matrix!")

    X = [[1]]
    Y = [0]

    reg = LinearRegression()
    with pytest.raises(NotImplementedError, match=error_msg):
        reg.fit(X, Y)

    # # Extra case #2: algebrically singular matrix but algorithmically not
    # # Works locally but not work in github checks
    # # May be because the inverse is super large
    # X = [[1, 1.5], [1.8, 2]]
    # Y = [1, 2]

    # reg = LinearRegression()
    # reg.fit(X, Y)

    # model = sklearn_LR()
    # model.fit(X, Y)

    # with pytest.raises(AssertionError):
    #     assert_array_almost_equal(reg.coef_, model.coef_)


def test_linear_regression_sample_weights(setup):
    # TODO: loop over sparse data as well

    rng = np.random.RandomState(0)

    # It would not work with under-determined systems
    for n_samples, n_features in ((6, 5),):
        y = rng.randn(n_samples)
        X = rng.randn(n_samples, n_features)
        sample_weight = 1.0 + rng.rand(n_samples)

        for intercept in (True, False):
            # LinearRegression with explicit sample_weight
            reg = LinearRegression(fit_intercept=intercept)
            reg.fit(X, y, sample_weight=sample_weight)
            coefs1 = reg.coef_
            inter1 = reg.intercept_

            assert reg.coef_.shape == (X.shape[1],)  # sanity checks
            assert reg.score(X, y).to_numpy() > 0.5

            # Closed form of the weighted least square
            # theta = (X^T W X)^(-1) * X^T W y
            W = np.diag(sample_weight)
            if intercept is False:
                X_aug = X
            else:
                dummy_column = np.ones(shape=(n_samples, 1))
                X_aug = np.concatenate((dummy_column, X), axis=1)

            coefs2 = linalg.solve(X_aug.T.dot(W).dot(X_aug), X_aug.T.dot(W).dot(y))

            if intercept is False:
                assert_array_almost_equal(coefs1, coefs2)
            else:
                assert_array_almost_equal(coefs1, coefs2[1:])
                assert_almost_equal(inter1.to_numpy(), coefs2[0])


def test_raises_value_error_if_positive_and_sparse(setup):
    error_msg = re.escape(
        "A sparse tensor was passed, but dense "
        "data is required. Use X.todense() to "
        "convert to a dense tensor."
    )
    # X must not be sparse if positive == True
    X = sparse.eye(10)
    y = np.ones(10)

    reg = LinearRegression(positive=True)

    with pytest.raises(TypeError, match=error_msg):
        reg.fit(X, y)


def test_raises_value_error_if_sample_weights_greater_than_1d(setup):
    error_msg = re.escape("Sample weights must be 1D array or scalar")

    X = rng.randn(10, 5)
    y = rng.randn(10)
    sample_weights_2D = rng.randn(10, 2) ** 2 + 1

    reg = LinearRegression()

    with pytest.raises(ValueError, match=error_msg):
        reg.fit(X, y, sample_weights_2D)


def test_fit_intercept(setup):
    # Test assertions on betas shape.
    X2 = np.array([[0.38349978, 0.61650022], [0.58853682, 0.41146318]])
    X3 = np.array(
        [[0.27677969, 0.70693172, 0.01628859], [0.08385139, 0.20692515, 0.70922346]]
    )
    y = np.array([1, 1])

    lr2_without_intercept = LinearRegression(fit_intercept=False).fit(X2, y)
    lr2_with_intercept = LinearRegression().fit(X2, y)

    lr3_without_intercept = LinearRegression(fit_intercept=False).fit(X3, y)
    lr3_with_intercept = LinearRegression().fit(X3, y)

    assert lr2_with_intercept.coef_.shape == lr2_without_intercept.coef_.shape
    assert lr3_with_intercept.coef_.shape == lr3_without_intercept.coef_.shape
    assert lr2_without_intercept.coef_.ndim == lr3_without_intercept.coef_.ndim


def test_linear_regression_sparse(setup, random_state=0):
    # Test that linear regression also works with sparse data
    random_state = check_random_state(random_state)
    for i in range(10):
        n = 100
        X = sparse.eye(n, n)
        beta = random_state.rand(n)
        y = X * beta[:, np.newaxis]
        ols = LinearRegression()

        error_msg = re.escape("Does not support sparse input!")
        with pytest.raises(NotImplementedError, match=error_msg):
            ols.fit(X, y.ravel())


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_linear_regression_sparse_equal_dense(setup, normalize, fit_intercept):
    # Test that linear regression agrees between sparse and dense
    rng = check_random_state(0)
    n_samples = 200
    n_features = 2
    X = rng.randn(n_samples, n_features)
    X[X < 0.1] = 0.0
    Xcsr = sparse.csr_matrix(X)
    y = rng.rand(n_samples)
    params = dict(normalize=normalize, fit_intercept=fit_intercept)
    clf_dense = LinearRegression(**params)
    clf_sparse = LinearRegression(**params)
    clf_dense.fit(X, y)

    if fit_intercept is False:
        error_msg = re.escape("Does not support sparse input!")
        with pytest.raises(NotImplementedError, match=error_msg):
            clf_sparse.fit(Xcsr, y)
    else:
        error_msg = re.escape("Does not support sparse input!")
        with pytest.raises(NotImplementedError, match=error_msg):
            clf_sparse.fit(Xcsr, y)


def test_linear_regression_multiple_outcome(setup, random_state=0):
    # Test multiple-outcome linear regressions
    X, y = make_regression(random_state=random_state)

    Y = np.vstack((y, y)).T
    n_features = X.shape[1]

    reg = LinearRegression()
    reg.fit((X), Y)
    assert reg.coef_.shape == (2, n_features)
    Y_pred = reg.predict(X)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    assert_array_almost_equal(np.vstack((y_pred, y_pred)).T, Y_pred, decimal=3)


def test_linear_regression_sparse_multiple_outcome(setup, random_state=0):
    # Test multiple-outcome linear regressions with sparse data
    random_state = check_random_state(random_state)
    X, y = make_sparse_uncorrelated(random_state=random_state)
    X = sparse.coo_matrix(X)
    Y = np.vstack((y, y)).T

    ols = LinearRegression()
    error_msg = re.escape("Does not support sparse input!")
    with pytest.raises(NotImplementedError, match=error_msg):
        ols.fit(X, Y)

    error_msg = re.escape("Does not support sparse input!")
    with pytest.raises(NotImplementedError, match=error_msg):
        ols.fit(X, y.ravel())


# # When optimize.nnls is implemented, one can utilize this test case
# def test_linear_regression_positive(setup):
#     # Test nonnegative LinearRegression on a simple dataset.
#     X = [[1], [2]]
#     y = [1, 2]

#     reg = LinearRegression(positive=True)
#     reg.fit(X, y)

#     assert_array_almost_equal(reg.coef_, [1])
#     assert_array_almost_equal(reg.intercept_, [0])
#     assert_array_almost_equal(reg.predict(X), [1, 2])

#     # test it also for degenerate input
#     X = [[1]]
#     y = [0]

#     reg = LinearRegression(positive=True)
#     reg.fit(X, y)
#     assert_allclose(reg.coef_, [0])
#     assert_allclose(reg.intercept_, [0])
#     assert_allclose(reg.predict(X), [0])


# # When optimize.nnls is implemented, one can utilize this test case
# def test_linear_regression_positive_multiple_outcome(setup, random_state=0):
#     # Test multiple-outcome nonnegative linear regressions
#     random_state = check_random_state(random_state)
#     X, y = make_sparse_uncorrelated(random_state=random_state)
#     Y = np.vstack((y, y)).T
#     n_features = X.shape[1]

#     ols = LinearRegression(positive=True)
#     ols.fit(X, Y)
#     assert ols.coef_.shape == (2, n_features)
#     assert np.all(ols.coef_.to_numpy() >= 0.)
#     Y_pred = ols.predict(X)
#     ols.fit(X, y.ravel())
#     y_pred = ols.predict(X)
#     assert_allclose(np.vstack((y_pred, y_pred)).T, Y_pred)


def test_linear_regression_positive_vs_nonpositive(setup):
    # Test differences with LinearRegression when positive=False.
    X, y = make_sparse_uncorrelated(random_state=0)

    # reg = LinearRegression(positive=True)
    reg = sklearn_LR(positive=True)
    reg.fit(X, y)
    regn = LinearRegression(positive=False)
    regn.fit(X, y)

    assert np.mean(((reg.coef_ - regn.coef_) ** 2).to_numpy()) > 1e-3


def test_linear_regression_positive_vs_nonpositive_when_positive(setup):
    # Test LinearRegression fitted coefficients
    # when the problem is positive.
    n_samples = 200
    n_features = 4
    X = rng.rand(n_samples, n_features)
    y = X[:, 0] + 2 * X[:, 1] + 3 * X[:, 2] + 1.5 * X[:, 3]

    # reg = LinearRegression(positive=True)
    reg = sklearn_LR(positive=True)
    reg.fit(X, y)
    regn = LinearRegression(positive=False)
    regn.fit(X, y)

    assert np.mean(((reg.coef_ - regn.coef_) ** 2).to_numpy()) < 1e-6


# # Failed: DID NOT WARN.
# # No such warning "pandas.DataFrame with sparse columns found."
# def test_linear_regression_pd_sparse_dataframe_warning():
#     pd = pytest.importorskip('pandas')
#     # restrict the pd versions < '0.24.0'
#     # as they have a bug in is_sparse func
#     if parse_version(pd.__version__) < parse_version('0.24.0'):
#         pytest.skip("pandas 0.24+ required.")

#     # Warning is raised only when some of the columns is sparse
#     df = pd.DataFrame({'0': np.random.randn(10)})
#     for col in range(1, 4):
#         arr = np.random.randn(10)
#         arr[:8] = 0
#         # all columns but the first column is sparse
#         if col != 0:
#             arr = pd.arrays.SparseArray(arr, fill_value=0)
#         df[str(col)] = arr

#     msg = "pandas.DataFrame with sparse columns found."
#     with pytest.warns(UserWarning, match=msg):
#         reg = LinearRegression()
#         reg.fit(df.iloc[:, 0:2], df.iloc[:, 3])

#     # does not warn when the whole dataframe is sparse
#     df['0'] = pd.arrays.SparseArray(df['0'], fill_value=0)
#     assert hasattr(df, "sparse")

#     with pytest.warns(None) as record:
#         reg.fit(df.iloc[:, 0:2], df.iloc[:, 3])
#     assert not record


def test_preprocess_data(setup):
    n_samples = 200
    n_features = 2
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    expected_X_mean = np.mean(X, axis=0)
    expected_X_norm = np.std(X, axis=0) * np.sqrt(X.shape[0])
    expected_y_mean = np.mean(y, axis=0)

    Xt, yt, X_mean, y_mean, X_norm = _preprocess_data(
        X, y, fit_intercept=False, normalize=False
    )
    assert_array_almost_equal(X_mean, np.zeros(n_features))
    assert_array_almost_equal(y_mean, 0)
    assert_array_almost_equal(X_norm, np.ones(n_features))
    assert_array_almost_equal(Xt, X)
    assert_array_almost_equal(yt, y)

    Xt, yt, X_mean, y_mean, X_norm = _preprocess_data(
        X, y, fit_intercept=True, normalize=False
    )
    assert_array_almost_equal(X_mean, expected_X_mean)
    assert_array_almost_equal(y_mean, expected_y_mean)
    assert_array_almost_equal(X_norm, np.ones(n_features))
    assert_array_almost_equal(Xt, X - expected_X_mean)
    assert_array_almost_equal(yt, y - expected_y_mean)

    Xt, yt, X_mean, y_mean, X_norm = _preprocess_data(
        X, y, fit_intercept=True, normalize=True
    )
    assert_array_almost_equal(X_mean, expected_X_mean)
    assert_array_almost_equal(y_mean, expected_y_mean)
    assert_array_almost_equal(X_norm, expected_X_norm)
    assert_array_almost_equal(Xt, (X - expected_X_mean) / expected_X_norm)
    assert_array_almost_equal(yt, y - expected_y_mean)


def test_preprocess_data_multioutput(setup):
    n_samples = 200
    n_features = 3
    n_outputs = 2
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples, n_outputs)
    expected_y_mean = np.mean(y, axis=0)

    # case 1
    _, yt, _, y_mean, _ = _preprocess_data(X, y, fit_intercept=False, normalize=False)
    assert_array_almost_equal(y_mean, np.zeros(n_outputs))
    assert_array_almost_equal(yt, y)

    _, yt, _, y_mean, _ = _preprocess_data(X, y, fit_intercept=True, normalize=False)
    assert_array_almost_equal(y_mean, expected_y_mean)
    assert_array_almost_equal(yt, y - y_mean)

    _, yt, _, y_mean, _ = _preprocess_data(X, y, fit_intercept=True, normalize=True)
    assert_array_almost_equal(y_mean, expected_y_mean)
    assert_array_almost_equal(yt, y - y_mean)

    # case 2
    X = sparse.csc_matrix(X)
    error_msg = "Does not support sparse input!"
    with pytest.raises(NotImplementedError, match=error_msg):
        _, yt, _, y_mean, _ = _preprocess_data(
            X, y, fit_intercept=False, normalize=False
        )

    with pytest.raises(NotImplementedError, match=error_msg):
        _, yt, _, y_mean, _ = _preprocess_data(
            X, y, fit_intercept=True, normalize=False
        )

    with pytest.raises(NotImplementedError, match=error_msg):
        _, yt, _, y_mean, _ = _preprocess_data(X, y, fit_intercept=True, normalize=True)


def test_preprocess_data_weighted(setup):
    n_samples = 200
    n_features = 2
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    sample_weight = rng.rand(n_samples)
    expected_X_mean = np.average(X, axis=0, weights=sample_weight)
    expected_y_mean = np.average(y, axis=0, weights=sample_weight)

    # XXX: if normalize=True, should we expect a weighted standard deviation?
    #      Currently not weighted, but calculated with respect to weighted mean
    expected_X_norm = (
        np.sqrt(X.shape[0]) * np.mean((X - expected_X_mean) ** 2, axis=0) ** 0.5
    )

    Xt, yt, X_mean, y_mean, X_norm = _preprocess_data(
        X, y, fit_intercept=True, normalize=False, sample_weight=sample_weight
    )
    assert_array_almost_equal(X_mean, expected_X_mean)
    assert_array_almost_equal(y_mean, expected_y_mean)
    assert_array_almost_equal(X_norm, np.ones(n_features))
    assert_array_almost_equal(Xt, X - expected_X_mean)
    assert_array_almost_equal(yt, y - expected_y_mean)

    Xt, yt, X_mean, y_mean, X_norm = _preprocess_data(
        X, y, fit_intercept=True, normalize=True, sample_weight=sample_weight
    )
    assert_array_almost_equal(X_mean, expected_X_mean)
    assert_array_almost_equal(y_mean, expected_y_mean)
    assert_array_almost_equal(X_norm, expected_X_norm)
    assert_array_almost_equal(Xt, (X - expected_X_mean) / expected_X_norm)
    assert_array_almost_equal(yt, y - expected_y_mean)


def test_sparse_preprocess_data_with_return_mean(setup):
    n_samples = 200
    n_features = 2
    # random_state not supported yet in sparse.rand
    X = sparse.rand(n_samples, n_features, density=0.5)  # , random_state=rng
    X = sparse.csr_matrix(X)
    y = rng.rand(n_samples)

    error_msg = re.escape("Does not support sparse input!")
    with pytest.raises(NotImplementedError, match=error_msg):
        Xt, yt, X_mean, y_mean, X_norm = _preprocess_data(
            X, y, fit_intercept=False, normalize=False, return_mean=True
        )

    error_msg = re.escape("Does not support sparse input!")
    with pytest.raises(NotImplementedError, match=error_msg):
        Xt, yt, X_mean, y_mean, X_norm = _preprocess_data(
            X,
            y,
            fit_intercept=True,
            normalize=False,
            return_mean=True,
            check_input=False,
        )

    error_msg = re.escape("Does not support sparse input!")
    with pytest.raises(NotImplementedError, match=error_msg):
        Xt, yt, X_mean, y_mean, X_norm = _preprocess_data(
            X,
            y,
            fit_intercept=True,
            normalize=True,
            return_mean=True,
            check_input=False,
        )


# # AttributeError: 'TensorData' object has no attribute 'getformat'
# def test_csr_preprocess_data():
#     # Test output format of _preprocess_data, when input is csr
#     X, y = make_regression()
#     X[X < 2.5] = 0.0
#     csr = sparse.csr_matrix(X)
#     csr_, y, _, _, _ = _preprocess_data(csr, y, True)
#     assert csr_.getformat() == 'csr'


@pytest.mark.parametrize("is_sparse", (True, False))
@pytest.mark.parametrize("to_copy", (True, False))
def test_preprocess_copy_data_no_checks(setup, is_sparse, to_copy):
    X, y = make_regression()
    X[X < 2.5] = 0.0

    if is_sparse:
        X = sparse.csr_matrix(X)
        error_msg = re.escape("Does not support sparse input!")
        with pytest.raises(NotImplementedError, match=error_msg):
            X_, y_, _, _, _ = _preprocess_data(
                X, y, True, copy=to_copy, check_input=False
            )
    else:
        X_, y_, _, _, _ = _preprocess_data(X, y, True, copy=to_copy, check_input=False)

        if to_copy and is_sparse:
            assert not np.may_share_memory(X_.data, X.data)
        elif to_copy:
            assert not np.may_share_memory(X_.to_numpy(), X)
        elif is_sparse:
            assert np.may_share_memory(X_.data, X.data)
        # else:  # fake pass
        #     assert np.may_share_memory(X_.to_numpy(), X)


def test_dtype_preprocess_data(setup):
    n_samples = 200
    n_features = 2
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)

    X_32 = np.asarray(X, dtype=np.float32)
    y_32 = np.asarray(y, dtype=np.float32)
    X_64 = np.asarray(X, dtype=np.float64)
    y_64 = np.asarray(y, dtype=np.float64)

    for fit_intercept in [True, False]:
        for normalize in [True, False]:
            Xt_32, yt_32, X_mean_32, y_mean_32, X_norm_32 = _preprocess_data(
                X_32,
                y_32,
                fit_intercept=fit_intercept,
                normalize=normalize,
                return_mean=True,
            )

            Xt_64, yt_64, X_mean_64, y_mean_64, X_norm_64 = _preprocess_data(
                X_64,
                y_64,
                fit_intercept=fit_intercept,
                normalize=normalize,
                return_mean=True,
            )

            Xt_3264, yt_3264, X_mean_3264, y_mean_3264, X_norm_3264 = _preprocess_data(
                X_32,
                y_64,
                fit_intercept=fit_intercept,
                normalize=normalize,
                return_mean=True,
            )

            Xt_6432, yt_6432, X_mean_6432, y_mean_6432, X_norm_6432 = _preprocess_data(
                X_64,
                y_32,
                fit_intercept=fit_intercept,
                normalize=normalize,
                return_mean=True,
            )

            assert Xt_32.dtype == np.float32
            assert yt_32.dtype == np.float32
            assert X_mean_32.dtype == np.float32
            assert y_mean_32.dtype == np.float32
            assert X_norm_32.dtype == np.float32

            assert Xt_64.dtype == np.float64
            assert yt_64.dtype == np.float64
            assert X_mean_64.dtype == np.float64
            assert y_mean_64.dtype == np.float64
            assert X_norm_64.dtype == np.float64

            assert Xt_3264.dtype == np.float32
            assert yt_3264.dtype == np.float32
            assert X_mean_3264.dtype == np.float32
            assert y_mean_3264.dtype == np.float32
            assert X_norm_3264.dtype == np.float32

            assert Xt_6432.dtype == np.float64
            assert yt_6432.dtype == np.float64
            assert X_mean_6432.dtype == np.float64
            assert y_mean_6432.dtype == np.float64
            assert X_norm_6432.dtype == np.float64

            assert X_32.dtype == np.float32
            assert y_32.dtype == np.float32
            assert X_64.dtype == np.float64
            assert y_64.dtype == np.float64

            assert_array_almost_equal(Xt_32, Xt_64)
            assert_array_almost_equal(yt_32, yt_64)
            assert_array_almost_equal(X_mean_32, X_mean_64)
            assert_array_almost_equal(y_mean_32, y_mean_64)
            assert_array_almost_equal(X_norm_32, X_norm_64)


@pytest.mark.parametrize("n_targets", [None, 2])
def test_rescale_data_dense(setup, n_targets):
    n_samples = 200
    n_features = 2

    sample_weight = 1.0 + rng.rand(n_samples)
    X = rng.rand(n_samples, n_features)
    if n_targets is None:
        y = rng.rand(n_samples)
    else:
        y = rng.rand(n_samples, n_targets)
    rescaled_X, rescaled_y = _rescale_data(X, y, sample_weight)
    rescaled_X2 = X * np.sqrt(sample_weight)[:, np.newaxis]
    if n_targets is None:
        rescaled_y2 = y * np.sqrt(sample_weight)
    else:
        rescaled_y2 = y * np.sqrt(sample_weight)[:, np.newaxis]
    assert_array_almost_equal(rescaled_X, rescaled_X2)
    assert_array_almost_equal(rescaled_y, rescaled_y2)


def test_fused_types_make_dataset(setup):
    iris = load_iris()

    X_32 = iris.data.astype(np.float32)
    y_32 = iris.target.astype(np.float32)
    X_csr_32 = sparse.csr_matrix(X_32)
    sample_weight_32 = np.arange(y_32.size, dtype=np.float32)

    X_64 = iris.data.astype(np.float64)
    y_64 = iris.target.astype(np.float64)
    X_csr_64 = sparse.csr_matrix(X_64)
    sample_weight_64 = np.arange(y_64.size, dtype=np.float64)

    # array
    dataset_32, _ = make_dataset(X_32, y_32, sample_weight_32)
    dataset_64, _ = make_dataset(X_64, y_64, sample_weight_64)
    xi_32, yi_32, _, _ = dataset_32._next_py()
    xi_64, yi_64, _, _ = dataset_64._next_py()
    xi_data_32, _, _ = xi_32
    xi_data_64, _, _ = xi_64

    assert xi_data_32.dtype == np.float32
    assert xi_data_64.dtype == np.float64
    assert_allclose(yi_64, yi_32, rtol=rtol)

    # csr
    datasetcsr_32, _ = make_dataset(X_csr_32, y_32, sample_weight_32)
    datasetcsr_64, _ = make_dataset(X_csr_64, y_64, sample_weight_64)
    xicsr_32, yicsr_32, _, _ = datasetcsr_32._next_py()
    xicsr_64, yicsr_64, _, _ = datasetcsr_64._next_py()
    xicsr_data_32, _, _ = xicsr_32
    xicsr_data_64, _, _ = xicsr_64

    assert xicsr_data_32.dtype == np.float32
    assert xicsr_data_64.dtype == np.float64

    assert_allclose(xicsr_data_64, xicsr_data_32, rtol=rtol)
    assert_allclose(yicsr_64, yicsr_32, rtol=rtol)

    assert_array_equal(xi_data_32, xicsr_data_32)
    assert_array_equal(xi_data_64, xicsr_data_64)
    assert_array_equal(yi_32, yicsr_32)
    assert_array_equal(yi_64, yicsr_64)


def test_raise_notimplemented_when_positive(setup):
    error_msg = re.escape("Does not support positive coefficients!")

    X = [[1], [2]]
    y = [1, 2]

    reg = LinearRegression(positive=True)
    with pytest.raises(NotImplementedError, match=error_msg):
        reg.fit(X, y)
