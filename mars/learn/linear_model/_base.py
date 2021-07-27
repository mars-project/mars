from abc import ABCMeta, abstractmethod
import numbers


from sklearn.utils.validation import (check_is_fitted,
                                      _deprecate_positional_args)
from sklearn.utils.sparsefuncs import mean_variance_axis, inplace_column_scale
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize as f_normalize
from sklearn.utils.fixes import delayed

import numpy as np
from joblib import Parallel
import scipy.sparse as sp
from scipy import sparse
from scipy import optimize
from scipy.sparse.linalg import lsqr as sparse_lsqr
from scipy import linalg

from ..base import BaseEstimator, RegressorMixin, MultiOutputMixin
from ..utils.validation import _check_sample_weight, check_array, FLOAT_DTYPES

# from mars.tensor.datasource import tensor as astensor


def _preprocess_data(
    X,
    y,
    fit_intercept,
    normalize=False,
    copy=True,
    sample_weight=None,
    return_mean=False,
    check_input=True,
):
    """Center and scale data.

    Centers data to have mean zero along axis 0. If fit_intercept=False or if
    the X is a sparse matrix, no centering is done, but normalization can still
    be applied. The function returns the statistics necessary to reconstruct
    the input data, which are X_offset, y_offset, X_scale, such that the output

        X = (X - X_offset) / X_scale

    X_scale is the L2 norm of X - X_offset. If sample_weight is not None,
    then the weighted mean of X and y is zero, and not the mean itself. If
    return_mean=True, the mean, eventually weighted, is returned, independently
    of whether X was centered (option used for optimization with sparse data in
    coordinate_descend).

    This is here because nearly all linear models will want their data to be
    centered. This function also systematically makes y consistent with X.dtype
    """
    if isinstance(sample_weight, numbers.Number):
        sample_weight = None
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    if check_input:
        X = check_array(X, copy=copy, accept_sparse=['csr', 'csc'],
                        dtype=FLOAT_DTYPES)
    elif copy:
        if sp.issparse(X):
            X = X.copy()
        else:
            X = X.copy(order='K')

    y = np.asarray(y, dtype=X.dtype)

    if fit_intercept:
        if sp.issparse(X):
            X_offset, X_var = mean_variance_axis(X, axis=0)
            if not return_mean:
                X_offset[:] = X.dtype.type(0)

            if normalize:

                # TODO: f_normalize could be used here as well but the function
                # inplace_csr_row_normalize_l2 must be changed such that it
                # can return also the norms computed internally

                # transform variance to norm in-place
                X_var *= X.shape[0]
                X_scale = np.sqrt(X_var, X_var)
                del X_var
                X_scale[X_scale == 0] = 1
                inplace_column_scale(X, 1. / X_scale)
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)

        else:
            X_offset = np.average(X, axis=0, weights=sample_weight)
            X -= X_offset
            if normalize:
                X, X_scale = f_normalize(X, axis=0, copy=False,
                                         return_norm=True)
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)
        y_offset = np.average(y, axis=0, weights=sample_weight)
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        X_scale = np.ones(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale


# TODO: _rescale_data should be factored into _preprocess_data.
# Currently, the fact that sag implements its own way to deal with
# sample_weight makes the refactoring tricky.


def _rescale_data(X, y, sample_weight):
    """Rescale data sample-wise by square root of sample_weight.

    For many linear models, this enables easy support for sample_weight.

    Returns
    -------
    X_rescaled : {array-like, sparse matrix}

    y_rescaled : {array-like, sparse matrix}
    """
    n_samples = X.shape[0]
    sample_weight = np.asarray(sample_weight)
    if sample_weight.ndim == 0:
        sample_weight = np.full(
            n_samples,
            sample_weight,
            dtype=sample_weight.dtype)
    sample_weight = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix(
        (sample_weight, 0),
        shape=(n_samples, n_samples))
    X = safe_sparse_dot(sw_matrix, X)
    y = safe_sparse_dot(sw_matrix, y)
    return X, y


class LinearModel(BaseEstimator, metaclass=ABCMeta):
    """Base class for Linear Models"""

    @abstractmethod
    def fit(self, X, y):
        """Fit model."""

    def _decision_function(self, X):
        check_is_fitted(self)

        X = self._validate_data(X,
                                y="no_validation",
                                accept_sparse=["csr", "csc", "coo"],
                                reset=False)
        return safe_sparse_dot(X,
                               self.coef_.T,
                               dense_output=True) + self.intercept_

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        return self._decision_function(X)

    _preprocess_data = staticmethod(_preprocess_data)

    def _set_intercept(self, X_offset, y_offset, X_scale):
        """Set the intercept_"""
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_scale
            self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)
        else:
            self.intercept_ = 0.0

    def _more_tags(self):
        return {"requires_y": True}


class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
    """
    Ordinary least squares Linear Regression.

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

        .. deprecated:: 1.0
           `normalize` was deprecated in version 1.0 and will be
           removed in 1.2.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This will only provide
        speedup for n_targets > 1 and sufficient large problems.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive. This
        option is only supported for dense arrays.

        .. versionadded:: 0.24

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    rank_ : int
        Rank of matrix `X`. Only available when `X` is dense.

    singular_ : array of shape (min(X, y),)
        Singular values of `X`. Only available when `X` is dense.

    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if
        `fit_intercept = False`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    See Also
    --------
    Ridge : Ridge regression addresses some of the
        problems of Ordinary Least Squares by imposing a penalty on the
        size of the coefficients with l2 regularization.
    Lasso : The Lasso is a linear model that estimates
        sparse coefficients with l1 regularization.
    ElasticNet : Elastic-Net is a linear regression
        model trained with both l1 and l2 -norm regularization of the
        coefficients.

    Notes
    -----
    From the implementation point of view, this is just plain Ordinary
    Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares
    (scipy.optimize.nnls) wrapped as a predictor object.

    Examples
    --------
    TBD
    """
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        fit_intercept=True,
        normalize="deprecated",
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        n_jobs_ = self.n_jobs

        accept_sparse = False if self.positive else ['csr', 'csc', 'coo']

        X, y = self._validate_data(X, y, accept_sparse=accept_sparse,
                                   y_numeric=True, multi_output=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=sample_weight,
            return_mean=True)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        if self.positive:
            if y.ndim < 2:
                self.coef_, self._residues = optimize.nnls(X, y)
            else:
                # scipy.optimize.nnls cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(optimize.nnls)(
                        X, y[:, j]) for j in range(y.shape[1]))
                self.coef_, self._residues = map(np.vstack, zip(*outs))
        elif sp.issparse(X):
            X_offset_scale = X_offset / X_scale

            def matvec(b):
                return X.dot(b) - b.dot(X_offset_scale)

            def rmatvec(b):
                return X.T.dot(b) - X_offset_scale * np.sum(b)

            X_centered = sparse.linalg.LinearOperator(shape=X.shape,
                                                      matvec=matvec,
                                                      rmatvec=rmatvec)

            if y.ndim < 2:
                out = sparse_lsqr(X_centered, y)
                self.coef_ = out[0]
                self._residues = out[3]
            else:
                # sparse_lstsq cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(sparse_lsqr)(X_centered, y[:, j].ravel())
                    for j in range(y.shape[1]))
                self.coef_ = np.vstack([out[0] for out in outs])
                self._residues = np.vstack([out[3] for out in outs])
        else:
            self.coef_, self._residues, self.rank_, self.singular_ = \
                linalg.lstsq(X, y)
            self.coef_ = self.coef_.T

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)
        self._set_intercept(X_offset, y_offset, X_scale)
        return self
