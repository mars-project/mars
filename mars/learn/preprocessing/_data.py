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

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES

from ... import tensor as mt
from ...tensor.core import TENSOR_TYPE
from ..base import BaseEstimator
from ..utils.validation import check_array


def _handle_zeros_in_scale(scale, copy=True):
    """Makes sure that whenever scale is zero, we handle it correctly.

    This happens in most scalers when we have constant features.
    """

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):  # pragma: no cover
        if scale == .0:
            scale = 1.
        return scale
    elif hasattr(scale, 'ndim') and scale.ndim == 0:  # pragma: no cover
        # scalar that is tensor
        return mt.where(scale == .0, 1., scale)
    elif isinstance(scale, (np.ndarray, TENSOR_TYPE)):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


class MinMaxScaler(TransformerMixin, BaseEstimator):
    """Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.

    The transformation is given by::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).

    clip: bool, default=False
        Set to True to clip transformed values of held-out data to
        provided `feature range`.

    Attributes
    ----------
    min_ : Tensor of shape (n_features,)
        Per feature adjustment for minimum. Equivalent to
        ``min - X.min(axis=0) * self.scale_``

    scale_ : Tensor of shape (n_features,)
        Per feature relative scaling of the data. Equivalent to
        ``(max - min) / (X.max(axis=0) - X.min(axis=0))``

    data_min_ : ndarray of shape (n_features,)
        Per feature minimum seen in the data

    data_max_ : ndarray of shape (n_features,)
        Per feature maximum seen in the data

    data_range_ : ndarray of shape (n_features,)
        Per feature range ``(data_max_ - data_min_)`` seen in the data

    n_samples_seen_ : int
        The number of samples processed by the estimator.
        It will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.

    Examples
    --------
    >>> from mars.learn.preprocessing import MinMaxScaler
    >>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    >>> scaler = MinMaxScaler()
    >>> print(scaler.fit(data))
    MinMaxScaler()
    >>> print(scaler.data_max_)
    [ 1. 18.]
    >>> print(scaler.transform(data))
    [[0.   0.  ]
     [0.25 0.25]
     [0.5  0.5 ]
     [1.   1.  ]]
    >>> print(scaler.transform([[2, 2]]))
    [[1.5 0. ]]

    See Also
    --------
    minmax_scale : Equivalent function without the estimator API.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """

    def __init__(self, feature_range=(0, 1), copy=True, clip=False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def _reset(self):  # pragma: no cover
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X, y=None, session=None, run_kwargs=None):
        """Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y, session=session, run_kwargs=run_kwargs)

    def partial_fit(self, X, y=None, session=None, run_kwargs=None):
        """Online computation of min and max on X for later scaling.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(feature_range))

        if mt.tensor(X).issparse():  # pragma: no cover
            raise TypeError("MinMaxScaler does not support sparse input. "
                            "Consider using MaxAbsScaler instead.")

        first_pass = not hasattr(self, 'n_samples_seen_')
        X = self._validate_data(X, reset=first_pass,
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite="allow-nan")

        if np.isnan(X.shape[0]):  # pragma: no cover
            X.execute(session=session, **(run_kwargs or dict()))

        data_min = mt.nanmin(X, axis=0)
        data_max = mt.nanmax(X, axis=0)

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            data_min = mt.minimum(self.data_min_, data_min)  # pylint: disable=access-member-before-definition
            data_max = mt.maximum(self.data_max_, data_max)  # pylint: disable=access-member-before-definition
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = ((feature_range[1] - feature_range[0]) /
                       _handle_zeros_in_scale(data_range))
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        mt.ExecutableTuple([self.scale_, self.min_, self.data_min_,
                            self.data_max_, self.data_range_]).execute(
            session=session, **(run_kwargs or dict()))
        return self

    def transform(self, X, session=None, run_kwargs=None):
        """Scale features of X according to feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        X = self._validate_data(X, copy=self.copy, dtype=FLOAT_DTYPES,
                                force_all_finite="allow-nan", reset=False)

        X *= self.scale_
        X += self.min_
        if self.clip:
            X = mt.clip(X, self.feature_range[0], self.feature_range[1])
        return X.execute(session=session, **(run_kwargs or dict()))

    def inverse_transform(self, X, session=None, run_kwargs=None):
        """Undo the scaling of X according to feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
                        force_all_finite="allow-nan")

        X -= self.min_
        X /= self.scale_
        return X.execute(session=session, **(run_kwargs or dict()))

    def _more_tags(self):  # pylint: disable=no-self-use
        return {'allow_nan': True}


def minmax_scale(X, feature_range=(0, 1), *, axis=0, copy=True,
                 session=None, run_kwargs=None):
    """Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, i.e. between
    zero and one.

    The transformation is given by (when ``axis=0``)::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    The transformation is calculated as (when ``axis=0``)::

       X_scaled = scale * X + min - X.min(axis=0) * scale
       where scale = (max - min) / (X.max(axis=0) - X.min(axis=0))

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    .. versionadded:: 0.17
       *minmax_scale* function interface
       to :class:`~sklearn.preprocessing.MinMaxScaler`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data.

    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    axis : int, default=0
        Axis used to scale along. If 0, independently scale each feature,
        otherwise (if 1) scale each sample.

    copy : bool, default=True
        Set to False to perform inplace scaling and avoid a copy (if the input
        is already a numpy array).

    Returns
    -------
    X_tr : ndarray of shape (n_samples, n_features)
        The transformed data.

    .. warning:: Risk of data leak

        Do not use :func:`~sklearn.preprocessing.minmax_scale` unless you know
        what you are doing. A common mistake is to apply it to the entire data
        *before* splitting into training and test sets. This will bias the
        model evaluation because information would have leaked from the test
        set to the training set.
        In general, we recommend using
        :class:`~sklearn.preprocessing.MinMaxScaler` within a
        :ref:`Pipeline <pipeline>` in order to prevent most risks of data
        leaking: `pipe = make_pipeline(MinMaxScaler(), LogisticRegression())`.

    See Also
    --------
    MinMaxScaler : Performs scaling to a given range using the Transformer
        API (e.g. as part of a preprocessing
        :class:`~sklearn.pipeline.Pipeline`).

    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """  # noqa
    # Unlike the scaler object, this function allows 1d input.
    # If copy is required, it will be done inside the scaler object.
    X = check_array(X, copy=False, ensure_2d=False,
                    dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
    original_ndim = X.ndim

    if original_ndim == 1:
        X = X.reshape(X.shape[0], 1)

    s = MinMaxScaler(feature_range=feature_range, copy=copy)
    if axis == 0:
        X = s.fit_transform(X)
    else:
        X = s.fit_transform(X.T).T

    if original_ndim == 1:
        X = X.ravel()

    return X.execute(session=session, **(run_kwargs or dict()))
