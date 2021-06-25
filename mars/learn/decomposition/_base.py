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


from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin

from ... import tensor as mt
from ...tensor import linalg
from ..utils import check_array
from ..utils.validation import check_is_fitted


# -----------------------------------------------------------
# Original implementation is in `sklearn.decomposition.base`.
# -----------------------------------------------------------


class _BasePCA(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """Base class for PCA methods.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    def get_covariance(self, session=None):
        """Compute data covariance with the generative model.

        ``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
        where S**2 contains the explained variances, and sigma2 contains the
        noise variances.

        Returns
        -------
        cov : Tensor, shape=(n_features, n_features)
            Estimated covariance of data.
        """
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * mt.sqrt(exp_var[:, mt.newaxis])
        exp_var_diff = mt.maximum(exp_var - self.noise_variance_, 0.)
        cov = mt.dot(components_.T * exp_var_diff, components_)
        cov.flat[::len(cov) + 1] += self.noise_variance_  # modify diag inplace
        cov.execute(session=session)
        return cov

    def get_precision(self, session=None):
        """Compute data precision matrix with the generative model.

        Equals the inverse of the covariance but computed with
        the matrix inversion lemma for efficiency.

        Returns
        -------
        precision : Tensor, shape=(n_features, n_features)
            Estimated precision of data.
        """
        n_features = self.components_.shape[1]

        # handle corner cases first
        if self.n_components_ == 0:
            precision = mt.eye(n_features) / self.noise_variance_
            precision.execute(session=session)
            return precision
        if self.n_components_ == n_features:
            precision = linalg.inv(self.get_covariance())
            precision.execute(session=session)
            return precision

        # Get precision using matrix inversion lemma
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * mt.sqrt(exp_var[:, mt.newaxis])
        exp_var_diff = mt.maximum(exp_var - self.noise_variance_, 0.)
        precision = mt.dot(components_, components_.T) / self.noise_variance_
        precision.flat[::len(precision) + 1] += 1. / exp_var_diff
        precision = mt.dot(components_.T,
                           mt.dot(linalg.inv(precision), components_))
        precision /= -(self.noise_variance_ ** 2)
        precision.flat[::len(precision) + 1] += 1. / self.noise_variance_
        precision.execute(session=session)
        return precision

    @abstractmethod
    def fit(X, y=None, session=None, run_kwargs=None):
        """Placeholder for fit. Subclasses should implement this method!

        Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

    def transform(self, X, session=None):
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        session : session to run

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        Examples
        --------

        >>> import numpy as np
        >>> from sklearn.decomposition import IncrementalPCA
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> ipca = IncrementalPCA(n_components=2, batch_size=3)
        >>> ipca.fit(X)
        IncrementalPCA(batch_size=3, copy=True, n_components=2, whiten=False)
        >>> ipca.transform(X) # doctest: +SKIP
        """
        check_is_fitted(self, ['mean_', 'components_'], all_or_any=all)

        X = check_array(X)
        if self.mean_ is not None:
            X = X - self.mean_
        X_transformed = mt.dot(X, self.components_.T)
        if self.whiten:
            X_transformed /= mt.sqrt(self.explained_variance_)
        X_transformed.execute(session=session)
        return X_transformed

    def inverse_transform(self, X, session=None):
        """Transform data back to its original space.

        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.
        session : session to run

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)

        Notes
        -----
        If whitening is enabled, inverse_transform will compute the
        exact inverse operation, which includes reversing whitening.
        """
        if self.whiten:
            ret = (mt.dot(X, mt.sqrt(self.explained_variance_[:, mt.newaxis]) *
                          self.components_) + self.mean_)
        else:
            ret = (mt.dot(X, self.components_) + self.mean_)
        ret.execute(session=session)
        return ret
