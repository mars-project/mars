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
from sklearn.base import BaseEstimator, TransformerMixin

from ... import tensor as mt
from ...tensor.linalg import randomized_svd
from ...tensor.utils import check_random_state
from ...core import ExecutableTuple
from ..utils import check_array

__all__ = ["TruncatedSVD"]


class TruncatedSVD(BaseEstimator, TransformerMixin):
    """Dimensionality reduction using truncated SVD (aka LSA).

    This transformer performs linear dimensionality reduction by means of
    truncated singular value decomposition (SVD). Contrary to PCA, this
    estimator does not center the data before computing the singular value
    decomposition. This means it can work with scipy.sparse matrices
    efficiently.

    In particular, truncated SVD works on term count/tf-idf matrices as
    returned by the vectorizers in sklearn.feature_extraction.text. In that
    context, it is known as latent semantic analysis (LSA).

    This estimator supports two algorithms: a fast randomized SVD solver, and
    a "naive" algorithm that uses ARPACK as an eigensolver on (X * X.T) or
    (X.T * X), whichever is more efficient.

    Read more in the :ref:`User Guide <LSA>`.

    Parameters
    ----------
    n_components : int, default = 2
        Desired dimensionality of output data.
        Must be strictly less than the number of features.
        The default value is useful for visualisation. For LSA, a value of
        100 is recommended.

    algorithm : string, default = "randomized"
        SVD solver to use. Either "arpack" for the ARPACK wrapper in SciPy
        (scipy.sparse.linalg.svds), or "randomized" for the randomized
        algorithm due to Halko (2009).

    n_iter : int, optional (default 5)
        Number of iterations for randomized SVD solver. Not used by ARPACK.
        The default is larger than the default in `randomized_svd` to handle
        sparse matrices that may have large slowly decaying spectrum.

    random_state : int, RandomState instance or None, optional, default = None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    tol : float, optional
        Tolerance for ARPACK. 0 means machine precision. Ignored by randomized
        SVD solver.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)

    explained_variance_ : array, shape (n_components,)
        The variance of the training samples transformed by a projection to
        each component.

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.

    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

    Examples
    --------
    >>> from mars.learn.decomposition import TruncatedSVD
    >>> import mars.tensor as mt
    >>> from sklearn.random_projection import sparse_random_matrix
    >>> X = mt.tensor(sparse_random_matrix(100, 100, density=0.01, random_state=42))
    >>> svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    >>> svd.fit(X)  # doctest: +NORMALIZE_WHITESPACE
    TruncatedSVD(algorithm='randomized', n_components=5, n_iter=7,
            random_state=42, tol=0.0)
    >>> print(svd.explained_variance_ratio_)  # doctest: +ELLIPSIS
    [0.0606... 0.0584... 0.0497... 0.0434... 0.0372...]
    >>> print(svd.explained_variance_ratio_.sum())  # doctest: +ELLIPSIS
    0.249...
    >>> print(svd.singular_values_)  # doctest: +ELLIPSIS
    [2.5841... 2.5245... 2.3201... 2.1753... 2.0443...]

    See also
    --------
    PCA

    References
    ----------
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) https://arxiv.org/pdf/0909.4061.pdf

    Notes
    -----
    SVD suffers from a problem called "sign indeterminacy", which means the
    sign of the ``components_`` and the output from transform depend on the
    algorithm and random state. To work around this, fit instances of this
    class to data once, then keep the instance around to do transformations.

    """
    def __init__(self, n_components=2, algorithm="randomized", n_iter=5,
                 random_state=None, tol=0.):
        self.algorithm = algorithm
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol

    def fit(self, X, y=None, session=None):
        """Fit LSI model on training data X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.
        session : session to run

        y : Ignored

        Returns
        -------
        self : object
            Returns the transformer object.
        """
        self.fit_transform(X, session=session)
        return self

    def fit_transform(self, X, y=None, session=None):
        """Fit LSI model to X and perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.
        session : session to run

        y : Ignored

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        X = check_array(X, accept_sparse=['csr', 'csc'],
                        ensure_min_features=2)
        random_state = check_random_state(self.random_state)

        if self.algorithm == "arpack":
            # U, Sigma, VT = svds(X, k=self.n_components, tol=self.tol)
            # # svds doesn't abide by scipy.linalg.svd/randomized_svd
            # # conventions, so reverse its outputs.
            # Sigma = Sigma[::-1]
            # U, VT = svd_flip(U[:, ::-1], VT[::-1])
            raise NotImplementedError('Does not support arpack for truncated_svd')

        elif self.algorithm == "randomized":
            k = self.n_components
            n_features = X.shape[1]
            if k >= n_features:
                raise ValueError("n_components must be < n_features; "
                                 f"got {k} >= {n_features}")
            U, Sigma, VT = randomized_svd(X, self.n_components,
                                          n_iter=self.n_iter,
                                          random_state=random_state)
        else:
            raise ValueError(f"unknown algorithm {self.algorithm!r}")

        self.components_ = VT

        # Calculate explained variance & explained variance ratio
        X_transformed = U * Sigma
        self.explained_variance_ = exp_var = np.var(X_transformed, axis=0)
        full_var = mt.var(X, axis=0).sum()
        self.explained_variance_ratio_ = exp_var / full_var
        self.singular_values_ = Sigma  # Store the singular values.

        to_run_tensors = [X_transformed, self.components_, self.explained_variance_,
                          self.explained_variance_ratio_, self.singular_values_]

        ExecutableTuple(to_run_tensors).execute(session=session)
        return X_transformed

    def transform(self, X, session=None):
        """Perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data.
        session : session to run

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        X = check_array(X, accept_sparse='csr')
        ret = mt.dot(X, self.components_.T)
        ret.execute(session=session)
        return ret

    def inverse_transform(self, X, session=None):
        """Transform X back to its original space.

        Returns an array X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data.
        session : session to run

        Returns
        -------
        X_original : array, shape (n_samples, n_features)
            Note that this is always a dense array.
        """
        X = check_array(X)
        ret = mt.dot(X, self.components_)
        ret.execute(session=session)
        return ret
