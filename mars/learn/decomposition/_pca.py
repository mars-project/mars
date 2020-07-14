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


from math import log, sqrt
import numbers

import numpy as np
from scipy.special import gammaln
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import fast_logdet

from ... import tensor as mt
from ... import remote as mr
from ...core import Base, Entity
from ...tensor.array_utils import get_array_module
from ...tensor.core import TENSOR_TYPE
from ...tensor.utils import check_random_state
from ...tensor.linalg import randomized_svd
from ...tensor.linalg.randomized_svd import svd_flip
from ...lib.sparse import issparse
from ...core import ExecutableTuple
from ..utils import check_array
from ._base import _BasePCA


def _assess_dimension(spectrum, rank, n_samples):
    """Compute the log-likelihood of a rank ``rank`` dataset.

    The dataset is assumed to be embedded in gaussian noise of shape(n,
    dimf) having spectrum ``spectrum``.

    Parameters
    ----------
    spectrum : array of shape (n_features)
        Data spectrum.
    rank : int
        Tested rank value. It should be strictly lower than n_features,
        otherwise the method isn't specified (division by zero in equation
        (31) from the paper).
    n_samples : int
        Number of samples.

    Returns
    -------
    ll : float,
        The log-likelihood

    Notes
    -----
    This implements the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`
    """

    xp = get_array_module(spectrum, nosparse=True)

    n_features = spectrum.shape[0]
    if not 1 <= rank < n_features:  # pragma: no cover
        raise ValueError("the tested rank should be in [1, n_features - 1]")

    eps = 1e-15

    if spectrum[rank - 1] < eps:  # pragma: no cover
        # When the tested rank is associated with a small eigenvalue, there's
        # no point in computing the log-likelihood: it's going to be very
        # small and won't be the max anyway. Also, it can lead to numerical
        # issues below when computing pa, in particular in log((spectrum[i] -
        # spectrum[j]) because this will take the log of something very small.
        return -np.inf

    pu = -rank * log(2.)
    for i in range(1, rank + 1):
        pu += (gammaln((n_features - i + 1) / 2.) -
               log(np.pi) * (n_features - i + 1) / 2.)

    pl = xp.sum(xp.log(spectrum[:rank]))
    pl = -pl * n_samples / 2.

    v = max(eps, xp.sum(spectrum[rank:]) / (n_features - rank))
    pv = -xp.log(v) * n_samples * (n_features - rank) / 2.

    m = n_features * rank - rank * (rank + 1.) / 2.
    pp = log(2. * np.pi) * (m + rank) / 2.

    pa = 0.
    spectrum_ = spectrum.copy()
    spectrum_[rank:n_features] = v
    for i in range(rank):
        for j in range(i + 1, len(spectrum)):
            pa += log((spectrum[i] - spectrum[j]) *
                      (1. / spectrum_[j] - 1. / spectrum_[i])) + log(n_samples)

    ll = pu + pl + pv + pp - pa / 2. - rank * log(n_samples) / 2.

    return ll


def _infer_dimension(spectrum, n_samples):
    """Infers the dimension of a dataset with a given spectrum.

    The returned value will be in [1, n_features - 1].
    """
    if isinstance(spectrum, (Base, Entity)):
        spectrum = spectrum.fetch()
    xp = get_array_module(spectrum, nosparse=True)

    ll = xp.empty_like(spectrum)
    ll[0] = -np.inf  # we don't want to return n_components = 0
    for rank in range(1, spectrum.shape[0]):
        ll[rank] = _assess_dimension(spectrum, rank, n_samples)
    return ll.argmax()


class PCA(_BasePCA):
    """Principal component analysis (PCA)

    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the SVD.

    It uses the LAPACK implementation of the full SVD or a randomized truncated
    SVD by the method of Halko et al. 2009, depending on the shape of the input
    data and the number of components to extract.

    It can also use the scipy.sparse.linalg ARPACK implementation of the
    truncated SVD.

    Notice that this class does not support sparse input. See
    :class:`TruncatedSVD` for an alternative with sparse data.

    Read more in the :ref:`User Guide <PCA>`.

    Parameters
    ----------
    n_components : int, float, None or string
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
        MLE is used to guess the dimension. Use of ``n_components == 'mle'``
        will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.

        If ``svd_solver == 'arpack'``, the number of components must be
        strictly less than the minimum of n_features and n_samples.

        Hence, the None case results in::

            n_components == min(n_samples, n_features) - 1

    copy : bool (default True)
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    whiten : bool, optional (default False)
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    svd_solver : string {'auto', 'full', 'arpack', 'randomized'}
        auto :
            the solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < min(X.shape)
        randomized :
            run randomized SVD by the method of Halko et al.

    tol : float >= 0, optional (default .0)
        Tolerance for singular values computed by svd_solver == 'arpack'.

    iterated_power : int >= 0, or 'auto', (default 'auto')
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.

    random_state : int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``svd_solver`` == 'arpack' or 'randomized'.

    Attributes
    ----------
    components_ : tensor, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        ``explained_variance_``.

    explained_variance_ : tensor, shape (n_components,)
        The amount of variance explained by each of the selected components.

        Equal to n_components largest eigenvalues
        of the covariance matrix of X.

    explained_variance_ratio_ : tensor, shape (n_components,)
        Percentage of variance explained by each of the selected components.

        If ``n_components`` is not set then all components are stored and the
        sum of the ratios is equal to 1.0.

    singular_values_ : tensor, shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

    mean_ : tensor, shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

        Equal to `X.mean(axis=0)`.

    n_components_ : int
        The estimated number of components. When n_components is set
        to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
        number is estimated from input data. Otherwise it equals the parameter
        n_components, or the lesser value of n_features and n_samples
        if n_components is None.

    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        compute the estimated data covariance and score samples.

        Equal to the average of (min(n_features, n_samples) - n_components)
        smallest eigenvalues of the covariance matrix of X.

    References
    ----------
    For n_components == 'mle', this class uses the method of *Minka, T. P.
    "Automatic choice of dimensionality for PCA". In NIPS, pp. 598-604*

    Implements the probabilistic PCA model from:
    Tipping, M. E., and Bishop, C. M. (1999). "Probabilistic principal
    component analysis". Journal of the Royal Statistical Society:
    Series B (Statistical Methodology), 61(3), 611-622.
    via the score and score_samples methods.
    See http://www.miketipping.com/papers/met-mppca.pdf

    For svd_solver == 'arpack', refer to `scipy.sparse.linalg.svds`.

    For svd_solver == 'randomized', see:
    *Halko, N., Martinsson, P. G., and Tropp, J. A. (2011).
    "Finding structure with randomness: Probabilistic algorithms for
    constructing approximate matrix decompositions".
    SIAM review, 53(2), 217-288.* and also
    *Martinsson, P. G., Rokhlin, V., and Tygert, M. (2011).
    "A randomized algorithm for the decomposition of matrices".
    Applied and Computational Harmonic Analysis, 30(1), 47-68.*


    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.learn.decomposition import PCA
    >>> X = mt.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> pca = PCA(n_components=2)
    >>> pca.fit(X)  # doctest: +NORMALIZE_WHITESPACE
    PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)
    >>> print(pca.explained_variance_ratio_)  # doctest: +ELLIPSIS
    [0.9924... 0.0075...]
    >>> print(pca.singular_values_)  # doctest: +ELLIPSIS
    [6.30061... 0.54980...]

    >>> pca = PCA(n_components=2, svd_solver='full')
    >>> pca.fit(X)                 # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
      svd_solver='full', tol=0.0, whiten=False)
    >>> print(pca.explained_variance_ratio_)  # doctest: +ELLIPSIS
    [0.9924... 0.00755...]
    >>> print(pca.singular_values_)  # doctest: +ELLIPSIS
    [6.30061... 0.54980...]

    See also
    --------
    KernelPCA
    SparsePCA
    TruncatedSVD
    IncrementalPCA
    """

    def __init__(self, n_components=None, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    def fit(self, X, y=None, session=None, run_kwargs=None):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X, session=session, run=True, run_kwargs=run_kwargs)
        return self

    def fit_transform(self, X, y=None, session=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        U, S, _ = self._fit(X, session=session, run=False)
        U = U[:, :self.n_components_]

        if self.whiten:
            # X_new = X * V / S * sqrt(n_samples) = U * sqrt(n_samples)
            U *= sqrt(X.shape[0] - 1)
        else:
            # X_new = X * V = U * S * V^T * V = U * S
            U *= S[:self.n_components_]

        self._run([U], session=session)
        return U

    def _run(self, result, session=None, run_kwargs=None):
        to_run_tensors = list(result)
        if isinstance(self.noise_variance_, TENSOR_TYPE):
            to_run_tensors.append(self.noise_variance_)
        to_run_tensors.append(self.components_)
        to_run_tensors.append(self.explained_variance_)
        to_run_tensors.append(self.explained_variance_ratio_)
        to_run_tensors.append(self.singular_values_)

        ExecutableTuple(to_run_tensors).execute(session=session, **(run_kwargs or {}))

    def _fit(self, X, session=None, run=True, run_kwargs=None):
        """Dispatch to the right submethod depending on the chosen solver."""

        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        if (hasattr(X, 'issparse') and X.issparse()) or issparse(X):
            raise TypeError('PCA does not support sparse input. See '
                            'TruncatedSVD for a possible alternative.')

        X = check_array(X, dtype=[mt.float64, mt.float32], ensure_2d=True,
                        copy=self.copy)

        # Handle n_components==None
        if self.n_components is None:
            if self.svd_solver != 'arpack':
                n_components = min(X.shape)
            else:
                n_components = min(X.shape) - 1
        else:
            n_components = self.n_components

        # Handle svd_solver
        self._fit_svd_solver = self.svd_solver
        if self._fit_svd_solver == 'auto':
            # Small problem or n_components == 'mle', just call full PCA
            if max(X.shape) <= 500 or n_components == 'mle':
                self._fit_svd_solver = 'full'
            elif n_components >= 1 and n_components < .8 * min(X.shape):
                self._fit_svd_solver = 'randomized'
            # This is also the case of n_components in (0,1)
            else:
                self._fit_svd_solver = 'full'

        # Call different fits for either full or truncated SVD
        if self._fit_svd_solver == 'full':
            ret = self._fit_full(X, n_components, session=session)
        elif self._fit_svd_solver in ['arpack', 'randomized']:
            ret = self._fit_truncated(X, n_components, self._fit_svd_solver)
        else:
            raise ValueError("Unrecognized svd_solver='{0}'"
                             "".format(self._fit_svd_solver))

        if run:
            self._run(ret, session=session, run_kwargs=run_kwargs)
        return ret

    def _fit_full(self, X, n_components, session=None, run_kwargs=None):
        """Fit the model by computing full SVD on X"""
        n_samples, n_features = X.shape

        if n_components == 'mle':
            if n_samples < n_features:
                raise ValueError("n_components='mle' is only supported "
                                 "if n_samples >= n_features")
        elif not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError("n_components=%r must be between 0 and "
                             "min(n_samples, n_features)=%r with "
                             "svd_solver='full'"
                             % (n_components, min(n_samples, n_features)))
        elif n_components >= 1:
            if not isinstance(n_components, (numbers.Integral, np.integer)):
                raise ValueError("n_components=%r must be of type int "
                                 "when greater than or equal to 1, "
                                 "was of type=%r"
                                 % (n_components, type(n_components)))

        # Center data
        self.mean_ = mt.mean(X, axis=0)
        X -= self.mean_

        U, S, V = mt.linalg.svd(X)
        # flip eigenvectors' sign to enforce deterministic output
        U, V = svd_flip(U, V)

        components_ = V

        # Get variance explained by singular values
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = S.copy()  # Store the singular values.

        # Postprocess the number of components required
        if n_components == 'mle':
            n_components = mr.spawn(_infer_dimension, args=(explained_variance_, n_samples))
            ExecutableTuple([n_components, U, V]).execute(session=session, **(run_kwargs or dict()))
            n_components = n_components.fetch()
        elif 0 < n_components < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            # ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            ratio_cumsum = explained_variance_ratio_.cumsum()
            n_components = (mt.searchsorted(ratio_cumsum, n_components) + 1)\
                .to_numpy(session=session, **(run_kwargs or dict()))

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = \
            explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]

        return U, S, V

    def _fit_truncated(self, X, n_components, svd_solver):
        """Fit the model by computing truncated SVD (by ARPACK or randomized)
        on X
        """
        n_samples, n_features = X.shape

        if isinstance(n_components, str):
            raise ValueError("n_components=%r cannot be a string "
                             "with svd_solver='%s'"
                             % (n_components, svd_solver))
        elif not 1 <= n_components <= min(n_samples, n_features):
            raise ValueError("n_components=%r must be between 1 and "
                             "min(n_samples, n_features)=%r with "
                             "svd_solver='%s'"
                             % (n_components, min(n_samples, n_features),
                                svd_solver))
        elif not isinstance(n_components, (numbers.Integral, np.integer)):
            raise ValueError("n_components=%r must be of type int "
                             "when greater than or equal to 1, was of type=%r"
                             % (n_components, type(n_components)))
        elif svd_solver == 'arpack' and n_components == min(n_samples,
                                                            n_features):
            raise ValueError("n_components=%r must be strictly less than "
                             "min(n_samples, n_features)=%r with "
                             "svd_solver='%s'"
                             % (n_components, min(n_samples, n_features),
                                svd_solver))

        random_state = check_random_state(self.random_state)

        # Center data
        self.mean_ = mt.mean(X, axis=0)
        X -= self.mean_

        if svd_solver == 'arpack':
            # # random init solution, as ARPACK does it internally
            # v0 = random_state.uniform(-1, 1, size=min(X.shape))
            # U, S, V = svds(X, k=n_components, tol=self.tol, v0=v0)
            # # svds doesn't abide by scipy.linalg.svd/randomized_svd
            # # conventions, so reverse its outputs.
            # S = S[::-1]
            # # flip eigenvectors' sign to enforce deterministic output
            # U, V = svd_flip(U[:, ::-1], V[::-1])
            raise NotImplementedError('Does not support arpack svd_resolver')

        elif svd_solver == 'randomized':
            # sign flipping is done inside
            U, S, V = randomized_svd(X, n_components=n_components,
                                     n_iter=self.iterated_power,
                                     flip_sign=True,
                                     random_state=random_state)

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = V
        self.n_components_ = n_components

        # Get variance explained by singular values
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = mt.var(X, ddof=1, axis=0)
        self.explained_variance_ratio_ = \
            self.explained_variance_ / total_var.sum()
        self.singular_values_ = S.copy()  # Store the singular values.

        if self.n_components_ < min(n_features, n_samples):
            self.noise_variance_ = (total_var.sum() -
                                    self.explained_variance_.sum())
            self.noise_variance_ /= min(n_features, n_samples) - n_components
        else:
            self.noise_variance_ = 0.

        return U, S, V

    def _score_samples(self, X, session=None):
        check_is_fitted(self, 'mean_')

        X = check_array(X)
        Xr = X - self.mean_
        n_features = X.shape[1]
        precision = self.get_precision().fetch(session=session)
        log_like = -.5 * (Xr * (mt.dot(Xr, precision))).sum(axis=1)
        log_like -= .5 * (n_features * log(2. * mt.pi) -
                          fast_logdet(precision))
        return log_like

    def score_samples(self, X, session=None):
        """Return the log-likelihood of each sample.

        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf

        Parameters
        ----------
        X : tensor, shape(n_samples, n_features)
            The data.

        Returns
        -------
        ll : tensor, shape (n_samples,)
            Log-likelihood of each sample under the current model
        """
        log_like = self._score_samples(X, session=session)
        log_like.execute(session=session)
        return log_like

    def score(self, X, y=None, session=None):
        """Return the average log-likelihood of all samples.

        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf

        Parameters
        ----------
        X : tensor, shape(n_samples, n_features)
            The data.

        y : Ignored

        Returns
        -------
        ll : float
            Average log-likelihood of the samples under the current model
        """
        ret = mt.mean(self._score_samples(X))
        ret.execute(session=session)
        return ret
