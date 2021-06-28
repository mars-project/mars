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

import warnings

import numpy as np
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.exceptions import ConvergenceWarning

from ... import tensor as mt
from ...tensor.utils import check_random_state
from ..base import BaseEstimator
from ..metrics.pairwise import euclidean_distances
from ..utils.extmath import row_norms
from ..utils.validation import _check_sample_weight, check_array, \
    _num_samples, check_is_fitted
from ._k_means_common import _inertia
from ._k_means_elkan_iter import init_bounds, elkan_iter
from ._k_means_init import _k_init, _scalable_k_init
from ._k_means_lloyd_iter import lloyd_iter


###############################################################################
# K-means batch estimation by EM (expectation maximization)

def _validate_center_shape(X, n_centers, centers):
    """Check if centers is compatible with X and n_centers"""
    if len(centers) != n_centers:
        raise ValueError('The shape of the initial centers (%s) '
                         'does not match the number of clusters %i'
                         % (centers.shape, n_centers))
    if centers.shape[1] != X.shape[1]:
        raise ValueError(
            "The number of features of the initial centers %s "
            "does not match the number of features of the data %s."
            % (centers.shape[1], X.shape[1]))


def _tolerance(X, tol):
    """Return a tolerance which is independent of the dataset"""
    variances = mt.var(X, axis=0)
    return mt.mean(variances) * tol


def _check_normalize_sample_weight(sample_weight, X):
    """Set sample_weight if None, and check for correct dtype"""

    sample_weight_was_none = sample_weight is None

    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
    if not sample_weight_was_none:
        # normalize the weights to sum up to n_samples
        # an array of 1 (i.e. samples_weight is None) is already normalized
        n_samples = len(sample_weight)
        scale = n_samples / sample_weight.sum()
        sample_weight *= scale
    return sample_weight


def k_means(X, n_clusters, sample_weight=None, init='k-means||',
            n_init=10, max_iter=300, verbose=False, tol=1e-4,
            random_state=None, copy_x=True,
            algorithm="auto", oversampling_factor=2,
            init_iter=5, return_n_iter=False):
    """K-means clustering algorithm.

    Parameters
    ----------
    X : Tensor, shape (n_samples, n_features)
        The observations to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory copy
        if the given data is not C-contiguous.

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    sample_weight : array-like, shape (n_samples,), optional
        The weights for each observation in X. If None, all observations
        are assigned equal weight (default: None)

    init : {'k-means++', 'k-means||', 'random', or tensor, or a callable}, optional
        Method for initialization, default to 'k-means||':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'k-means||': scalable k-means++.

        'random': choose k observations (rows) at random from data for
        the initial centroids.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.

    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : boolean, optional
        Verbosity mode.

    tol : float, optional
        The relative increment in the results before declaring convergence.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, optional
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True (default), then the original data is
        not modified, ensuring X is C-contiguous.  If False, the original data
        is modified, and put back before the function returns, but small
        numerical differences may be introduced by subtracting and then adding
        the data mean, in this case it will also not ensure that data is
        C-contiguous which may cause a significant slowdown.

    algorithm : "auto", "full" or "elkan", default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient by using the triangle
        inequality, but currently doesn't support sparse data. "auto" chooses
        "elkan" for dense data and "full" for sparse data.

    oversampling_factor: int, default=2
        Only work for kmeans||, used in each iteration in kmeans||.

    init_iter: int, default=5
        Only work for kmeans||, indicates how may iterations required.

    return_n_iter : bool, optional
        Whether or not to return the number of iterations.

    Returns
    -------
    centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.

    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    best_n_iter : int
        Number of iterations corresponding to the best results.
        Returned only if `return_n_iter` is set to True.
    """

    est = KMeans(
        n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter,
        verbose=verbose, tol=tol, random_state=random_state,
        copy_x=copy_x, algorithm=algorithm,
        oversampling_factor=oversampling_factor, init_iter=init_iter
    ).fit(X, sample_weight=sample_weight)
    if return_n_iter:
        return est.cluster_centers_, est.labels_, est.inertia_, est.n_iter_
    else:
        return est.cluster_centers_, est.labels_, est.inertia_


def _kmeans_single_elkan(X, sample_weight, centers_init, n_clusters, max_iter=300,
                         verbose=False, x_squared_norms=None,
                         tol=1e-4, X_mean=None, session=None, run_kwargs=None):
    sample_weight = _check_normalize_sample_weight(sample_weight, X)

    centers = centers_init
    # execute X, centers and tol first
    tol = mt.asarray(tol)
    to_run = [X, sample_weight, centers, x_squared_norms, tol]
    if X_mean is not None:
        to_run.append(X_mean)
    mt.ExecutableTuple(to_run).execute(session=session, **(run_kwargs or dict()))
    tol = tol.fetch(session=session)

    if verbose:
        print('Initialization complete')

    center_half_distances = euclidean_distances(centers) / 2
    distance_next_center = mt.partition(mt.asarray(center_half_distances),
                                        kth=1, axis=0)[1]
    center_shift = mt.zeros(n_clusters, dtype=X.dtype)

    labels, upper_bounds, lower_bounds = \
        init_bounds(X, centers, center_half_distances, n_clusters)

    for i in range(max_iter):
        to_runs = []

        centers_new, weight_in_clusters, upper_bounds, lower_bounds, labels, center_shift = \
            elkan_iter(X, sample_weight, centers, center_half_distances,
                       distance_next_center, upper_bounds, lower_bounds,
                       labels, center_shift, session=session, run_kwargs=run_kwargs)
        to_runs.extend([centers_new, weight_in_clusters, upper_bounds,
                        lower_bounds, labels, center_shift])

        # compute new pairwise distances between centers and closest other
        # center of each center for next iterations
        center_half_distances = euclidean_distances(centers_new) / 2
        distance_next_center = mt.partition(
            mt.asarray(center_half_distances), kth=1, axis=0)[1]
        to_runs.extend([center_half_distances, distance_next_center])

        if verbose:
            inertia = _inertia(X, sample_weight, centers, labels)
            to_runs.append(inertia)

        center_shift_tot = (center_shift ** 2).sum()
        to_runs.append(center_shift_tot)

        mt.ExecutableTuple(to_runs).execute(session=session, **(run_kwargs or dict()))

        if verbose:
            inertia_data = inertia.fetch(session=session)
            print(f"Iteration {i}, inertia {inertia_data}")

        center_shift_tot = center_shift_tot.fetch(session=session)
        if center_shift_tot <= tol:
            if verbose:  # pragma: no cover
                print(f"Converged at iteration {i}: center shift {center_shift_tot} "
                      f"within tolerance {tol}")
            break

        centers, centers_new = centers_new, centers

    if center_shift_tot > 0:
        # rerun E-step so that predicted labels match cluster centers
        centers_new, weight_in_clusters, upper_bounds, lower_bounds, labels, center_shift = \
            elkan_iter(X, sample_weight, centers, center_half_distances,
                       distance_next_center, upper_bounds, lower_bounds,
                       labels, center_shift, update_centers=False,
                       session=session, run_kwargs=run_kwargs)

    inertia = _inertia(X, sample_weight, centers, labels)

    mt.ExecutableTuple([labels, inertia, centers]).execute(
        session=session, **(run_kwargs or dict()))
    return labels, inertia, centers, i + 1


def _kmeans_single_lloyd(X, sample_weight, centers_init, n_clusters, max_iter=300,
                         verbose=False, x_squared_norms=None,
                         tol=1e-4, X_mean=None, session=None, run_kwargs=None):
    sample_weight = _check_normalize_sample_weight(sample_weight, X)

    centers = centers_init
    # execute X, centers and tol first
    tol = mt.asarray(tol)
    to_run = [X, centers, x_squared_norms, tol]
    if X_mean is not None:
        to_run.append(X_mean)
    mt.ExecutableTuple(to_run).execute(session=session, **(run_kwargs or dict()))
    tol = tol.fetch(session=session)

    if verbose:  # pragma: no cover
        print("Initialization complete")

    labels = mt.full(X.shape[0], -1, dtype=mt.int32)
    center_shift = mt.zeros(n_clusters, dtype=X.dtype)

    for i in range(max_iter):
        to_runs = []

        centers_new, weight_in_clusters, labels, center_shift = \
            lloyd_iter(X, sample_weight, x_squared_norms, centers,
                       labels, center_shift, update_centers=True,
                       session=session, run_kwargs=run_kwargs)
        to_runs.extend([centers_new, weight_in_clusters, labels, center_shift])

        if verbose:
            inertia = _inertia(X, sample_weight, centers, labels)
            to_runs.append(inertia)

        center_shift_tot = (center_shift ** 2).sum()
        to_runs.append(center_shift_tot)

        mt.ExecutableTuple(to_runs).execute(session=session, **(run_kwargs or dict()))

        if verbose:  # pragma: no cover
            inertia_data = inertia.fetch(session=session)
            print(f"Iteration {i}, inertia {inertia_data}")

        center_shift_tot = center_shift_tot.fetch(session=session)
        if center_shift_tot <= tol:
            if verbose:  # pragma: no cover
                print(f"Converged at iteration {i}: center shift {center_shift_tot} "
                      f"within tolerance {tol}")
            break

        centers, centers_new = centers_new, centers

    if center_shift_tot > 0:
        # rerun E-step so that predicted labels match cluster centers
        centers_new, weight_in_clusters, labels, center_shift = \
            lloyd_iter(X, sample_weight, x_squared_norms, centers,
                       labels, center_shift, update_centers=False,
                       session=session, run_kwargs=run_kwargs)

    inertia = _inertia(X, sample_weight, centers, labels)

    mt.ExecutableTuple([labels, inertia, centers]).execute(
        session=session, **(run_kwargs or dict()))
    return labels, inertia, centers, i + 1


def _labels_inertia(X, sample_weight, x_squared_norms, centers,
                    session=None, run_kwargs=None):
    """E step of the K-means EM algorithm.

    Compute the labels and the inertia of the given samples and centers.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples to assign to the labels. If sparse matrix, must be in
        CSR format.

    sample_weight : array-like of shape (n_samples,)
        The weights for each observation in X.

    x_squared_norms : Tensor of shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.

    centers : Tensor, shape (n_clusters, n_features)
        The cluster centers.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The resulting assignment

    inertia : float
        Sum of squared distances of samples to their closest cluster center.
    """
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]

    sample_weight = _check_normalize_sample_weight(sample_weight, X)
    labels = mt.full(n_samples, -1, dtype=np.int32)
    weight_in_clusters = mt.zeros(n_clusters, dtype=centers.dtype)
    center_shift = mt.zeros_like(weight_in_clusters)

    centers, weight_in_clusters, labels, center_shift = \
        lloyd_iter(X, sample_weight, x_squared_norms, centers, labels,
                   center_shift, update_centers=False,
                   session=session, run_kwargs=run_kwargs)

    inertia = _inertia(X, sample_weight, centers, labels)

    return labels, inertia


def _init_centroids(X, n_clusters=8, init="k-means++", random_state=None,
                    x_squared_norms=None, init_size=None,
                    oversampling_factor=2, init_iter=5):
    """Compute the initial centroids

    Parameters
    ----------

    X : Tensor of shape (n_samples, n_features)
        The input samples.

    n_clusters : int, default=8
        number of centroids.

    init : {'k-means++', 'k-means||', 'random', tensor, callable}, default="k-means++"
        Method for initialization.

    random_state : int, RandomState instance, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    x_squared_norms : tensor of shape (n_samples,), default=None
        Squared euclidean norm of each data point. Pass it if you have it at
        hands already to avoid it being recomputed here. Default: None

    init_size : int, default=None
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than k.

    Returns
    -------
    centers : tensor of shape(k, n_features)
    """
    random_state = check_random_state(random_state).to_numpy()
    n_samples = X.shape[0]

    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)

    if init_size is not None and init_size < n_samples:  # pragma: no cover
        if init_size < n_clusters:
            warnings.warn(
                f"init_size={init_size} should be larger than k={n_clusters}. "
                "Setting it to 3*k",
                RuntimeWarning, stacklevel=2)
            init_size = 3 * n_clusters
        init_indices = random_state.randint(0, n_samples, init_size)
        X = X[init_indices]
        x_squared_norms = x_squared_norms[init_indices]
        n_samples = X.shape[0]
    elif n_samples < n_clusters:
        raise ValueError(
            f"n_samples={n_samples} should be larger than n_clusters={n_clusters}")

    if isinstance(init, str) and init == 'k-means++':
        centers = _k_init(X, n_clusters, random_state=random_state,
                          x_squared_norms=x_squared_norms)
    elif isinstance(init, str) and init == 'k-means||':
        centers = _scalable_k_init(X, n_clusters, random_state=random_state,
                                   x_squared_norms=x_squared_norms,
                                   oversampling_factor=oversampling_factor,
                                   init_iter=init_iter)
    elif isinstance(init, str) and init == 'random':
        seeds = random_state.choice(n_samples, size=n_clusters, replace=False)
        centers = X[seeds].rechunk((n_clusters, X.shape[1]))
    elif hasattr(init, '__array__'):
        # ensure that the centers have the same dtype as X
        # this is a requirement of fused types of cython
        centers = mt.array(init, dtype=X.dtype)
    elif callable(init):
        centers = init(X, n_clusters, random_state=random_state)
        centers = mt.asarray(centers, dtype=X.dtype)
    else:  # pragma: no cover
        raise ValueError("the init parameter for the k-means should "
                         "be 'k-means++' or 'random' or a tensor, "
                         f"'{init}' (type '{type(init)}') was passed.")

    if centers.issparse():
        centers = centers.todense()

    _validate_center_shape(X, n_clusters, centers)
    return centers


class KMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """K-Means clustering.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'k-means||', 'random'} or tensor of shape \
            (n_clusters, n_features), default='k-means||'
        Method for initialization, defaults to 'k-means||':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'k-means||': scalable k-means++.

        'random': choose k observations (rows) at random from data for
        the initial centroids.

        If a tensor is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    n_init : int, default=1
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True (default), then the original data is
        not modified, ensuring X is C-contiguous.  If False, the original data
        is modified, and put back before the function returns, but small
        numerical differences may be introduced by subtracting and then adding
        the data mean, in this case it will also not ensure that data is
        C-contiguous which may cause a significant slowdown.

    algorithm : {"auto", "full", "elkan"}, default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient by using the triangle
        inequality, but currently doesn't support sparse data. "auto" chooses
        "elkan" for dense data and "full" for sparse data.

    oversampling_factor: int, default=2
        Only work for kmeans||, used in each iteration in kmeans||.

    init_iter: int, default=5
        Only work for kmeans||, indicates how may iterations required.

    Attributes
    ----------
    cluster_centers_ : tensor of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : tensor of shape (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    n_iter_ : int
        Number of iterations run.

    See Also
    --------

    MiniBatchKMeans
        Alternative online implementation that does incremental updates
        of the centers positions using mini-batches.
        For large scale learning (say n_samples > 10k) MiniBatchKMeans is
        probably much faster than the default batch implementation.

    Notes
    -----
    The k-means problem is solved using either Lloyd's or Elkan's algorithm.

    The average complexity is given by O(k n T), were n is the number of
    samples and T is the number of iteration.

    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features. (D. Arthur and S. Vassilvitskii,
    'How slow is the k-means method?' SoCG2006)

    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.

    If the algorithm stops before fully converging (because of ``tol`` or
    ``max_iter``), ``labels_`` and ``cluster_centers_`` will not be consistent,
    i.e. the ``cluster_centers_`` will not be the means of the points in each
    cluster. Also, the estimator will reassign ``labels_`` after the last
    iteration to make ``labels_`` consistent with ``predict`` on the training
    set.

    Examples
    --------

    >>> from mars.learn.cluster import KMeans
    >>> import mars.tensor as mt
    >>> X = mt.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0, init='k-means++').fit(X)
    >>> kmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])
    """
    def __init__(self, n_clusters=8, init='k-means||', n_init=1,
                 max_iter=300, tol=1e-4, verbose=0, random_state=None,
                 copy_x=True, algorithm='auto', oversampling_factor=2,
                 init_iter=5):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm
        self.oversampling_factor = oversampling_factor
        self.init_iter = init_iter

    def _check_params(self, X):
        # n_init
        if self.n_init <= 0:
            raise ValueError(
                f"n_init should be > 0, got {self.n_init} instead.")
        self._n_init = self.n_init

        # max_iter
        if self.max_iter <= 0:
            raise ValueError(
                f"max_iter should be > 0, got {self.max_iter} instead.")

        # n_clusters
        if X.shape[0] < self.n_clusters:
            raise ValueError(f"n_samples={X.shape[0]} should be >= "
                             f"n_clusters={self.n_clusters}.")

        # tol
        self._tol = _tolerance(X, self.tol)

        # algorithm
        if self.algorithm not in ("auto", "full", "elkan"):
            raise ValueError(f"Algorithm must be 'auto', 'full' or 'elkan', "
                             f"got {self.algorithm} instead.")

        self._algorithm = self.algorithm
        if self._algorithm == "auto":
            # note(xuye.qin):
            # Different from scikit-learn,
            # for now, full seems more efficient when data is large,
            # elkan needs to be tuned more
            # old: algorithm = "full" if self.n_clusters == 1 else "elkan"
            self._algorithm = "full"
        if self._algorithm == "elkan" and self.n_clusters == 1:
            warnings.warn("algorithm='elkan' doesn't make sense for a single "
                          "cluster. Using 'full' instead.", RuntimeWarning)
            self._algorithm = "full"

        # init
        if not (hasattr(self.init, '__array__') or callable(self.init)
                or (isinstance(self.init, str)
                    and self.init in ["k-means++", "k-means||", "random"])):
            raise ValueError(
                f"init should be either 'k-means++'， 'k-mean||', 'random', "
                f"a tensor, a ndarray or a "
                f"callable, got '{self.init}' instead.")

        if hasattr(self.init, '__array__') and self._n_init != 1:
            warnings.warn(
                f"Explicit initial center position passed: performing only"
                f" one init in {self.__class__.__name__} instead of "
                f"n_init={self._n_init}.", RuntimeWarning, stacklevel=2)
            self._n_init = 1

    def _check_test_data(self, X):
        X = check_array(X, accept_sparse=True, dtype=[np.float64, np.float32],
                        order='C', accept_large_sparse=False)
        n_samples, n_features = X.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:  # pragma: no cover
            raise ValueError(f"Incorrect number of features. Got {n_features} features, "
                             f"expected {expected_n_features}")

        return X

    def fit(self, X, y=None, sample_weight=None, session=None, run_kwargs=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self
            Fitted estimator.
        """
        expect_chunk_size_on_columns = mt.tensor(X).shape[1]
        if not np.isnan(expect_chunk_size_on_columns):
            X = mt.tensor(X, chunk_size={1: expect_chunk_size_on_columns})

        X = self._validate_data(X, accept_sparse=True,
                                dtype=[np.float64, np.float32],
                                order='C', copy=self.copy_x,
                                accept_large_sparse=False)
        # verify that the number of samples given is larger than k
        if np.isnan(_num_samples(X)):  # pragma: no cover
            X.execute(session=session, **(run_kwargs or dict()))

        self._check_params(X)
        random_state = check_random_state(self.random_state).to_numpy()

        tol = _tolerance(X, self.tol)

        # Validate init array
        init = self.init
        if hasattr(init, '__array__'):
            init = check_array(init, dtype=X.dtype.type, copy=True, order='C')
            _validate_center_shape(X, self.n_clusters, init)

        # subtract of mean of x for more accurate distance computations
        X_mean = None
        if not X.issparse():
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if hasattr(init, '__array__'):
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        best_labels, best_inertia, best_centers = None, None, None

        if self._algorithm == "full":
            kmeans_single = _kmeans_single_lloyd
        else:
            kmeans_single = _kmeans_single_elkan

        for i in range(self._n_init):  # pylint: disable=unused-variable
            # Initialize centers
            centers_init = _init_centroids(
                X, self.n_clusters, init, random_state=random_state,
                x_squared_norms=x_squared_norms,
                oversampling_factor=self.oversampling_factor,
                init_iter=self.init_iter)

            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_single(
                X, sample_weight, centers_init, self.n_clusters,
                max_iter=self.max_iter, verbose=self.verbose, tol=tol,
                x_squared_norms=x_squared_norms, X_mean=X_mean,
                session=session, run_kwargs=run_kwargs)
            inertia = inertia.fetch(session=session)
            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        if not X.issparse():
            if not self.copy_x:  # pragma: no cover
                X += X_mean
            best_centers += X_mean
            best_centers.execute(session=session, **(run_kwargs or dict()))

        distinct_clusters = len(set(best_labels.fetch(session=session)))
        if distinct_clusters < self.n_clusters:  # pragma: no cover
            warnings.warn(
                f"Number of distinct clusters ({distinct_clusters}) found smaller than "
                f"n_clusters ({self.n_clusters}). Possibly due to duplicate points in X.",
                ConvergenceWarning, stacklevel=2)

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def fit_predict(self, X, y=None, sample_weight=None, session=None, run_kwargs=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, sample_weight=sample_weight,
                        session=session, run_kwargs=run_kwargs).labels_

    def fit_transform(self, X, y=None, sample_weight=None,
                      session=None, run_kwargs=None):
        """Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        X_new : array of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        # Currently, this just skips a copy of the data if it is not in
        # np.array or CSR format already.
        # XXX This skips _check_test_data, which may change the dtype;
        # we should refactor the input validation.
        self.fit(X, sample_weight=sample_weight,
                 session=session, run_kwargs=run_kwargs)
        return self._transform(X, session=session, run_kwargs=run_kwargs)

    def transform(self, X, session=None, run_kwargs=None):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : tensor of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        return self._transform(X, session=session, run_kwargs=run_kwargs)

    def _transform(self, X, session=None, run_kwargs=None):
        """guts of transform method; no input validation"""
        return euclidean_distances(X, self.cluster_centers_).execute(
            session=session, **(run_kwargs or dict()))

    def predict(self, X, sample_weight=None, session=None, run_kwargs=None):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : tensor of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)

        result = _labels_inertia(X, sample_weight, x_squared_norms,
                                 self.cluster_centers_, session=session,
                                 run_kwargs=run_kwargs)[0]
        result.execute(session=session, *(run_kwargs or dict()))
        return result

    def score(self, X, y=None, sample_weight=None, session=None, run_kwargs=None):
        """Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)

        result = -_labels_inertia(X, sample_weight, x_squared_norms,
                                  self.cluster_centers_, session=session,
                                  run_kwargs=run_kwargs)[1]
        result.execute(session=session, **(run_kwargs or dict()))
        return result
