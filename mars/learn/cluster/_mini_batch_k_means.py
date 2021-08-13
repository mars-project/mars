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

from ... import tensor as mt
from ...tensor.utils import check_random_state
from ..utils.extmath import row_norms
from ..utils.validation import _num_samples, check_array

from ._kmeans import KMeans
from ._kmeans import _validate_center_shape, _tolerance
from ._kmeans import _labels_inertia, _init_centroids
from ._kmeans import _check_normalize_sample_weight
from ._mini_batch_step import _mini_batch_step


def _mini_batch_convergence(batch_size, max_no_improvement, tol, n_samples,
                            centers_squared_diff, batch_inertia, context,
                            verbose=0, iteration_idx=None, n_iter=None,
                            session=None, run_kwargs=None):
    """Helper function to encapsulate the early stopping logic.

    Parameters
    ----------

    batch_size : int
        The batch size used in the last iteration, which is used to normalize
        inertia to be able to compare values when batch size changes.

    max_no_improvement : int
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.

    tol : float
        Control early stopping based on the relative center changes as
        measured by a smoothed, variance-normalized of the mean center
        squared position changes. This early stopping heuristics is
        closer to the one used for the batch variant of the algorithms
        but induces a slight computational and memory overhead over the
        inertia heuristic.

    n_samples : int
        The number of samples in the dataset.

    centers_squared_diff : float
        Squared distances between previous and updated cluster centers.

    batch_inertia : float
        Sum of squared distances of samples to their closest cluster center.

    context : dict
        Record information about previous iterations to determine whether to
        end the iteration prematurely.

    verbose : boolean, optional
        Verbosity mode.

    iteration_idx : int
        The number of iterations have used MiniBatch. Only work for the
        verbose mode to output the corresponding information.

    n_iter : int
        The total number of iterations using MiniBatch. Only work for the
        verbose mode to output the corresponding information.

    Returns
    -------
    convergence_flag : bool
        The flag of early stopping.
    """
    # Fetch data for computing directly because there is no Tensor
    centers_squared_diff = centers_squared_diff.fetch(session=session)
    batch_inertia = batch_inertia.fetch(session=session)

    # Normalize inertia to be able to compare values when batch size changes
    batch_inertia /= batch_size
    centers_squared_diff /= batch_size

    # Compute an Exponentially Weighted Average of the squared
    # diff to monitor the convergence while discarding
    # minibatch-local stochastic variability:
    # https://en.wikipedia.org/wiki/Moving_average
    ewa_diff = context.get('ewa_diff')
    ewa_inertia = context.get('ewa_inertia')
    if ewa_diff is None:
        ewa_diff = centers_squared_diff
        ewa_inertia = batch_inertia
    else:
        alpha = float(batch_size) * 2.0 / (n_samples + 1)
        alpha = 1.0 if alpha > 1.0 else alpha
        ewa_diff = ewa_diff * (1 - alpha) + centers_squared_diff * alpha
        ewa_inertia = ewa_inertia * (1 - alpha) + batch_inertia * alpha

    # Log progress to be able to monitor convergence
    if verbose:
        print(f"Minibatch iteration {iteration_idx + 1}/{n_iter}:"
              f"mean batch inertia:{batch_inertia}, ewa_inertia:{ewa_inertia}")

    # Early stopping based on absolute tolerance on squared change of
    # centers position (using EWA smoothing)
    if tol > 0.0 and ewa_diff <= tol:
        if verbose:
            print(f"Converged (small centers change at iteration"
                  f"{iteration_idx + 1}/{n_iter}")
        return True

    # Early stopping heuristic due to lack of improvement on smoothed inertia
    ewa_inertia_min = context.get('ewa_inertia_min')
    no_improvement = context.get('no_improvement', 0)
    if ewa_inertia_min is None or ewa_inertia < ewa_inertia_min:
        no_improvement = 0
        ewa_inertia_min = ewa_inertia
    else:
        no_improvement += 1

    if (max_no_improvement is not None
            and no_improvement >= max_no_improvement):
        if verbose:
            print(f"Converged (lack of improvement in inertia at iteration "
                  f"{iteration_idx + 1}/{n_iter}")
        return True

    # Update the convergence context to maintain state across successive calls:
    context['ewa_diff'] = ewa_diff
    context['ewa_inertia'] = ewa_inertia
    context['ewa_inertia_min'] = ewa_inertia_min
    context['no_improvement'] = no_improvement
    return False


class MiniBatchKMeans(KMeans):
    """Mini-Batch K-Means clustering

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

    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

    batch_size : int, default=100
        Size of the mini batches.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    tol : float, default=0.0
        Control early stopping based on the relative center changes as
        measured by a smoothed, variance-normalized of the mean center
        squared position changes. This early stopping heuristics is
        closer to the one used for the batch variant of the algorithms
        but induces a slight computational and memory overhead over the
        inertia heuristic.

        To disable convergence detection based on normalized center
        change, set tol to 0.0 (default).

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.

        To disable convergence detection based on inertia, set
        max_no_improvement to None.

    init_size : int, default=None
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than n_clusters.

        If `None`, `init_size= 3 * batch_size`.

    n_init : int, default=3
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the
        best of the ``n_init`` initializations as measured by inertia.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more easily reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.

    Attributes
    ----------

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : int
        Labels of each point (if compute_labels is set to True).

    inertia_ : float
        The value of the inertia criterion associated with the chosen
        partition (if compute_labels is set to True). The inertia is
        defined as the sum of square distances of samples to their nearest
        neighbor.

    n_iter_ : int
        Number of batches processed.

    counts_ : ndarray of shape (n_clusters,)
        Weigth sum of each cluster.

    init_size_ : int
        The effective number of samples used for the initialization.
    """

    def __init__(self, n_clusters=8, init='k-means||', max_iter=100,
                 batch_size=100, verbose=0, random_state=None, tol=0.0,
                 max_no_improvement=10, init_size=None, n_init=3,
                 reassignment_ratio=0.01):

        super().__init__(n_clusters=n_clusters, init=init, n_init=n_init,
                         max_iter=max_iter, tol=tol, verbose=verbose,
                         random_state=random_state)
        self.batch_size = batch_size
        self.max_no_improvement = max_no_improvement
        self.init_size = init_size
        self.reassignment_ratio = reassignment_ratio

    def _check_params(self, X):
        super()._check_params(X)

        # batch_size
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size should be > 0, got {self.batch_size} instead.")

        # max_no_improvement
        if self.max_no_improvement is not None and self.max_no_improvement < 0:
            raise ValueError(
                f"max_no_improvement should be >= 0, got "
                f"{self.max_no_improvement} instead.")

        # init_size
        if self.init_size is not None and self.init_size <= 0:
            raise ValueError(
                f"init_size should be >0, got {self.init_size} instead.")
        self._init_size = self.init_size
        if self._init_size is None:
            self._init_size = 3 * self.batch_size
            if self._init_size < self.n_clusters:
                self._init_size = 3 * self.n_clusters
        elif self._init_size < self.n_clusters:
            warnings.warn(
                f"init_size={self._init_size} should be larger than "
                f"n_clusters={self.n_clusters}. Setting it to"
                f"min(3 * n_clusters, n_samples)",
                RuntimeWarning, stacklevel=2)
            self._init_size = 3 * self.n_clusters
        self._init_size = min(self._init_size, X.shape[0])

        # reassignment_ratio
        if self.reassignment_ratio < 0:
            raise ValueError(
                f"reassignment_ratio should be >= 0, got "
                f"{self.reassignment_ratio} instead.")

    def fit(self, X, y=None, sample_weight=None, session=None, run_kwargs=None):
        """Compute mini batch k-means clustering.

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

        if np.isnan(_num_samples(X)):
            X.execute(session=session, **(run_kwargs or dict()))
        self._check_params(X)

        random_state = check_random_state(self.random_state).to_numpy()

        # Get a tolerance which is independent of the dataset
        tol = _tolerance(X, self.tol)
        tol = tol.execute(session=session).fetch(session=session)

        # sample_weight
        sample_weight = _check_normalize_sample_weight(sample_weight, X)

        # Validate init array
        init = self.init
        if hasattr(init, '__array__'):
            init = check_array(init, dtype=X.dtype.type, copy=True, order='C')
            _validate_center_shape(X, self.n_clusters, init)

        n_samples, n_features = X.shape

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        # init dataset
        init_indices = random_state.randint(0, n_samples, self._init_size)

        X_init_batch = X[init_indices]
        x_squared_norms_init_batch = x_squared_norms[init_indices]
        sample_weight_init_batch = sample_weight[init_indices]

        best_init_centers, best_init_inertia = None, None
        # Since each iteration of MiniBatch method is carried out on the batch
        # that partially sampled in the whole data set, there is no distance
        # information between all samples and the current center,
        # so elKAN method is not supported.
        # mini_batch_iter = _mini_batch_step
        mini_batch_iter = _mini_batch_step

        # mini-batch initialize centers
        for init_idx in range(self._n_init):
            # for init_idx in range(1):
            if self.verbose:
                print(f"Init {init_idx + 1}/{self._n_init} with method: {init}")

            weight_sums = mt.zeros(self.n_clusters, dtype=sample_weight.dtype)
            weight_sums.execute(session=session)

            # Initialize the centers using only a fraction of the data as we
            # expect n_samples to be very large when using MiniBatchKMeans
            centers_init = _init_centroids(
                X, self.n_clusters, init, random_state=random_state,
                x_squared_norms=x_squared_norms, init_size=self._init_size,
                oversampling_factor=self.oversampling_factor,
                init_iter=self.init_iter)

            # compute the label assignment on the init dataset
            centers_new, inertia, squared_diff = mini_batch_iter(
                X_init_batch, sample_weight_init_batch,
                x_squared_norms_init_batch, centers_init, self.n_clusters,
                compute_squared_diff=False, weight_sums=weight_sums,
                random_state=random_state, verbose=self.verbose,
                session=session, run_kwargs=run_kwargs)

            if self.verbose:
                inertia_data = inertia.fetch(session=session)
                print(f"Inertia for init {init_idx+1}/{self.n_init}: {inertia_data}")

            # Keep only the best cluster centers across independent inits on
            # the common validatoin set
            inertia.execute(session=session)
            inertia = inertia.fetch(session=session)
            if best_init_inertia is None or inertia < best_init_inertia:
                best_init_centers = centers_new
                best_init_inertia = inertia
                self._counts = weight_sums
        # Init End

        # Empty context to be used inplace by the convergence check routine
        convergence_context = {}

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        n_iter = int(self.max_iter * n_batches)

        centers = best_init_centers
        inertia = best_init_inertia
        # Perform the iterative optimization until the final convergence criterion
        for iteration_idx in range(n_iter):
            # Sample a minibatch from the full dataset
            batch_indices = random_state.randint(0, n_samples, self.batch_size)

            min_weight = self._counts.min()
            min_weight.execute(session=session, **(run_kwargs or dict()))
            min_weight = min_weight.fetch(session=session)

            # Perform the actual update step on the minibatch data
            centers_new, inertia_new, squared_diff = mini_batch_iter(
                X[batch_indices], sample_weight[batch_indices],
                x_squared_norms[batch_indices], centers, self.n_clusters,
                compute_squared_diff=tol > 0.0, weight_sums=self._counts,
                # Here we randomly choose whether to perform
                # random reassignment: the choice is done as a function
                # of the iteration index, and the minimum number of
                # counts, in order to force this reassignment to happen
                # every once in a while
                random_reassign=((iteration_idx + 1)
                                 % (10 + int(min_weight)) == 0),
                random_state=random_state,
                reassignment_ratio=self.reassignment_ratio,
                verbose=self.verbose, session=session, run_kwargs=run_kwargs)

            centers = centers_new

            # Monitor convergence and do early stopping if necessary
            if _mini_batch_convergence(
                    self.batch_size, self.max_no_improvement, tol, n_samples,
                    squared_diff, inertia_new, convergence_context,
                    verbose=self.verbose, iteration_idx=iteration_idx,
                    n_iter=n_iter, session=session, run_kwargs=run_kwargs):
                break

        labels, inertia = _labels_inertia(X, sample_weight, x_squared_norms, centers,
                                          session=session, run_kwargs=run_kwargs)

        to_runs = [centers, labels, inertia]
        mt.ExecutableTuple(to_runs).execute(session=session, **(run_kwargs or dict()))

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = iteration_idx
        return self


def mini_batch_k_means(X, n_clusters, sample_weight=None, init='k-means||',
                       n_init=10, max_iter=300, verbose=False, tol=1e-4, random_state=None,
                       batch_size=100, max_no_improvement=10, init_size=None,
                       reassignment_ratio=0.01, return_n_iter=False):
    """Mini-Batch K-Means clustering algorithm

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

    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

    batch_size : int, default=100
        Size of the mini batches.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    tol : float, default=0.0
        Control early stopping based on the relative center changes as
        measured by a smoothed, variance-normalized of the mean center
        squared position changes. This early stopping heuristics is
        closer to the one used for the batch variant of the algorithms
        but induces a slight computational and memory overhead over the
        inertia heuristic.

        To disable convergence detection based on normalized center
        change, set tol to 0.0 (default).

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.

        To disable convergence detection based on inertia, set
        max_no_improvement to None.

    init_size : int, default=None
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than n_clusters.

        If `None`, `init_size= 3 * batch_size`.

    n_init : int, default=3
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the
        best of the ``n_init`` initializations as measured by inertia.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more easily reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.

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
    est = MiniBatchKMeans(
        n_clusters=n_clusters, init=init, max_iter=max_iter, batch_size=batch_size,
        verbose=verbose, random_state=random_state, tol=tol,
        max_no_improvement=max_no_improvement, init_size=init_size,
        n_init=n_init, reassignment_ratio=reassignment_ratio
    ).fit(X, sample_weight=sample_weight)
    if return_n_iter:
        return est.cluster_centers_, est.labels_, est.inertia_, est.n_iter_
    else:
        return est.cluster_centers_, est.labels_, est.inertia_
