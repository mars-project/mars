# cython: profile=False, boundscheck=False, wraparound=False, cdivision=True
#
# Author: Andreas Mueller
#
# Licence: BSD 3 clause

# TODO: We still need to use ndarrays instead of typed memoryviews when using
# fused types and when the array may be read-only (for instance when it's
# provided by the user). This is fixed in cython > 0.3.

import numpy as np
cimport numpy as np
cimport cython
from cython cimport floating
from sklearn.utils.extmath import row_norms

from ._k_means_fast cimport _euclidean_dense_dense
from ._k_means_fast cimport _euclidean_sparse_dense


np.import_array()


def init_bounds_dense(
        np.ndarray[floating, ndim=2, mode='c'] X,  # IN
        floating[:, ::1] centers,                  # IN
        floating[:, ::1] center_half_distances,    # IN
        int[::1] labels,                           # OUT
        floating[::1] upper_bounds,                # OUT
        floating[:, ::1] lower_bounds):            # OUT
    """Initialize upper and lower bounds for each sample for dense input data.

    Given X, centers and the pairwise distances divided by 2.0 between the
    centers this calculates the upper bounds and lower bounds for each sample.
    The upper bound for each sample is set to the distance between the sample
    and the closest center.

    The lower bound for each sample is a one-dimensional array of n_clusters.
    For each sample i assume that the previously assigned cluster is c1 and the
    previous closest distance is dist, for a new cluster c2, the
    lower_bound[i][c2] is set to distance between the sample and this new
    cluster, if and only if dist > center_half_distances[c1][c2]. This prevents
    computation of unnecessary distances for each sample to the clusters that
    it is unlikely to be assigned to.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=floating
        The input data.

    centers : ndarray of shape (n_clusters, n_features), dtype=floating
        The cluster centers.

    center_half_distances : ndarray of shape (n_clusters, n_clusters), \
            dtype=floating
        The half of the distance between any 2 clusters centers.

    labels : ndarray of shape(n_samples), dtype=int
        The label for each sample. This array is modified in place.

    upper_bounds : ndarray of shape(n_samples,), dtype=floating
        The upper bound on the distance between each sample and its closest
        cluster center. This array is modified in place.

    lower_bounds : ndarray, of shape(n_samples, n_clusters), dtype=floating
        The lower bound on the distance between each sample and each cluster
        center. This array is modified in place.
    """
    cdef:
        int n_samples = X.shape[0]
        int n_clusters = centers.shape[0]
        int n_features = X.shape[1]

        floating min_dist, dist
        int best_cluster, i, j

    for i in range(n_samples):
        best_cluster = 0
        min_dist = _euclidean_dense_dense(&X[i, 0], &centers[0, 0],
                                          n_features, False)
        lower_bounds[i, 0] = min_dist
        for j in range(1, n_clusters):
            if min_dist > center_half_distances[best_cluster, j]:
                dist = _euclidean_dense_dense(&X[i, 0], &centers[j, 0],
                                              n_features, False)
                lower_bounds[i, j] = dist
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j
        labels[i] = best_cluster
        upper_bounds[i] = min_dist


def init_bounds_sparse(
        X,                                       # IN
        floating[:, ::1] centers,                # IN
        floating[:, ::1] center_half_distances,  # IN
        int[::1] labels,                         # OUT
        floating[::1] upper_bounds,              # OUT
        floating[:, ::1] lower_bounds):          # OUT
    """Initialize upper and lower bounds for each sample for sparse input data.

    Given X, centers and the pairwise distances divided by 2.0 between the
    centers this calculates the upper bounds and lower bounds for each sample.
    The upper bound for each sample is set to the distance between the sample
    and the closest center.

    The lower bound for each sample is a one-dimensional array of n_clusters.
    For each sample i assume that the previously assigned cluster is c1 and the
    previous closest distance is dist, for a new cluster c2, the
    lower_bound[i][c2] is set to distance between the sample and this new
    cluster, if and only if dist > center_half_distances[c1][c2]. This prevents
    computation of unnecessary distances for each sample to the clusters that
    it is unlikely to be assigned to.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features), dtype=floating
        The input data. Must be in CSR format.

    centers : ndarray of shape (n_clusters, n_features), dtype=floating
        The cluster centers.

    center_half_distances : ndarray of shape (n_clusters, n_clusters), \
            dtype=floating
        The half of the distance between any 2 clusters centers.

    labels : ndarray of shape(n_samples), dtype=int
        The label for each sample. This array is modified in place.

    upper_bounds : ndarray of shape(n_samples,), dtype=floating
        The upper bound on the distance between each sample and its closest
        cluster center. This array is modified in place.

    lower_bounds : ndarray of shape(n_samples, n_clusters), dtype=floating
        The lower bound on the distance between each sample and each cluster
        center. This array is modified in place.
    """
    cdef:
        int n_samples = X.shape[0]
        int n_clusters = centers.shape[0]
        int n_features = X.shape[1]

        floating[::1] X_data = X.data
        int[::1] X_indices = X.indices
        int[::1] X_indptr = X.indptr

        floating min_dist, dist
        int best_cluster, i, j

        floating[::1] centers_squared_norms = row_norms(centers, squared=True)

    for i in range(n_samples):
        best_cluster = 0
        min_dist = _euclidean_sparse_dense(
            X_data[X_indptr[i]: X_indptr[i + 1]],
            X_indices[X_indptr[i]: X_indptr[i + 1]],
            centers[0], centers_squared_norms[0], False)

        lower_bounds[i, 0] = min_dist
        for j in range(1, n_clusters):
            if min_dist > center_half_distances[best_cluster, j]:
                dist = _euclidean_sparse_dense(
                    X_data[X_indptr[i]: X_indptr[i + 1]],
                    X_indices[X_indptr[i]: X_indptr[i + 1]],
                    centers[j], centers_squared_norms[j], False)
                lower_bounds[i, j] = dist
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j
        labels[i] = best_cluster
        upper_bounds[i] = min_dist


def update_chunk_dense(
        np.ndarray[floating, ndim=2, mode='c'] X,  # IN
        floating[::1] sample_weight,               # IN
        floating[:, ::1] centers_old,              # IN
        floating[:, ::1] center_half_distances,    # IN
        floating[::1] distance_next_center,        # IN
        int[::1] labels,                           # INOUT
        floating[::1] upper_bounds,                # INOUT
        floating[:, ::1] lower_bounds,             # INOUT
        floating[:, ::1] centers_new,              # OUT
        floating[::1] weight_in_clusters,          # OUT
        bint update_centers=True):
    return _update_chunk_dense(&X[0, 0], sample_weight, centers_old,
                               center_half_distances,
                               distance_next_center, labels,
                               upper_bounds, lower_bounds,
                               &centers_new[0, 0], &weight_in_clusters[0],
                               update_centers)


cdef void _update_chunk_dense(
        floating *X,                             # IN
        # expecting C aligned 2D array. XXX: Can be
        # replaced by const memoryview when cython min
        # version is >= 0.3
        floating[::1] sample_weight,             # IN
        floating[:, ::1] centers_old,            # IN
        floating[:, ::1] center_half_distances,  # IN
        floating[::1] distance_next_center,      # IN
        int[::1] labels,                         # INOUT
        floating[::1] upper_bounds,              # INOUT
        floating[:, ::1] lower_bounds,           # INOUT
        floating *centers_new,                   # OUT
        floating *weight_in_clusters,            # OUT
        bint update_centers) nogil:
    """K-means combined EM step for one dense data chunk.

    Compute the partial contribution of a single data chunk to the labels and
    centers.
    """
    cdef:
        int n_samples = labels.shape[0]
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]

        floating upper_bound, distance
        int i, j, k, label

    for i in range(n_samples):
        upper_bound = upper_bounds[i]
        bounds_tight = 0
        label = labels[i]

        # Next center is not far away from the currently assigned center.
        # Sample might need to be assigned to another center.
        if not distance_next_center[label] >= upper_bound:

            for j in range(n_clusters):

                # If this holds, then center_index is a good candidate for the
                # sample to be relabelled, and we need to confirm this by
                # recomputing the upper and lower bounds.
                if (j != label
                    and (upper_bound > lower_bounds[i, j])
                    and (upper_bound > center_half_distances[label, j])):

                    # Recompute upper bound by calculating the actual distance
                    # between the sample and its current assigned center.
                    if not bounds_tight:
                        upper_bound = _euclidean_dense_dense(
                            X + i * n_features, &centers_old[label, 0], n_features, False)
                        lower_bounds[i, label] = upper_bound
                        bounds_tight = 1

                    # If the condition still holds, then compute the actual
                    # distance between the sample and center. If this is less
                    # than the previous distance, reassign label.
                    if (upper_bound > lower_bounds[i, j]
                        or (upper_bound > center_half_distances[label, j])):

                        distance = _euclidean_dense_dense(
                            X + i * n_features, &centers_old[j, 0], n_features, False)
                        lower_bounds[i, j] = distance
                        if distance < upper_bound:
                            label = j
                            upper_bound = distance

            labels[i] = label
            upper_bounds[i] = upper_bound

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            for k in range(n_features):
                centers_new[label * n_features + k] += X[i * n_features + k] * sample_weight[i]


def update_chunk_sparse(
        X,                                         # IN
        floating[::1] sample_weight,               # IN
        floating[:, ::1] centers_old,              # IN
        floating[:, ::1] center_half_distances,    # IN
        floating[::1] distance_next_center,        # IN
        int[::1] labels,                           # INOUT
        floating[::1] upper_bounds,                # INOUT
        floating[:, ::1] lower_bounds,             # INOUT
        floating[:, ::1] centers_new,              # OUT
        floating[::1] weight_in_clusters,          # OUT
        bint update_centers=True):
    cdef:
        floating[::1] X_data = X.data
        int[::1] X_indices = X.indices
        int[::1] X_indptr = X.indptr

        floating[::1] centers_squared_norms = row_norms(centers_old, squared=True)

    return _update_chunk_sparse(
        X_data, X_indices, X_indptr, sample_weight, centers_old,
        centers_squared_norms, center_half_distances,
        distance_next_center, labels, upper_bounds, lower_bounds,
        &centers_new[0, 0], &weight_in_clusters[0], update_centers
    )


cdef void _update_chunk_sparse(
        floating[::1] X_data,                    # IN
        int[::1] X_indices,                      # IN
        int[::1] X_indptr,                       # IN
        floating[::1] sample_weight,             # IN
        floating[:, ::1] centers_old,            # IN
        floating[::1] centers_squared_norms,     # IN
        floating[:, ::1] center_half_distances,  # IN
        floating[::1] distance_next_center,      # IN
        int[::1] labels,                         # INOUT
        floating[::1] upper_bounds,              # INOUT
        floating[:, ::1] lower_bounds,           # INOUT
        floating *centers_new,                   # OUT
        floating *weight_in_clusters,            # OUT
        bint update_centers) nogil:
    """K-means combined EM step for one sparse data chunk.

    Compute the partial contribution of a single data chunk to the labels and
    centers.
    """
    cdef:
        int n_samples = labels.shape[0]
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]

        floating upper_bound, distance
        int i, j, k, label
        int s = X_indptr[0]

    for i in range(n_samples):
        upper_bound = upper_bounds[i]
        bounds_tight = 0
        label = labels[i]

        # Next center is not far away from the currently assigned center.
        # Sample might need to be assigned to another center.
        if not distance_next_center[label] >= upper_bound:

            for j in range(n_clusters):

                # If this holds, then center_index is a good candidate for the
                # sample to be relabelled, and we need to confirm this by
                # recomputing the upper and lower bounds.
                if (j != label
                    and (upper_bound > lower_bounds[i, j])
                    and (upper_bound > center_half_distances[label, j])):

                    # Recompute upper bound by calculating the actual distance
                    # between the sample and its current assigned center.
                    if not bounds_tight:
                        upper_bound = _euclidean_sparse_dense(
                            X_data[X_indptr[i] - s: X_indptr[i + 1] - s],
                            X_indices[X_indptr[i] - s: X_indptr[i + 1] - s],
                            centers_old[label], centers_squared_norms[label], False)
                        lower_bounds[i, label] = upper_bound
                        bounds_tight = 1

                    # If the condition still holds, then compute the actual
                    # distance between the sample and center. If this is less
                    # than the previous distance, reassign label.
                    if (upper_bound > lower_bounds[i, j]
                        or (upper_bound > center_half_distances[label, j])):
                        distance = _euclidean_sparse_dense(
                            X_data[X_indptr[i] - s: X_indptr[i + 1] - s],
                            X_indices[X_indptr[i] - s: X_indptr[i + 1] - s],
                            centers_old[j], centers_squared_norms[j], False)
                        lower_bounds[i, j] = distance
                        if distance < upper_bound:
                            label = j
                            upper_bound = distance

            labels[i] = label
            upper_bounds[i] = upper_bound

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            for k in range(X_indptr[i] - s, X_indptr[i + 1] - s):
                centers_new[label * n_features + X_indices[k]] += X_data[k] * sample_weight[i]
