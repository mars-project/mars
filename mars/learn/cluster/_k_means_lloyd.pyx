# cython: profile=False, boundscheck=False, wraparound=False, cdivision=True
#
# Licence: BSD 3 clause

# TODO: We still need to use ndarrays instead of typed memoryviews when using
# fused types and when the array may be read-only (for instance when it's
# provided by the user). This is fixed in cython > 0.3.

import numpy as np
cimport numpy as np
from cython cimport floating
from libc.stdlib cimport malloc, free
from libc.float cimport DBL_MAX, FLT_MAX

from ..utils._cython_blas cimport _gemm
from ..utils._cython_blas cimport RowMajor, Trans, NoTrans


np.import_array()


def update_chunk_dense(
        np.ndarray[floating, ndim=2, mode='c'] X,  # IN
        floating[::1] sample_weight,               # IN
        floating[::1] x_squared_norms,             # IN
        floating[:, ::1] centers_old,              # IN
        floating[::1] centers_squared_norms,       # IN
        int[::1] labels,                           # OUT
        floating[:, ::1] centers_new,              # OUT
        floating[::1] weight_in_clusters,          # OUT
        bint update_centers=True):
    cdef:
        int n_samples = X.shape[0]
        int n_clusters = centers_old.shape[0]
        floating *pairwise_distances

    pairwise_distances = <floating*> malloc(n_samples * n_clusters * sizeof(floating))
    result = _update_chunk_dense(&X[0, 0], sample_weight, x_squared_norms,
                                 centers_old, centers_squared_norms,
                                 labels, &centers_new[0, 0],
                                 &weight_in_clusters[0], pairwise_distances,
                                 update_centers)
    free(pairwise_distances)
    return result


cdef void _update_chunk_dense(
        floating *X,                          # IN
        # expecting C aligned 2D array. XXX: Can be
        # replaced by const memoryview when cython min
        # version is >= 0.3
        floating[::1] sample_weight,          # IN
        floating[::1] x_squared_norms,        # IN
        floating[:, ::1] centers_old,         # IN
        floating[::1] centers_squared_norms,  # IN
        int[::1] labels,                      # OUT
        floating *centers_new,                # OUT
        floating *weight_in_clusters,         # OUT
        floating *pairwise_distances,         # OUT
        bint update_centers) nogil:
    """K-means combined EM step for one dense data chunk.

    Compute the partial contribution of a single data chunk to the labels and
    centers.
    """
    cdef:
        int n_samples = labels.shape[0]
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]

        floating sq_dist, min_sq_dist
        int i, j, k, label

    # Instead of computing the full pairwise squared distances matrix,
    # ||X - C||² = ||X||² - 2 X.C^T + ||C||², we only need to store
    # the - 2 X.C^T + ||C||² term since the argmin for a given sample only
    # depends on the centers.
    # pairwise_distances = ||C||²
    for i in range(n_samples):
        for j in range(n_clusters):
            pairwise_distances[i * n_clusters + j] = centers_squared_norms[j]

    # pairwise_distances += -2 * X.dot(C.T)
    _gemm(RowMajor, NoTrans, Trans, n_samples, n_clusters, n_features,
          -2.0, X, n_features, &centers_old[0, 0], n_features,
          1.0, pairwise_distances, n_clusters)

    for i in range(n_samples):
        min_sq_dist = pairwise_distances[i * n_clusters]
        label = 0
        for j in range(1, n_clusters):
            sq_dist = pairwise_distances[i * n_clusters + j]
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j
        labels[i] = label

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            for k in range(n_features):
                centers_new[label * n_features + k] += X[i * n_features + k] * sample_weight[i]


def update_chunk_sparse(
        X,                                         # IN
        floating[::1] sample_weight,               # IN
        floating[::1] x_squared_norms,             # IN
        floating[:, ::1] centers_old,              # IN
        floating[::1] centers_squared_norms,       # IN
        int[::1] labels,                           # OUT
        floating[:, ::1] centers_new,              # OUT
        floating[::1] weight_in_clusters,          # OUT
        bint update_centers=True):
    cdef:
        floating[::1] X_data = X.data
        int[::1] X_indices = X.indices
        int[::1] X_indptr = X.indptr

    X_data = X.data
    X_indices = X.indices
    X_indptr = X.indptr

    return _update_chunk_sparse(
        X_data, X_indices, X_indptr, sample_weight,
        x_squared_norms, centers_old, centers_squared_norms,
        labels, &centers_new[0, 0], &weight_in_clusters[0],
        update_centers
    )


cdef void _update_chunk_sparse(
        floating[::1] X_data,                 # IN
        int[::1] X_indices,                   # IN
        int[::1] X_indptr,                    # IN
        floating[::1] sample_weight,          # IN
        floating[::1] x_squared_norms,        # IN
        floating[:, ::1] centers_old,         # IN
        floating[::1] centers_squared_norms,  # IN
        int[::1] labels,                      # OUT
        floating *centers_new,                # OUT
        floating *weight_in_clusters,         # OUT
        bint update_centers) nogil:
    """K-means combined EM step for one sparse data chunk.

    Compute the partial contribution of a single data chunk to the labels and
    centers.
    """
    cdef:
        int n_samples = labels.shape[0]
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]

        floating sq_dist, min_sq_dist
        int i, j, k, label
        floating max_floating = FLT_MAX if floating is float else DBL_MAX
        int s = X_indptr[0]

    # XXX Precompute the pairwise distances matrix is not worth for sparse
    # currently. Should be tested when BLAS (sparse x dense) matrix
    # multiplication is available.
    for i in range(n_samples):
        min_sq_dist = max_floating
        label = 0

        for j in range(n_clusters):
            sq_dist = 0.0
            for k in range(X_indptr[i] - s, X_indptr[i + 1] - s):
                sq_dist += centers_old[j, X_indices[k]] * X_data[k]

            # Instead of computing the full squared distance with each cluster,
            # ||X - C||² = ||X||² - 2 X.C^T + ||C||², we only need to compute
            # the - 2 X.C^T + ||C||² term since the argmin for a given sample
            # only depends on the centers C.
            sq_dist = centers_squared_norms[j] -2 * sq_dist
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j

        labels[i] = label

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            for k in range(X_indptr[i] - s, X_indptr[i + 1] - s):
                centers_new[label * n_features + X_indices[k]] += X_data[k] * sample_weight[i]
