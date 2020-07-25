# cython: profile=False, boundscheck=False, wraparound=False, cdivision=True
# Profiling is enabled by default as the overhead does not seem to be
# measurable on this specific use case.

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#
# License: BSD 3 clause

# TODO: We still need to use ndarrays instead of typed memoryviews when using
# fused types and when the array may be read-only (for instance when it's
# provided by the user). This is fixed in cython > 0.3.

import numpy as np
cimport numpy as np
cimport cython
from cython cimport floating
from libc.math cimport sqrt

try:
    from sklearn.utils.extmath import row_norms
except ImportError:  # pragma: no cover
    row_norms = None


np.import_array()


ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT


cdef floating _euclidean_dense_dense(
        floating* a,  # IN
        floating* b,  # IN
        int n_features,
        bint squared) nogil:
    """Euclidean distance between a dense and b dense"""
    cdef:
        int i
        int n = n_features // 4
        int rem = n_features % 4
        floating result = 0

    # We manually unroll the loop for better cache optimization.
    for i in range(n):
        result += ((a[0] - b[0]) * (a[0] - b[0])
                  +(a[1] - b[1]) * (a[1] - b[1])
                  +(a[2] - b[2]) * (a[2] - b[2])
                  +(a[3] - b[3]) * (a[3] - b[3]))
        a += 4; b += 4

    for i in range(rem):
        result += (a[i] - b[i]) * (a[i] - b[i])

    return result if squared else sqrt(result)


cdef floating _euclidean_sparse_dense(
        floating[::1] a_data,  # IN
        int[::1] a_indices,    # IN
        floating[::1] b,       # IN
        floating b_squared_norm,
        bint squared) nogil:
    """Euclidean distance between a sparse and b dense"""
    cdef:
        int nnz = a_indices.shape[0]
        int i
        floating tmp, bi
        floating result = 0.0

    for i in range(nnz):
        bi = b[a_indices[i]]
        tmp = a_data[i] - bi
        result += tmp * tmp - bi * bi

    result += b_squared_norm

    if result < 0: result = 0.0

    return result if squared else sqrt(result)


cpdef floating _inertia_dense(
        np.ndarray[floating, ndim=2, mode='c'] X,  # IN
        floating[::1] sample_weight,               # IN
        floating[:, ::1] centers,                  # IN
        int[::1] labels):                          # IN
    """Compute inertia for dense input data

    Sum of squared distance between each sample and its assigned center.
    """
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int i, j

        floating sq_dist = 0.0
        floating inertia = 0.0

    for i in range(n_samples):
        j = labels[i]
        sq_dist = _euclidean_dense_dense(&X[i, 0], &centers[j, 0],
                                         n_features, True)
        inertia += sq_dist * sample_weight[i]

    return inertia


cpdef floating _inertia_sparse(
        X,                            # IN
        floating[::1] sample_weight,  # IN
        floating[:, ::1] centers,     # IN
        int[::1] labels):             # IN
    """Compute inertia for sparse input data

    Sum of squared distance between each sample and its assigned center.
    """
    cdef:
        floating[::1] X_data = X.data
        int[::1] X_indices = X.indices
        int[::1] X_indptr = X.indptr

        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int i, j

        floating sq_dist = 0.0
        floating inertia = 0.0

        floating[::1] centers_squared_norms = row_norms(centers, squared=True)

    for i in range(n_samples):
        j = labels[i]
        sq_dist = _euclidean_sparse_dense(
            X_data[X_indptr[i]: X_indptr[i + 1]],
            X_indices[X_indptr[i]: X_indptr[i + 1]],
            centers[j], centers_squared_norms[j], True)
        inertia += sq_dist * sample_weight[i]

    return inertia


cdef void _average_centers(
        floating[:, ::1] centers,           # INOUT
        floating[::1] weight_in_clusters):  # IN
    """Average new centers wrt weights."""
    cdef:
        int n_clusters = centers.shape[0]
        int n_features = centers.shape[1]
        int j, k
        floating alpha

    for j in range(n_clusters):
        if weight_in_clusters[j] > 0:
            alpha = 1.0 / weight_in_clusters[j]
            for k in range(n_features):
                centers[j, k] *= alpha


cdef void _center_shift(
        floating[:, ::1] centers_old,  # IN
        floating[:, ::1] centers_new,  # IN
        floating[::1] center_shift):   # OUT
    """Compute shift between old and new centers."""
    cdef:
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]
        int j

    for j in range(n_clusters):
        center_shift[j] = _euclidean_dense_dense(
            &centers_new[j, 0], &centers_old[j, 0], n_features, False)


def update_center(
        floating[:, ::1] centers_old,       # IN
        floating[:, ::1] centers_new,       # INOUT
        floating[::1] center_shift,         # OUT
        floating[::1] weight_in_clusters):  # IN
    _average_centers(centers_new, weight_in_clusters)
    _center_shift(centers_old, centers_new, center_shift)


def merge_update_chunks(int n_clusters,
                        int n_features,
                        floating[::1] weight_in_clusters,
                        floating[::1] weight_in_clusters_chunk,
                        floating[:, ::1] centers_new,
                        floating[:, ::1] centers_new_chunk):
    for j in range(n_clusters):
        weight_in_clusters[j] += weight_in_clusters_chunk[j]
        for k in range(n_features):
            centers_new[j, k] += centers_new_chunk[j, k]


def update_upper_lower_bounds(
        floating[::1] upper_bounds,                # INOUT
        floating[:, ::1] lower_bounds,             # INOUT
        int[::1] labels,                           # IN
        floating[::1] center_shift):               # IN
    cdef:
        int n_samples = upper_bounds.shape[0]
        int n_clusters = lower_bounds.shape[1]

    for i in range(n_samples):
        upper_bounds[i] += center_shift[labels[i]]

        for j in range(n_clusters):
            lower_bounds[i, j] -= center_shift[j]
            if lower_bounds[i, j] < 0:
                lower_bounds[i, j] = 0
