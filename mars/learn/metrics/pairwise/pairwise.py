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

import warnings
from functools import partial

try:
    from sklearn.exceptions import DataConversionWarning
except ImportError:  # pragma: no cover
    DataConversionWarning = None

from ....tensor.spatial import distance
from ...utils.validation import check_non_negative
from .core import PairwiseDistances
from .manhattan import manhattan_distances
from .cosine import cosine_distances
from .euclidean import euclidean_distances
from .haversine import haversine_distances


_VALID_METRICS = ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock',
                  'braycurtis', 'canberra', 'chebyshev', 'correlation',
                  'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski',
                  'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
                  'russellrao', 'seuclidean', 'sokalmichener',
                  'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski',
                  'haversine']

# Helper functions - distance
PAIRWISE_DISTANCE_FUNCTIONS = {
    # If updating this dictionary, update the doc in both distance_metrics()
    # and also in pairwise_distances()!
    'cityblock': manhattan_distances,
    'cosine': cosine_distances,
    'euclidean': euclidean_distances,
    'haversine': haversine_distances,
    'l2': euclidean_distances,
    'l1': manhattan_distances,
    'manhattan': manhattan_distances,
    'precomputed': None,  # HACK: precomputed is always allowed, never called
}

# These distances recquire boolean tensors, when using mars.tensor.spatial.distance
PAIRWISE_BOOLEAN_FUNCTIONS = [
    'dice',
    'jaccard',
    'kulsinski',
    'matching',
    'rogerstanimoto',
    'russellrao',
    'sokalmichener',
    'sokalsneath',
    'yule',
]


def pairwise_distances(X, Y=None, metric="euclidean", **kwds):
    if (metric not in _VALID_METRICS and
            not callable(metric) and metric != "precomputed"):
        raise ValueError("Unknown metric %s. "
                         "Valid metrics are %s, or 'precomputed', or a "
                         "callable" % (metric, _VALID_METRICS))

    if metric == "precomputed":
        X, _ = PairwiseDistances.check_pairwise_arrays(X, Y, precomputed=True)

        whom = ("`pairwise_distances`. Precomputed distance "
                " need to have non-negative values.")
        X = check_non_negative(X, whom=whom)
        return X
    elif metric in PAIRWISE_DISTANCE_FUNCTIONS:
        func = PAIRWISE_DISTANCE_FUNCTIONS[metric]
    else:
        # including when metric is callable
        dtype = bool if metric in PAIRWISE_BOOLEAN_FUNCTIONS else None

        if (dtype == bool and
                (X.dtype != bool or (Y is not None and Y.dtype != bool)) and
                DataConversionWarning is not None):
            msg = "Data was converted to boolean for metric %s" % metric
            warnings.warn(msg, DataConversionWarning)

        X, Y = PairwiseDistances.check_pairwise_arrays(X, Y, dtype=dtype)
        if X is Y:
            return distance.squareform(distance.pdist(X, metric=metric, **kwds))
        func = partial(distance.cdist, metric=metric, **kwds)

    return func(X, Y, **kwds)
