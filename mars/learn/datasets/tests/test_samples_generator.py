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

from functools import partial
from collections import defaultdict

import numpy as np
from sklearn.utils._testing import (
    assert_array_almost_equal,
    assert_raises,
    assert_almost_equal,
    assert_raise_message,
)

from .... import tensor as mt
from ....tensor.linalg import svd
from ..samples_generator import (
    make_low_rank_matrix,
    make_classification,
    make_regression,
    make_blobs,
)


def test_make_classification(setup):
    weights = [0.1, 0.25]
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=5,
        n_redundant=1,
        n_repeated=1,
        n_classes=3,
        n_clusters_per_class=1,
        hypercube=False,
        shift=None,
        scale=None,
        weights=weights,
        random_state=0,
        flip_y=-1,
    )

    assert weights == [0.1, 0.25]
    assert X.shape == (100, 20)
    assert y.shape == (100,)
    assert mt.unique(y).to_numpy().shape == (3,)
    assert (y == 0).sum().to_numpy() == 10
    assert (y == 1).sum().to_numpy() == 25
    assert (y == 2).sum().to_numpy() == 65

    # Test for n_features > 30
    X, y = make_classification(
        n_samples=2000,
        n_features=31,
        n_informative=31,
        n_redundant=0,
        n_repeated=0,
        hypercube=True,
        scale=0.5,
        random_state=0,
    )

    X = X.to_numpy()
    assert X.shape == (2000, 31)
    assert y.shape == (2000,)
    assert (
        np.unique(X.view([("", X.dtype)] * X.shape[1]))
        .view(X.dtype)
        .reshape(-1, X.shape[1])
        .shape[0]
        == 2000
    )


def test_make_classification_informative_features(setup):
    """Test the construction of informative features in make_classification

    Also tests `n_clusters_per_class`, `n_classes`, `hypercube` and
    fully-specified `weights`.
    """
    # Create very separate clusters; check that vertices are unique and
    # correspond to classes
    class_sep = 1e6
    make = partial(
        make_classification,
        class_sep=class_sep,
        n_redundant=0,
        n_repeated=0,
        flip_y=0,
        shift=0,
        scale=1,
        shuffle=False,
    )

    for n_informative, weights, n_clusters_per_class in [
        (2, [1], 1),
        (2, [1 / 3] * 3, 1),
        (2, [1 / 4] * 4, 1),
        (2, [1 / 2] * 2, 2),
        (2, [3 / 4, 1 / 4], 2),
        (10, [1 / 3] * 3, 10),
        (np.int_(64), [1], 1),
    ]:
        n_classes = len(weights)
        n_clusters = n_classes * n_clusters_per_class
        n_samples = n_clusters * 50

        for hypercube in (False, True):
            generated = make(
                n_samples=n_samples,
                n_classes=n_classes,
                weights=weights,
                n_features=n_informative,
                n_informative=n_informative,
                n_clusters_per_class=n_clusters_per_class,
                hypercube=hypercube,
                random_state=0,
            )

            X, y = mt.ExecutableTuple(generated).execute().fetch()
            assert X.shape == (n_samples, n_informative)
            assert y.shape == (n_samples,)

            # Cluster by sign, viewed as strings to allow uniquing
            signs = np.sign(X)
            signs = signs.view(dtype=f"|S{signs.strides[0]}")
            unique_signs, cluster_index = np.unique(signs, return_inverse=True)

            assert len(unique_signs) == n_clusters

            clusters_by_class = defaultdict(set)
            for cluster, cls in zip(cluster_index, y):
                clusters_by_class[cls].add(cluster)
            for clusters in clusters_by_class.values():
                assert len(clusters) == n_clusters_per_class
            assert len(clusters_by_class) == n_classes

            assert_array_almost_equal(
                np.bincount(y) / len(y) // weights,
                [1] * n_classes,
                err_msg="Wrong number of samples per class",
            )

            # Ensure on vertices of hypercube
            for cluster in range(len(unique_signs)):
                centroid = X[cluster_index == cluster].mean(axis=0)
                if hypercube:
                    assert_array_almost_equal(
                        np.abs(centroid) / class_sep,
                        np.ones(n_informative),
                        decimal=5,
                        err_msg="Clusters are not centered on hypercube vertices",
                    )
                else:
                    assert_raises(
                        AssertionError,
                        assert_array_almost_equal,
                        np.abs(centroid) / class_sep,
                        np.ones(n_informative),
                        decimal=5,
                        err_msg="Clusters should not be centered "
                        "on hypercube vertices",
                    )

    assert_raises(
        ValueError,
        make,
        n_features=2,
        n_informative=2,
        n_classes=5,
        n_clusters_per_class=1,
    )
    assert_raises(
        ValueError,
        make,
        n_features=2,
        n_informative=2,
        n_classes=3,
        n_clusters_per_class=2,
    )


def test_make_regression(setup):
    X, y, c = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=3,
        effective_rank=5,
        coef=True,
        bias=0.0,
        noise=1.0,
        random_state=0,
    )
    X, y, c = mt.ExecutableTuple((X, y, c)).execute().fetch()

    assert X.shape == (100, 10), "X shape mismatch"
    assert y.shape == (100,), "y shape mismatch"
    assert c.shape == (10,), "coef shape mismatch"
    assert sum(c != 0.0) == 3, "Unexpected number of informative features"

    # Test that y ~= np.dot(X, c) + bias + N(0, 1.0).
    assert_almost_equal(np.std(y - np.dot(X, c)), 1.0, decimal=1)

    # Test with small number of features.
    X, y = make_regression(n_samples=100, n_features=1)  # n_informative=3
    assert X.shape == (100, 1)


def test_make_regression_multitarget():
    X, y, c = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=3,
        n_targets=3,
        coef=True,
        noise=1.0,
        random_state=0,
    )
    X, y, c = mt.ExecutableTuple((X, y, c)).execute().fetch()

    assert X.shape == (100, 10), "X shape mismatch"
    assert y.shape == (100, 3), "y shape mismatch"
    assert c.shape == (10, 3), "coef shape mismatch"
    np.testing.assert_array_equal(
        sum(c != 0.0), 3, "Unexpected number of informative features"
    )

    # Test that y ~= np.dot(X, c) + bias + N(0, 1.0)
    assert_almost_equal(np.std(y - np.dot(X, c)), 1.0, decimal=1)


def test_make_blobs(setup):
    cluster_stds = np.array([0.05, 0.2, 0.4])
    cluster_centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    X, y = make_blobs(
        random_state=0,
        n_samples=50,
        n_features=2,
        centers=cluster_centers,
        cluster_std=cluster_stds,
    )
    X, y = mt.ExecutableTuple((X, y)).execute().fetch()
    assert X.shape == (50, 2)
    assert y.shape == (50,)
    assert np.unique(y).shape == (3,)
    for i, (ctr, std) in enumerate(zip(cluster_centers, cluster_stds)):
        assert_almost_equal((X[y == i] - ctr).std(), std, 1, "Unexpected std")


def test_make_blobs_n_samples_list(setup):
    n_samples = [50, 30, 20]
    X, y = make_blobs(n_samples=n_samples, n_features=2, random_state=0)
    X, y = mt.ExecutableTuple((X, y)).execute().fetch()

    assert X.shape == (sum(n_samples), 2)
    assert all(np.bincount(y, minlength=len(n_samples)) == n_samples) is True


def test_make_blobs_n_samples_list_with_centers(setup):
    n_samples = [20, 20, 20]
    centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cluster_stds = np.array([0.05, 0.2, 0.4])
    X, y = make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=cluster_stds, random_state=0
    )
    X, y = mt.ExecutableTuple((X, y)).execute().fetch()

    assert X.shape == (sum(n_samples), 2)
    assert all(np.bincount(y, minlength=len(n_samples)) == n_samples) is True
    for i, (ctr, std) in enumerate(zip(centers, cluster_stds)):
        assert_almost_equal((X[y == i] - ctr).std(), std, 1, "Unexpected std")


def test_make_blobs_n_samples_centers_none(setup):
    for n_samples in [[5, 3, 0], np.array([5, 3, 0]), tuple([5, 3, 0])]:
        centers = None
        X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=0)
        X, y = mt.ExecutableTuple((X, y)).execute().fetch()

        assert X.shape == (sum(n_samples), 2)
        assert all(np.bincount(y, minlength=len(n_samples)) == n_samples) is True


def test_make_blobs_error(setup):
    n_samples = [20, 20, 20]
    centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cluster_stds = np.array([0.05, 0.2, 0.4])
    wrong_centers_msg = (
        "Length of `n_samples` not consistent "
        f"with number of centers. Got n_samples = {n_samples} "
        f"and centers = {centers[:-1]}"
    )
    assert_raise_message(
        ValueError, wrong_centers_msg, make_blobs, n_samples, centers=centers[:-1]
    )
    wrong_std_msg = (
        "Length of `clusters_std` not consistent with "
        f"number of centers. Got centers = {mt.tensor(centers)} "
        f"and cluster_std = {cluster_stds[:-1]}"
    )
    assert_raise_message(
        ValueError,
        wrong_std_msg,
        make_blobs,
        n_samples,
        centers=centers,
        cluster_std=cluster_stds[:-1],
    )
    wrong_type_msg = f"Parameter `centers` must be array-like. Got {3!r} instead"
    assert_raise_message(ValueError, wrong_type_msg, make_blobs, n_samples, centers=3)


def test_make_low_rank_matrix(setup):
    X = make_low_rank_matrix(
        n_samples=50,
        n_features=25,
        effective_rank=5,
        tail_strength=0.01,
        random_state=0,
    )

    assert X.shape == (50, 25)

    _, s, _ = svd(X)
    assert (s.sum() - 5).to_numpy() < 0.1
