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

import sys
from io import StringIO

import numpy as np
import pytest
import scipy.sparse as sp

try:
    from sklearn.datasets import make_blobs
    from sklearn.metrics.cluster import v_measure_score
    from sklearn.utils._testing import assert_raise_message
except ImportError:
    pass

from .... import tensor as mt
from ....core import TileableGraphBuilder, TileableGraph, ChunkGraphBuilder
from ....config import options
from .. import KMeans, k_means
from .._kmeans import _init_centroids


@pytest.mark.skipif(KMeans is None, reason="scikit-learn not installed")
@pytest.mark.parametrize("representation", ["dense", "sparse"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("algo", ["full", "elkan"])
def test_k_means_results(setup, representation, dtype, algo):
    array_constr = {"dense": np.array, "sparse": sp.csr_matrix}[representation]

    X = array_constr([[0, 0], [0.5, 0], [0.5, 1], [1, 1]], dtype=dtype)
    sample_weight = [3, 1, 1, 3]  # will be rescaled to [1.5, 0.5, 0.5, 1.5]
    init_centers = np.array([[0, 0], [1, 1]], dtype=dtype)

    expected_labels = [0, 0, 1, 1]
    expected_inertia = 0.1875
    expected_centers = np.array([[0.125, 0], [0.875, 1]], dtype=dtype)
    expected_n_iter = 2

    kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers, algorithm=algo)
    kmeans.fit(X, sample_weight=sample_weight)

    np.testing.assert_array_equal(kmeans.labels_, expected_labels)
    np.testing.assert_almost_equal(kmeans.inertia_, expected_inertia)
    np.testing.assert_array_almost_equal(kmeans.cluster_centers_, expected_centers)
    assert kmeans.n_iter_ == expected_n_iter


@pytest.mark.skipif(KMeans is None, reason="scikit-learn not installed")
@pytest.mark.parametrize("representation", ["dense", "sparse"])
@pytest.mark.parametrize("algo", ["full", "elkan"])
def test_relocated_clusters(setup, representation, algo):
    # check that empty clusters are relocated as expected

    # second center too far from others points will be empty at first iter
    init_centers = np.array([[0.5, 0.5], [3, 3]])

    expected_labels = [0, 0, 1, 1]
    expected_inertia = 0.25
    expected_centers = [[0.25, 0], [0.75, 1]]
    expected_n_iter = 3

    array_constr = {"dense": np.array, "sparse": sp.csr_matrix}[representation]
    X = array_constr([[0, 0], [0.5, 0], [0.5, 1], [1, 1]])

    kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers, algorithm=algo)
    kmeans.fit(X)

    np.testing.assert_array_equal(kmeans.labels_, expected_labels)
    np.testing.assert_almost_equal(kmeans.inertia_, expected_inertia)
    np.testing.assert_array_almost_equal(kmeans.cluster_centers_, expected_centers)
    assert kmeans.n_iter_ == expected_n_iter


@pytest.mark.skipif(KMeans is None, reason="scikit-learn not installed")
@pytest.mark.parametrize("distribution", ["normal", "blobs"])
@pytest.mark.parametrize("tol", [1e-2, 1e-4, 1e-8])
def test_elkan_results(setup, distribution, tol):
    # check that results are identical between lloyd and elkan algorithms

    rnd = np.random.RandomState(0)
    if distribution == "normal":
        X = rnd.normal(size=(5000, 10))
    else:
        X, _ = make_blobs(random_state=rnd)

    km_full = KMeans(
        algorithm="full",
        n_clusters=5,
        random_state=0,
        n_init=1,
        tol=tol,
        init="k-means++",
    )
    km_elkan = KMeans(
        algorithm="elkan",
        n_clusters=5,
        random_state=0,
        n_init=1,
        tol=tol,
        init="k-means++",
    )

    km_full.fit(X)
    km_elkan.fit(X)
    np.testing.assert_allclose(km_elkan.cluster_centers_, km_full.cluster_centers_)
    np.testing.assert_array_equal(km_elkan.labels_, km_full.labels_)

    assert km_elkan.n_iter_ == km_full.n_iter_
    assert km_elkan.inertia_ == pytest.approx(km_full.inertia_, rel=1e-6)


@pytest.mark.skipif(KMeans is None, reason="scikit-learn not installed")
def test_k_means_convergence(setup):
    for algorithm in ["full", "elkan"]:
        # Check that KMeans stops when convergence is reached when tol=0. (#16075)
        rnd = np.random.RandomState(0)
        X = rnd.normal(size=(5000, 10))

        km = KMeans(
            algorithm=algorithm,
            n_clusters=5,
            random_state=0,
            n_init=1,
            tol=0,
            max_iter=300,
            init="k-means++",
        ).fit(X)

        assert km.n_iter_ < 300


@pytest.mark.skipif(KMeans is None, reason="scikit-learn not installed")
def test_elkan_results_sparse(setup):
    for distribution in ["normal", "blobs"]:
        # check that results are identical between lloyd and elkan algorithms
        # with sparse input
        rnd = np.random.RandomState(0)
        if distribution == "normal":
            X = sp.random(100, 100, density=0.1, format="csr", random_state=rnd)
            X.data = rnd.randn(len(X.data))
        else:
            X, _ = make_blobs(n_samples=100, n_features=100, random_state=rnd)
            X = sp.csr_matrix(X)

        km_full = KMeans(
            algorithm="full", n_clusters=5, random_state=0, n_init=1, init="k-means++"
        )
        km_elkan = KMeans(
            algorithm="elkan", n_clusters=5, random_state=0, n_init=1, init="k-means++"
        )

        km_full.fit(X)
        km_elkan.fit(X)
        np.testing.assert_allclose(km_elkan.cluster_centers_, km_full.cluster_centers_)
        np.testing.assert_allclose(km_elkan.labels_, km_full.labels_)


@pytest.mark.skipif(KMeans is None, reason="scikit-learn not installed")
def test_k_means_new_centers(setup):
    # Explore the part of the code where a new center is reassigned
    X = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ]
    )
    labels = [0, 1, 2, 1, 1, 2]
    bad_centers = np.array([[+0, 1, 0, 0], [0.2, 0, 0.2, 0.2], [+0, 0, 0, 0]])

    km = KMeans(
        n_clusters=3,
        init=bad_centers,
        n_init=1,
        max_iter=10,
        random_state=1,
        algorithm="elkan",
    )
    for this_X in (X, sp.coo_matrix(X)):
        km.fit(this_X)
        this_labels = km.labels_.fetch()
        # Reorder the labels so that the first instance is in cluster 0,
        # the second in cluster 1, ...
        this_labels = np.unique(this_labels, return_index=True)[1][this_labels]
        np.testing.assert_array_equal(this_labels, labels)


def _check_fitted_model(km, n_clusters, n_features, true_labels):
    # check that the number of clusters centers and distinct labels match
    # the expectation
    centers = km.cluster_centers_
    assert centers.shape == (n_clusters, n_features)

    labels = km.labels_.fetch()
    assert np.unique(labels).shape[0] == n_clusters

    # check that the labels assignment are perfect (up to a permutation)
    assert v_measure_score(true_labels, labels) == 1.0
    assert km.inertia_ > 0.0

    # check error on dataset being too small
    assert_raise_message(
        ValueError,
        "n_samples=1 should be >= n_clusters=%d" % km.n_clusters,
        km.fit,
        [[0.0, 1.0]],
    )


@pytest.mark.skipif(KMeans is None, reason="scikit-learn not installed")
def test_k_means_init(setup):
    # non centered, sparse centers to check the
    centers = np.array(
        [
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 5.0, 1.0],
        ]
    )
    n_samples = 100
    n_clusters, n_features = centers.shape
    X, true_labels = make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42
    )
    X_csr = sp.csr_matrix(X)
    for data in [X, X_csr]:
        for init in ["random", "k-means++", "k-means||", centers.copy()]:
            data = mt.tensor(data, chunk_size=50)
            km = KMeans(
                init=init,
                n_clusters=n_clusters,
                random_state=42,
                n_init=1,
                algorithm="elkan",
            )
            km.fit(data)
            _check_fitted_model(km, n_clusters, n_features, true_labels)

    X = mt.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=1, init="k-means||").fit(X)
    assert sorted(kmeans.cluster_centers_.fetch().tolist()) == sorted(
        [[10.0, 2.0], [1.0, 2.0]]
    )


@pytest.mark.skipif(KMeans is None, reason="scikit-learn not installed")
def test_k_means_n_init(setup):
    rnd = np.random.RandomState(0)
    X = rnd.normal(size=(40, 2))

    # two regression tests on bad n_init argument
    # previous bug: n_init <= 0 threw non-informative TypeError (#3858)
    with pytest.raises(ValueError, match="n_init"):
        KMeans(n_init=0, init="k-means++").fit(X)
    with pytest.raises(ValueError, match="n_init"):
        KMeans(n_init=-1, init="k-means++").fit(X)


@pytest.mark.skipif(KMeans is None, reason="scikit-learn not installed")
def test_k_means_explicit_init_shape(setup):
    # test for sensible errors when giving explicit init
    # with wrong number of features or clusters
    rnd = np.random.RandomState(0)
    X = rnd.normal(size=(40, 3))

    # mismatch of number of features
    km = KMeans(n_init=1, init=X[:, :2], n_clusters=len(X), algorithm="elkan")
    msg = "does not match the number of features of the data"
    with pytest.raises(ValueError, match=msg):
        km.fit(X)
    # for callable init
    km = KMeans(
        n_init=1,
        init=lambda X_, k, random_state: X_[:, :2],
        n_clusters=len(X),
        algorithm="elkan",
    )
    with pytest.raises(ValueError, match=msg):
        km.fit(X)
    # mismatch of number of clusters
    msg = "does not match the number of clusters"
    km = KMeans(n_init=1, init=X[:2, :], n_clusters=3, algorithm="elkan")
    with pytest.raises(ValueError, match=msg):
        km.fit(X)
    # for callable init
    km = KMeans(
        n_init=1,
        init=lambda X_, k, random_state: X_[:2, :],
        n_clusters=3,
        algorithm="elkan",
    )
    with pytest.raises(ValueError, match=msg):
        km.fit(X)


@pytest.mark.skipif(KMeans is None, reason="scikit-learn not installed")
def test_k_means_fortran_aligned_data(setup):
    # Check the KMeans will work well, even if X is a fortran-aligned data.
    X = np.asfortranarray([[0, 0], [0, 1], [0, 1]])
    centers = np.array([[0, 0], [0, 1]])
    labels = np.array([0, 1, 1])
    km = KMeans(
        n_init=1, init=centers, random_state=42, n_clusters=2, algorithm="elkan"
    )
    km.fit(X)
    np.testing.assert_array_almost_equal(km.cluster_centers_, centers)
    np.testing.assert_array_equal(km.labels_, labels)


@pytest.mark.skipif(KMeans is None, reason="scikit-learn not installed")
@pytest.mark.parametrize("algo", ["full", "elkan"])
@pytest.mark.parametrize(
    "seed, max_iter, tol",
    [
        (0, 2, 1e-7),  # strict non-convergence
        (1, 2, 1e-1),  # loose non-convergence
        (3, 300, 1e-7),  # strict convergence
        (4, 300, 1e-1),  # loose convergence
    ],
)
def test_k_means_fit_predict(setup, algo, seed, max_iter, tol):
    # check that fit.predict gives same result as fit_predict
    rng = np.random.RandomState(seed)

    X = make_blobs(n_samples=1000, n_features=10, centers=10, random_state=rng)[0]

    kmeans = KMeans(
        algorithm=algo,
        n_clusters=10,
        random_state=seed,
        tol=tol,
        max_iter=max_iter,
        init="k-means++",
    )

    labels_1 = kmeans.fit(X).predict(X)
    labels_2 = kmeans.fit_predict(X)

    # Due to randomness in the order in which chunks of data are processed when
    # using more than one thread, the absolute values of the labels can be
    # different between the 2 strategies but they should correspond to the same
    # clustering.
    assert pytest.approx(v_measure_score(labels_1, labels_2)) == 1


@pytest.mark.skipif(KMeans is None, reason="scikit-learn not installed")
def test_transform(setup):
    centers = np.array(
        [
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 5.0, 1.0],
        ]
    )
    n_samples = 100
    n_clusters, n_features = centers.shape
    X = make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42
    )[0]

    km = KMeans(n_clusters=n_clusters, init="k-means++", algorithm="elkan")
    km.fit(X)
    X_new = km.transform(km.cluster_centers_).fetch()

    for c in range(n_clusters):
        assert X_new[c, c] == 0
        for c2 in range(n_clusters):
            if c != c2:
                assert X_new[c, c2] > 0


@pytest.mark.skipif(KMeans is None, reason="scikit-learn not installed")
def test_fit_transform(setup):
    centers = np.array(
        [
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 5.0, 1.0],
        ]
    )
    n_samples = 100
    X = make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42
    )[0]
    X1 = (
        KMeans(n_clusters=3, random_state=51, init="k-means++", algorithm="elkan")
        .fit(X)
        .transform(X)
    )
    X2 = KMeans(
        n_clusters=3, random_state=51, init="k-means++", algorithm="elkan"
    ).fit_transform(X)
    np.testing.assert_array_almost_equal(X1, X2)


@pytest.mark.skipif(KMeans is None, reason="scikit-learn not installed")
def test_score(setup):
    centers = np.array(
        [
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 5.0, 1.0],
        ]
    )
    n_samples = 100
    n_clusters, n_features = centers.shape
    X = make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42
    )[0]

    for algo in ["full", "elkan"]:
        # Check that fitting k-means with multiple inits gives better score
        km1 = KMeans(
            n_clusters=n_clusters,
            max_iter=1,
            random_state=42,
            n_init=1,
            algorithm=algo,
            init="k-means++",
        )
        s1 = km1.fit(X).score(X).fetch()
        km2 = KMeans(
            n_clusters=n_clusters,
            max_iter=10,
            random_state=42,
            n_init=1,
            algorithm=algo,
            init="k-means++",
        )
        s2 = km2.fit(X).score(X).fetch()
        assert s2 > s1


@pytest.mark.skipif(KMeans is None, reason="scikit-learn not installed")
def test_k_means_function(setup):
    # test calling the k_means function directly

    # non centered, sparse centers to check the
    centers = np.array(
        [
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 5.0, 1.0],
        ]
    )
    n_samples = 100
    n_clusters, n_features = centers.shape
    X, true_labels = make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42
    )

    # catch output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        cluster_centers, labels, inertia = k_means(
            X, n_clusters=n_clusters, sample_weight=None, verbose=True, init="k-means++"
        )
    finally:
        sys.stdout = old_stdout
    centers = cluster_centers
    assert centers.shape == (n_clusters, n_features)

    labels = labels.fetch()
    assert np.unique(labels).shape[0] == n_clusters

    # check that the labels assignment are perfect (up to a permutation)
    assert v_measure_score(true_labels, labels) == 1.0
    assert inertia > 0.0

    # check warning when centers are passed
    with pytest.warns(RuntimeWarning):
        k_means(
            X,
            n_clusters=n_clusters,
            sample_weight=None,
            init=centers,
        )

    # to many clusters desired
    with pytest.raises(ValueError):
        k_means(X, n_clusters=X.shape[0] + 1, sample_weight=None, init="k-means++")


@pytest.mark.skipif(KMeans is None, reason="scikit-learn not installed")
def test_k_means_init_large_n_clusters():
    chunk_bytes_limit = options.chunk_store_limit * 2
    n_cluster = 2000
    x = mt.random.rand(1000_000, 64, chunk_size=250_000)

    centers = _init_centroids(x, n_cluster, init="k-means||")
    t_graph = next(TileableGraphBuilder(TileableGraph([centers])).build())
    graph = next(ChunkGraphBuilder(t_graph).build())
    for c in graph:
        nbytes = c.nbytes
        if not np.isnan(nbytes):
            assert nbytes <= chunk_bytes_limit
