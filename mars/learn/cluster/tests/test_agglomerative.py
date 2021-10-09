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
import pytest
from scipy import sparse
from functools import partial
try:
    # from sklearn.cluster import ward_tree
    from sklearn.neighbors import kneighbors_graph
    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn.utils._testing import assert_array_equal
    from sklearn.utils._testing import assert_almost_equal
    from sklearn.feature_extraction.image import grid_to_graph
    from sklearn.metrics.cluster import normalized_mutual_info_score
    from sklearn.cluster._agglomerative import (
        _hc_cut,
        _TREE_BUILDERS,
        _fix_connectivity,
    )
except ImportError:
    pass

from .. import AgglomerativeClustering


@pytest.mark.skipif(AgglomerativeClustering is None, reason='scikit-learn not installed')
def test_structured_linkage_tree(setup):
    # Check that we obtain the correct solution for structured linkage trees.
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    # Avoiding a mask with only 'True' entries
    mask[4:7, 4:7] = 0
    X = rng.randn(50, 100)
    connectivity = grid_to_graph(*mask.shape)
    for tree_builder in _TREE_BUILDERS.values():
        children, n_components, n_leaves, parent = tree_builder(
            X.T, connectivity=connectivity
        )
        n_nodes = 2 * X.shape[1] - 1
        assert len(children) + n_leaves == n_nodes
        # Check that ward_tree raises a ValueError with a connectivity matrix
        # of the wrong shape
        with pytest.raises(ValueError):
            tree_builder(X.T, connectivity=np.ones((4, 4)))
        # Check that fitting with no samples raises an error
        with pytest.raises(ValueError):
            tree_builder(X.T[:0], connectivity=connectivity)


@pytest.mark.skipif(AgglomerativeClustering is None, reason='scikit-learn not installed')
def test_height_linkage_tree(setup):
    # Check that the height of the results of linkage tree is sorted.
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    X = rng.randn(50, 100)
    connectivity = grid_to_graph(*mask.shape)
    for linkage_func in _TREE_BUILDERS.values():
        children, n_nodes, n_leaves, parent = linkage_func(
            X.T, connectivity=connectivity
        )
        n_nodes = 2 * X.shape[1] - 1
        assert len(children) + n_leaves == n_nodes


@pytest.mark.skipif(AgglomerativeClustering is None, reason='scikit-learn not installed')
@pytest.mark.parametrize("n_clusters, distance_threshold", [(None, 0.5), (10, None)])
@pytest.mark.parametrize("compute_distances", [True, False])
@pytest.mark.parametrize("linkage", ["ward"])
def test_agglomerative_clustering_distances(setup,
    n_clusters, compute_distances, distance_threshold, linkage
):
    # Check that when `compute_distances` is True or `distance_threshold` is
    # given, the fitted model has an attribute `distances_`.
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        connectivity=connectivity,
        linkage=linkage,
        distance_threshold=distance_threshold,
        compute_distances=compute_distances,
    )
    clustering.fit(X)
    if compute_distances or (distance_threshold is not None):
        assert hasattr(clustering, "distances_")
        n_children = clustering.children_.shape[0]
        n_nodes = n_children + 1
        assert clustering.distances_.shape == (n_nodes - 1,)
    else:
        assert not hasattr(clustering, "distances_")


@pytest.mark.skipif(AgglomerativeClustering is None, reason='scikit-learn not installed')
def test_agglomerative_clustering(setup):
    # Check that we obtain the correct number of clusters with
    # agglomerative clustering.
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)
    linkage = "ward"
    clustering = AgglomerativeClustering(
        n_clusters=10, connectivity=connectivity, linkage=linkage
    )
    clustering.fit(X)
    labels = clustering.labels_.to_numpy()

    clustering = AgglomerativeClustering(
        n_clusters=10, connectivity=connectivity, linkage=linkage
    )
    # Check that we obtain the same solution with early-stopping of the
    # tree building
    clustering.compute_full_tree = False
    clustering.fit(X)
    assert_almost_equal(normalized_mutual_info_score(clustering.labels_.to_numpy(), labels), 1)
    clustering.connectivity = None
    clustering.fit(X)
    clustering.labels_ = clustering.labels_.to_numpy()
    assert np.size(np.unique(clustering.labels_)) == 10
    # Check that we raise a TypeError on dense matrices
    clustering = AgglomerativeClustering(
        n_clusters=10,
        connectivity=sparse.lil_matrix(connectivity.toarray()[:10, :10]),
        linkage=linkage,
    )
    with pytest.raises(ValueError):
        clustering.fit(X)

    # Test that using ward with another metric than euclidean raises an
    # exception
    clustering = AgglomerativeClustering(
        n_clusters=10,
        connectivity=connectivity.toarray(),
        affinity="manhattan",
        linkage="ward",
    )
    with pytest.raises(ValueError):
        clustering.fit(X)


@pytest.mark.skipif(AgglomerativeClustering is None, reason='scikit-learn not installed')
def test_identical_points(setup):
    # Ensure identical points are handled correctly when using mst with
    # a sparse connectivity matrix
    X = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]])
    true_labels = np.array([0, 0, 1, 1, 2, 2])
    connectivity = kneighbors_graph(X, n_neighbors=3, include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)
    connectivity, n_components = _fix_connectivity(X, connectivity, "euclidean")

    linkage = "ward"
    clustering = AgglomerativeClustering(
        n_clusters=3, linkage=linkage, connectivity=connectivity
    )
    clustering.fit(X)

    assert_almost_equal(
        normalized_mutual_info_score(clustering.labels_, true_labels), 1
    )


@pytest.mark.skipif(AgglomerativeClustering is None, reason='scikit-learn not installed')
def test_connectivity_propagation(setup):
    # Check that connectivity in the ward tree is propagated correctly during
    # merging.
    X = np.array(
        [
            (0.014, 0.120),
            (0.014, 0.099),
            (0.014, 0.097),
            (0.017, 0.153),
            (0.017, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.153),
            (0.018, 0.152),
            (0.018, 0.149),
            (0.018, 0.144),
        ]
    )
    connectivity = kneighbors_graph(X, 10, include_self=False)
    ward = AgglomerativeClustering(
        n_clusters=4, connectivity=connectivity, linkage="ward"
    )
    # If changes are not propagated correctly, fit crashes with an
    # IndexError
    ward.fit(X)


@pytest.mark.skipif(AgglomerativeClustering is None, reason='scikit-learn not installed')
def test_connectivity_fixing_non_lil(setup):
    # Check non regression of a bug if a non item assignable connectivity is
    # provided with more than one component.
    # create dummy data
    x = np.array([[0, 0], [1, 1]])
    # create a mask with several components to force connectivity fixing
    m = np.array([[True, False], [False, True]])
    c = grid_to_graph(n_x=2, n_y=2, mask=m)
    w = AgglomerativeClustering(connectivity=c, linkage="ward")
    with pytest.warns(UserWarning):
        w.fit(x)


@pytest.mark.skipif(AgglomerativeClustering is None, reason='scikit-learn not installed')
def test_connectivity_callable(setup):
    rng = np.random.RandomState(0)
    X = rng.rand(20, 5)
    connectivity = kneighbors_graph(X, 3, include_self=False)
    aglc1 = AgglomerativeClustering(connectivity=connectivity)
    aglc2 = AgglomerativeClustering(
        connectivity=partial(kneighbors_graph, n_neighbors=3, include_self=False)
    )
    aglc1.fit(X)
    aglc2.fit(X)
    assert_array_equal(aglc1.labels_, aglc2.labels_)


@pytest.mark.skipif(AgglomerativeClustering is None, reason='scikit-learn not installed')
def test_compute_full_tree(setup):
    # Test that the full tree is computed if n_clusters is small
    rng = np.random.RandomState(0)
    X = rng.randn(10, 2)
    connectivity = kneighbors_graph(X, 5, include_self=False)

    # When n_clusters is less, the full tree should be built
    # that is the number of merges should be n_samples - 1
    agc = AgglomerativeClustering(n_clusters=2, connectivity=connectivity)
    agc.fit(X)
    n_samples = X.shape[0]
    n_nodes = agc.children_.shape[0]
    assert n_nodes == n_samples - 1

    # When n_clusters is large, greater than max of 100 and 0.02 * n_samples.
    # we should stop when there are n_clusters.
    n_clusters = 101
    X = rng.randn(200, 2)
    connectivity = kneighbors_graph(X, 10, include_self=False)
    agc = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity)
    agc.fit(X)
    n_samples = X.shape[0]
    n_nodes = agc.children_.shape[0]
    assert n_nodes == n_samples - n_clusters


@pytest.mark.skipif(AgglomerativeClustering is None, reason='scikit-learn not installed')
@pytest.mark.parametrize("linkage", ["ward"])
def test_agglomerative_clustering_with_distance_threshold(linkage):
    # Check that we obtain the correct number of clusters with
    # agglomerative clustering with distance_threshold.
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)
    # test when distance threshold is set to 10
    distance_threshold = 10
    for conn in [None, connectivity]:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            connectivity=conn,
            linkage=linkage,
        )
        clustering.fit(X)
        clusters_produced = clustering.labels_
        num_clusters_produced = len(np.unique(clustering.labels_.to_numpy()))
        # test if the clusters produced match the point in the linkage tree
        # where the distance exceeds the threshold
        tree_builder = _TREE_BUILDERS[linkage]
        children, n_components, n_leaves, parent, distances = tree_builder(
            X, connectivity=conn, n_clusters=None, return_distance=True
        )
        num_clusters_at_threshold = (
            np.count_nonzero(distances >= distance_threshold) + 1
        )
        # test number of clusters produced
        assert num_clusters_at_threshold == num_clusters_produced
        # test clusters produced
        clusters_at_threshold = _hc_cut(
            n_clusters=num_clusters_produced, children=children, n_leaves=n_leaves
        )
        assert np.array_equiv(clusters_produced.to_numpy(), clusters_at_threshold)


@pytest.mark.skipif(AgglomerativeClustering is None, reason='scikit-learn not installed')
@pytest.mark.parametrize("linkage", ["ward"])
@pytest.mark.parametrize(
    ("threshold", "y_true"), [(0.5, [1, 0]), (1.0, [1, 0]), (1.5, [0, 0])]
)
def test_agglomerative_clustering_with_distance_threshold_edge_case(setup,
    linkage, threshold, y_true
):
    # test boundary case of distance_threshold matching the distance
    X = [[0], [1]]
    clusterer = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, linkage=linkage
    )
    y_pred = clusterer.fit_predict(X)
    assert adjusted_rand_score(y_true, y_pred) == 1


def test_dist_threshold_invalid_parameters(setup):
    X = [[0], [1]]
    with pytest.raises(ValueError, match="Exactly one of "):
        AgglomerativeClustering(n_clusters=None, distance_threshold=None).fit(X)

    with pytest.raises(ValueError, match="Exactly one of "):
        AgglomerativeClustering(n_clusters=2, distance_threshold=1).fit(X)

    X = [[0], [1]]
    with pytest.raises(ValueError, match="compute_full_tree must be True if"):
        AgglomerativeClustering(
            n_clusters=None, distance_threshold=1, compute_full_tree=False
        ).fit(X)
