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
    from sklearn.utils._testing import assert_allclose
    from sklearn.utils._testing import assert_array_equal
    from sklearn.utils._testing import assert_almost_equal  
    from sklearn.utils._testing import assert_array_almost_equal
except ImportError:
    pass

from .. import MiniBatchKMeans

from .... import tensor as mt
from .._mini_batch_k_means_operand import _mini_batch_step
from ...utils.extmath import row_norms
from ...metrics.pairwise import pairwise_distances


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


def align_labels(true_labels, labels):
    # align labels when cluster labels index are inconsistent
    # eg. [0 0 1 2 1] [1 1 0 2 0]
    n_cluster = true_labels.max() + 1
    print(n_cluster)
    mapping = np.zeros(n_cluster)
    for i in range(n_cluster):
        for j in range(len(labels)):
            if labels[j] == i:
                mapping[labels[j]] = true_labels[j]
                break
    align_labels = np.array([mapping[item] for item in labels], dtype=np.int)
    return align_labels



def _check_fitted_model(model):
    # check that the number of clusters centers and distinct labels match
    # the expectation
    centers = model.cluster_centers_
    assert centers.shape == (n_clusters, n_features)

    labels = model.labels_.fetch()
    assert np.unique(labels).shape[0] == n_clusters

    # check that the labels assignment are perfect (up to a permutation)
    if not np.all(labels == true_labels):
        labels = align_labels(true_labels, labels)
    assert_allclose(v_measure_score(true_labels, labels), 1.0)
    assert model.inertia_.fetch() > 0.0


@pytest.mark.skipif(MiniBatchKMeans is None, reason='scikit-learn no installed')
@pytest.mark.parametrize('representation', ['dense'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_mini_batch_k_means_results(setup, representation, dtype):
    array_constr = {'dense': np.array, 'sparse': sp.csr_matrix}[representation]

    X = array_constr([[0, 0], [0.5, 0], [0.5, 1], [1, 1]], dtype=dtype)
    sample_weight = [3, 1, 1, 3]  # will be rescaled to [1.5, 0.5, 0.5, 1.5]
    init_centers = np.array([[0, 0], [1, 1]], dtype=dtype)

    # due to randomness in miniBatch kmeans
    # testing expected results need to set random seed
    mt.random.seed(0)
    expected_labels = [0, 0, 1, 1]
    expected_inertia = 0.1875764
    expected_centers = np.array([[0.125874, 0], [0.881118, 1]], dtype=dtype)
    expected_n_iter = 16

    mini_batch_kmeans = MiniBatchKMeans(n_clusters=2, n_init=1, init=init_centers)
    mini_batch_kmeans.fit(X, sample_weight=sample_weight)

    print(mini_batch_kmeans.labels_)
    print(mini_batch_kmeans.inertia_)
    print(mini_batch_kmeans.cluster_centers_)
    print(mini_batch_kmeans.n_iter_)

    np.testing.assert_array_equal(mini_batch_kmeans.labels_, expected_labels)
    np.testing.assert_almost_equal(mini_batch_kmeans.inertia_.fetch(), expected_inertia)
    np.testing.assert_array_almost_equal(mini_batch_kmeans.cluster_centers_, expected_centers)
    assert mini_batch_kmeans.n_iter_ == expected_n_iter


@pytest.mark.skipif(MiniBatchKMeans is None, reason='scikit-learn no installed')
def test_mini_batch_k_means_convergence(setup):
    rnd = np.random.RandomState(0)
    X = rnd.normal(size=(5000, 10))

    mini_batch_km = MiniBatchKMeans(n_clusters=5, random_state=0, n_init=1,
                tol=0, max_iter=300, init='k-means++').fit(X)
    assert mini_batch_km.n_iter_ < 300


@pytest.mark.skipif(MiniBatchKMeans is None, reason='scikit-learn no installed')
def test_minibatch_reassign(setup):
    # Check the reassignment part of the minibatch step with very high or very
    # low reassignment ratio.
    mt_X = mt.tensor(X)
    sample_weight = mt.ones(n_samples)
    sample_weight.execute()
    x_squared_norms = row_norms(mt_X, squared=True)
    weight_sums = mt.zeros(n_clusters)  

    mb_k_means = MiniBatchKMeans(n_clusters=n_clusters, batch_size=100,
                                    random_state=42)
    mb_k_means.fit(X)

    score_before = mb_k_means.score(X)

    centers_new, inertia_new, squared_diff = _mini_batch_step(
            mt_X,
            sample_weight,
            x_squared_norms,
            mb_k_means.cluster_centers_,
            n_clusters,
            False,
            mb_k_means._counts,
            random_reassign=True,
            random_state=mt.random.RandomState(0).to_numpy(),
            reassignment_ratio=1.0,
    )
    mb_k_means.cluster_centers_ = centers_new
    assert score_before.fetch() > mb_k_means.score(mt_X).fetch()

    # Give a perfect initialization, with a small reassignment_ratio,
    # no center should be reassigned
    perfect_centers = mt.empty((n_clusters, n_features))
    for i in range(n_clusters):
        perfect_centers[i] = X[true_labels == i].mean(axis=0)

    weight_sums = mt.zeros(n_clusters)  
    test_center = mt.zeros(X.shape[0], mt.double)
    to_runs = [perfect_centers, weight_sums, test_center]
    mt.ExecutableTuple(to_runs).execute()

    clusters_before = perfect_centers
    centers_new, inertia_new, squared_diff = _mini_batch_step(
            mt_X,
            sample_weight,
            x_squared_norms,
            perfect_centers,
            n_clusters,
            False,
            weight_sums,
            random_reassign=True,
            random_state=mt.random.RandomState(0).to_numpy(),
            reassignment_ratio=1e-15,
    )
    assert_array_almost_equal(clusters_before, centers_new)


@pytest.mark.skipif(MiniBatchKMeans is None, reason='scikit-learn no installed')
def test_minibatch_with_many_reassignments(setup):
    # Test for the case that the number of clusters to reassign is bigger
    # than the batch_size. Run the test with 100 clusters and a batch_size of
    # 10 because it turned out that these values ensure that the number of
    # clusters to reassign is always bigger than the batch_size.
    MiniBatchKMeans(
        n_clusters=100,
        batch_size=10,
        init_size=n_samples,
        random_state=42,
        verbose=True,
    ).fit(X)


@pytest.mark.skipif(MiniBatchKMeans is None, reason='scikit-learn no installed')
def test_minibatch_kmeans_init_size(setup):
    # Check the internal _init_size attribute of MiniBatchKMeans

    # default init size should be 3 * batch_size
    km = MiniBatchKMeans(n_clusters=10, batch_size=5, n_init=1).fit(X)
    assert km._init_size == 15

    # if 3 * batch size < n_clusters, it should then be 3 * n_clusters
    km = MiniBatchKMeans(n_clusters=10, batch_size=1, n_init=1).fit(X)
    assert km._init_size == 30

    # it should not be larger than n_samples
    km = MiniBatchKMeans(n_clusters=10, batch_size=5, n_init=1,
                         init_size=n_samples + 1).fit(X)
    assert km._init_size == n_samples


@pytest.mark.skipif(MiniBatchKMeans is None, reason='scikit-learn no installed')
def test_score_max_iter(setup):
    # Check that fitting KMeans or MiniBatchKMeans with more iterations gives
    # better score
    X = np.random.RandomState(0).randn(100, 10)

    km1 = MiniBatchKMeans(n_init=1, random_state=42, max_iter=1)
    s1 = km1.fit(X).score(X)
    km2 = MiniBatchKMeans(n_init=1, random_state=42, max_iter=10)
    s2 = km2.fit(X).score(X)
    assert s2.fetch() > s1.fetch()


@pytest.mark.skipif(MiniBatchKMeans is None, reason='scikit-learn no installed')
@pytest.mark.parametrize("init", ["random", "k-means++"])
def test_predict(setup, init):
    # Check the predict method and the equivalence between fit.predict and
    # fit_predict.

    # There's a very small chance of failure with elkan on unstructured dataset
    # because predict method uses fast euclidean distances computation which
    # may cause small numerical instabilities.
    X, _ = make_blobs(n_samples=500, n_features=10, centers=10, random_state=0)

    # With n_init = 1
    km = MiniBatchKMeans(n_clusters=10, init=init, n_init=1, random_state=0)
    # if algorithm is not None:
    #     km.set_params(algorithm=algorithm)
    km.fit(X)
    labels = km.labels_

    # re-predict labels for training set using predict
    pred = km.predict(X)
    assert_array_equal(pred, labels)

    # re-predict labels for training set using fit_predict
    pred = km.fit_predict(X)
    assert_array_equal(pred, labels)

    # predict centroid labels
    pred = km.predict(km.cluster_centers_)
    assert_array_equal(pred, np.arange(10))

    # With n_init > 1
    # Due to randomness in the order in which chunks of data are processed when
    # using more than one thread, there might be different rounding errors for
    # the computation of the inertia between 2 runs. This might result in a
    # different ranking of 2 inits, hence a different labeling, even if they
    # give the same clustering. We only check the labels up to a permutation.
    km = MiniBatchKMeans(n_clusters=10, init=init, n_init=10, random_state=0)
    # if algorithm is not None:
    #     km.set_params(algorithm=algorithm)
    km.fit(X)
    labels = km.labels_

    # re-predict labels for training set using predict
    pred = km.predict(X)
    assert_allclose(v_measure_score(pred, labels), 1)

    # re-predict labels for training set using fit_predict
    pred = km.fit_predict(X)
    assert_allclose(v_measure_score(pred, labels), 1)

    # predict centroid labels
    pred = km.predict(km.cluster_centers_)
    assert_allclose(v_measure_score(pred, np.arange(10)), 1)


@pytest.mark.skipif(MiniBatchKMeans is None, reason='scikit-learn no installed')
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("init", ["k-means++", "ndarray"])
def test_integer_input(setup, dtype, init):
    # Check that KMeans and MiniBatchKMeans work with integer input.
    X = np.array([[0, 0], [10, 10], [12, 9], [-1, 1], [2, 0], [8, 10]])
    n_init = 1 if init == "ndarray" else 10
    init = X[:2] if init == "ndarray" else init
    km = MiniBatchKMeans(n_clusters=2, batch_size=2, init=init, n_init=n_init, random_state=0)
    km.fit(X)

    # Internally integer input should be converted to float64
    assert km.cluster_centers_.dtype == np.float64
    expected_labels = [0, 1, 1, 0, 0, 1]
    assert_allclose(v_measure_score(km.labels_, expected_labels), 1)


@pytest.mark.skipif(MiniBatchKMeans is None, reason='scikit-learn no installed')
def test_transform(setup):
    # Check the transform method
    km = MiniBatchKMeans(n_clusters=n_clusters).fit(X)

    # Transorfming cluster_centers_ should return the pairwise distances
    # between centers
    Xt = km.transform(km.cluster_centers_)
    assert_allclose(Xt, pairwise_distances(km.cluster_centers_))
    # In particular, diagonal must be 0
    xt_diagonal = mt.diag(Xt)
    xt_diagonal.execute()
    assert_array_equal(xt_diagonal, np.zeros(n_clusters))

    # Transorfming X should return the pairwise distances between X and the centers
    Xt = km.transform(X)
    assert_allclose(Xt, pairwise_distances(X, km.cluster_centers_))


@pytest.mark.skipif(MiniBatchKMeans is None, reason='scikit-learn no installed')
def test_fit_transform(setup):
    # Check equivalence between fit.transform and fit_transform
    X1 = MiniBatchKMeans(random_state=0, n_init=1).fit(X).transform(X)
    X2 = MiniBatchKMeans(random_state=0, n_init=1).fit_transform(X)
    assert_allclose(X1, X2)


@pytest.mark.skipif(MiniBatchKMeans is None, reason='scikit-learn no installed')
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_centers_not_mutated(setup, dtype):
    # Check that KMeans and MiniBatchKMeans won't mutate the user provided
    # init centers silently even if input data and init centers have the same
    # type.
    X_new_type = X.astype(dtype, copy=False)
    centers_new_type = centers.astype(dtype, copy=False)
    km = MiniBatchKMeans(init=centers_new_type, n_clusters=n_clusters, n_init=1)
    km.fit(X_new_type)
    assert not np.may_share_memory(km.cluster_centers_.to_numpy(), centers_new_type)


@pytest.mark.skipif(MiniBatchKMeans is None, reason='scikit-learn no installed')
def test_unit_weights_vs_no_weights(setup):
    # Check that not passing sample weights should be equivalent to passing
    # sample weights all equal to one.
    sample_weight = np.ones(n_samples)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init=1)
    km_none = km.fit(X, sample_weight=None)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init=1)
    km_ones = km.fit(X, sample_weight=sample_weight)
    assert_array_equal(km_none.labels_, km_ones.labels_)


@pytest.mark.skipif(MiniBatchKMeans is None, reason='scikit-learn no installed')
def test_scaled_weights(setup):
    # scaling all sample weights by a common factor
    # shouldn't change the result
    sample_weight = np.ones(n_samples)
    km_1 = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    km_2 = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    km_1.fit(X)
    km_2.fit(X)
    assert_almost_equal(v_measure_score(km_1.labels_, km_2.labels_), 1.0)
