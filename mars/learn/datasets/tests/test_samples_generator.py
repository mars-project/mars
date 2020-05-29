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

import unittest
from functools import partial
from collections import defaultdict

import numpy as np

from mars.session import new_session
from mars.tests.core import ExecutorForTest

try:
    import sklearn
    from sklearn.utils._testing import assert_array_almost_equal, assert_raises, \
        assert_almost_equal, assert_raise_message
except ImportError:  # pragma: no cover
    sklearn = None

from mars.learn.datasets.samples_generator import make_low_rank_matrix, \
    make_classification, make_blobs
from mars.tensor.linalg import svd
from mars import tensor as mt


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.session = new_session().as_default()
        self._old_executor = self.session._sess._executor
        self.executor = self.session._sess._executor = \
            ExecutorForTest('numpy', storage=self.session._sess._context)

    def tearDown(self) -> None:
        self.session._sess._executor = self._old_executor

    @unittest.skipIf(sklearn is None, 'sklearn not installed')
    def testMakeClassification(self):
        weights = [0.1, 0.25]
        X, y = make_classification(n_samples=100, n_features=20, n_informative=5,
                                   n_redundant=1, n_repeated=1, n_classes=3,
                                   n_clusters_per_class=1, hypercube=False,
                                   shift=None, scale=None, weights=weights,
                                   random_state=0, flip_y=-1)

        self.assertEqual(weights, [0.1, 0.25])
        self.assertEqual(X.shape, (100, 20), "X shape mismatch")
        self.assertEqual(y.shape, (100,), "y shape mismatch")
        self.assertEqual(mt.unique(y).to_numpy().shape, (3,), "Unexpected number of classes")
        self.assertEqual((y == 0).sum().to_numpy(), 10, "Unexpected number of samples in class #0")
        self.assertEqual((y == 1).sum().to_numpy(), 25, "Unexpected number of samples in class #1")
        self.assertEqual((y == 2).sum().to_numpy(), 65, "Unexpected number of samples in class #2")

        # Test for n_features > 30
        X, y = make_classification(n_samples=2000, n_features=31, n_informative=31,
                                   n_redundant=0, n_repeated=0, hypercube=True,
                                   scale=0.5, random_state=0)

        X = X.to_numpy()
        self.assertEqual(X.shape, (2000, 31), "X shape mismatch")
        self.assertEqual(y.shape, (2000,), "y shape mismatch")
        self.assertEqual(np.unique(X.view([('', X.dtype)]*X.shape[1])).view(X.dtype)
                         .reshape(-1, X.shape[1]).shape[0], 2000,
                         "Unexpected number of unique rows")

    @unittest.skipIf(sklearn is None, 'sklearn not installed')
    def testMakeClassificationInformativeFeatures(self):
        """Test the construction of informative features in make_classification

        Also tests `n_clusters_per_class`, `n_classes`, `hypercube` and
        fully-specified `weights`.
        """
        # Create very separate clusters; check that vertices are unique and
        # correspond to classes
        class_sep = 1e6
        make = partial(make_classification, class_sep=class_sep, n_redundant=0,
                       n_repeated=0, flip_y=0, shift=0, scale=1, shuffle=False)

        for n_informative, weights, n_clusters_per_class in [(2, [1], 1),
                                                             (2, [1/3] * 3, 1),
                                                             (2, [1/4] * 4, 1),
                                                             (2, [1/2] * 2, 2),
                                                             (2, [3/4, 1/4], 2),
                                                             (10, [1/3] * 3, 10),
                                                             (np.int(64), [1], 1)
                                                             ]:
            n_classes = len(weights)
            n_clusters = n_classes * n_clusters_per_class
            n_samples = n_clusters * 50

            for hypercube in (False, True):
                generated = make(n_samples=n_samples, n_classes=n_classes,
                                 weights=weights, n_features=n_informative,
                                 n_informative=n_informative,
                                 n_clusters_per_class=n_clusters_per_class,
                                 hypercube=hypercube, random_state=0)

                X, y = mt.ExecutableTuple(generated).execute().fetch()
                self.assertEqual(X.shape, (n_samples, n_informative))
                self.assertEqual(y.shape, (n_samples,))

                # Cluster by sign, viewed as strings to allow uniquing
                signs = np.sign(X)
                signs = signs.view(dtype='|S{0}'.format(signs.strides[0]))
                unique_signs, cluster_index = np.unique(signs,
                                                        return_inverse=True)

                self.assertEqual(len(unique_signs), n_clusters,
                                 "Wrong number of clusters, or not in distinct "
                                 "quadrants")

                clusters_by_class = defaultdict(set)
                for cluster, cls in zip(cluster_index, y):
                    clusters_by_class[cls].add(cluster)
                for clusters in clusters_by_class.values():
                    self.assertEqual(len(clusters), n_clusters_per_class,
                                     "Wrong number of clusters per class")
                self.assertEqual(len(clusters_by_class), n_classes,
                                 "Wrong number of classes")

                assert_array_almost_equal(np.bincount(y) / len(y) // weights,
                                          [1] * n_classes,
                                          err_msg="Wrong number of samples "
                                                  "per class")

                # Ensure on vertices of hypercube
                for cluster in range(len(unique_signs)):
                    centroid = X[cluster_index == cluster].mean(axis=0)
                    if hypercube:
                        assert_array_almost_equal(np.abs(centroid) / class_sep,
                                                  np.ones(n_informative),
                                                  decimal=5,
                                                  err_msg="Clusters are not "
                                                          "centered on hypercube "
                                                          "vertices")
                    else:
                        assert_raises(AssertionError,
                                      assert_array_almost_equal,
                                      np.abs(centroid) / class_sep,
                                      np.ones(n_informative),
                                      decimal=5,
                                      err_msg="Clusters should not be centered "
                                              "on hypercube vertices")

        assert_raises(ValueError, make, n_features=2, n_informative=2, n_classes=5,
                      n_clusters_per_class=1)
        assert_raises(ValueError, make, n_features=2, n_informative=2, n_classes=3,
                      n_clusters_per_class=2)

    @unittest.skipIf(sklearn is None, 'sklearn not installed')
    def testMakeBlobs(self):
        cluster_stds = np.array([0.05, 0.2, 0.4])
        cluster_centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        X, y = make_blobs(random_state=0, n_samples=50, n_features=2,
                          centers=cluster_centers, cluster_std=cluster_stds)
        X, y = mt.ExecutableTuple((X, y)).execute().fetch()
        self.assertEqual(X.shape, (50, 2), "X shape mismatch")
        self.assertEqual(y.shape, (50,), "y shape mismatch")
        self.assertEqual(np.unique(y).shape, (3,), "Unexpected number of blobs")
        for i, (ctr, std) in enumerate(zip(cluster_centers, cluster_stds)):
            assert_almost_equal((X[y == i] - ctr).std(), std, 1, "Unexpected std")

    def testMakeBlobsNSamplesList(self):
        n_samples = [50, 30, 20]
        X, y = make_blobs(n_samples=n_samples, n_features=2, random_state=0)
        X, y = mt.ExecutableTuple((X, y)).execute().fetch()

        self.assertEqual(X.shape, (sum(n_samples), 2), "X shape mismatch")
        self.assertTrue(all(np.bincount(y, minlength=len(n_samples)) == n_samples),
                        "Incorrect number of samples per blob")

    @unittest.skipIf(sklearn is None, 'sklearn not installed')
    def testMakeBlobsNSamplesListWithCenters(self):
        n_samples = [20, 20, 20]
        centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        cluster_stds = np.array([0.05, 0.2, 0.4])
        X, y = make_blobs(n_samples=n_samples, centers=centers,
                          cluster_std=cluster_stds, random_state=0)
        X, y = mt.ExecutableTuple((X, y)).execute().fetch()

        self.assertEqual(X.shape, (sum(n_samples), 2), "X shape mismatch")
        self.assertTrue(all(np.bincount(y, minlength=len(n_samples)) == n_samples),
                        "Incorrect number of samples per blob")
        for i, (ctr, std) in enumerate(zip(centers, cluster_stds)):
            assert_almost_equal((X[y == i] - ctr).std(), std, 1, "Unexpected std")

    def testMakeBlobsNSamplesCentersNone(self):
        for n_samples in [[5, 3, 0], np.array([5, 3, 0]), tuple([5, 3, 0])]:
            centers = None
            X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=0)
            X, y = mt.ExecutableTuple((X, y)).execute().fetch()

            self.assertEqual(X.shape, (sum(n_samples), 2), "X shape mismatch")
            self.assertTrue(all(np.bincount(y, minlength=len(n_samples)) == n_samples),
                            "Incorrect number of samples per blob")

    @unittest.skipIf(sklearn is None, 'sklearn not installed')
    def testMakeBlobsError(self):
        n_samples = [20, 20, 20]
        centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        cluster_stds = np.array([0.05, 0.2, 0.4])
        wrong_centers_msg = ("Length of `n_samples` not consistent "
                             "with number of centers. Got n_samples = {} "
                             "and centers = {}".format(n_samples, centers[:-1]))
        assert_raise_message(ValueError, wrong_centers_msg,
                             make_blobs, n_samples, centers=centers[:-1])
        wrong_std_msg = ("Length of `clusters_std` not consistent with "
                         "number of centers. Got centers = {} "
                         "and cluster_std = {}".format(mt.tensor(centers), cluster_stds[:-1]))
        assert_raise_message(ValueError, wrong_std_msg,
                             make_blobs, n_samples,
                             centers=centers, cluster_std=cluster_stds[:-1])
        wrong_type_msg = ("Parameter `centers` must be array-like. "
                          "Got {!r} instead".format(3))
        assert_raise_message(ValueError, wrong_type_msg,
                             make_blobs, n_samples, centers=3)

    def testMakeLowRankMatrix(self):
        X = make_low_rank_matrix(n_samples=50, n_features=25, effective_rank=5,
                                 tail_strength=0.01, random_state=0)

        self.assertEqual(X.shape, (50, 25), "X shape mismatch")

        _, s, _ = svd(X)
        self.assertLess((s.sum() - 5).to_numpy(n_parallel=1), 0.1, "X rank is not approximately 5")
