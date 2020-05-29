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

import numpy as np

from mars.session import new_session
from mars.tests.core import ExecutorForTest

try:
    import sklearn

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.utils._testing import assert_no_warnings, assert_warns
except ImportError:  # pragma: no cover
    sklearn = None

from mars import tensor as mt
from mars.learn.metrics.pairwise import rbf_kernel
from mars.learn.neighbors import NearestNeighbors
from mars.learn.semi_supervised import LabelPropagation


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.session = new_session().as_default()
        self._old_executor = self.session._sess._executor
        self.executor = self.session._sess._executor = \
            ExecutorForTest('numpy', storage=self.session._sess._context)

        self.estimators = [
            (LabelPropagation, {'kernel': 'rbf'}),
            (LabelPropagation, {'kernel': 'knn', 'n_neighbors': 2}),
            (LabelPropagation, {'kernel': lambda x, y: rbf_kernel(x, y, gamma=20)})
        ]

    def tearDown(self) -> None:
        self.session._sess._executor = self._old_executor

    def testFitTransduction(self):
        samples = [[1., 0.], [0., 2.], [1., 3.]]
        labels = [0, 1, -1]
        for estimator, parameters in self.estimators:
            clf = estimator(**parameters).fit(samples, labels)
            self.assertEqual(clf.transduction_[2].fetch(), 1)

    def testDistribution(self):
        samples = [[1., 0.], [0., 1.], [1., 1.]]
        labels = [0, 1, -1]
        for estimator, parameters in self.estimators:
            clf = estimator(**parameters).fit(samples, labels)
            if parameters['kernel'] == 'knn':
                continue    # unstable test; changes in k-NN ordering break it
            else:
                np.testing.assert_array_almost_equal(
                    np.asarray(clf.label_distributions_[2]),
                    np.array([.5, .5]), 2
                )

    def testPredict(self):
        samples = [[1., 0.], [0., 2.], [1., 3.]]
        labels = [0, 1, -1]
        for estimator, parameters in self.estimators:
            clf = estimator(**parameters).fit(samples, labels)
            np.testing.assert_array_equal(clf.predict([[0.5, 2.5]]).fetch(), np.array([1]))

    def testPredictProba(self):
        samples = [[1., 0.], [0., 1.], [1., 2.5]]
        labels = [0, 1, -1]
        for estimator, parameters in self.estimators:
            clf = estimator(**parameters).fit(samples, labels)
            np.testing.assert_almost_equal(clf.predict_proba([[1., 1.]]).fetch(),
                                           np.array([[0.5, 0.5]]))

    @unittest.skipIf(sklearn is None, 'sklearn not installed')
    def testLabelPropagationClosedForm(self):
        n_classes = 2
        X, y = make_classification(n_classes=n_classes, n_samples=200,
                                   random_state=0)
        y[::3] = -1
        Y = np.zeros((len(y), n_classes + 1))
        Y[np.arange(len(y)), y] = 1
        unlabelled_idx = Y[:, (-1,)].nonzero()[0]
        labelled_idx = (Y[:, (-1,)] == 0).nonzero()[0]

        clf = LabelPropagation(max_iter=10000, gamma=0.1)
        clf.fit(X, y)
        # adopting notation from Zhu et al 2002
        T_bar = clf._build_graph().to_numpy()
        Tuu = T_bar[tuple(np.meshgrid(unlabelled_idx, unlabelled_idx,
                                      indexing='ij'))]
        Tul = T_bar[tuple(np.meshgrid(unlabelled_idx, labelled_idx,
                                      indexing='ij'))]
        Y = Y[:, :-1]
        Y_l = Y[labelled_idx, :]
        Y_u = np.dot(np.dot(np.linalg.inv(np.eye(Tuu.shape[0]) - Tuu), Tul), Y_l)

        expected = Y.copy()
        expected[unlabelled_idx, :] = Y_u
        expected /= expected.sum(axis=1)[:, np.newaxis]

        np.testing.assert_array_almost_equal(
            expected, clf.label_distributions_.fetch(), 4)

    @unittest.skipIf(sklearn is None, 'sklearn not installed')
    def testConvergenceWarning(self):
        # This is a non-regression test for #5774
        X = np.array([[1., 0.], [0., 1.], [1., 2.5]])
        y = np.array([0, 1, -1])

        mdl = LabelPropagation(kernel='rbf', max_iter=1)
        assert_warns(ConvergenceWarning, mdl.fit, X, y)
        assert mdl.n_iter_ == mdl.max_iter

        mdl = LabelPropagation(kernel='rbf', max_iter=500)
        assert_no_warnings(mdl.fit, X, y)

    @unittest.skipIf(sklearn is None, 'sklearn not installed')
    def testPredictSparseCallableKernel(self):
        # This is a non-regression test for #15866

        # Custom sparse kernel (top-K RBF)
        def topk_rbf(X, Y=None, n_neighbors=10, gamma=1e-5):
            nn = NearestNeighbors(n_neighbors=10, metric='euclidean', n_jobs=-1)
            nn.fit(X)
            W = -1 * mt.power(nn.kneighbors_graph(Y, mode='distance'), 2) * gamma
            W = mt.exp(W)
            assert W.issparse()
            return W.T

        n_classes = 4
        n_samples = 500
        n_test = 10
        X, y = make_classification(n_classes=n_classes,
                                   n_samples=n_samples,
                                   n_features=20,
                                   n_informative=20,
                                   n_redundant=0,
                                   n_repeated=0,
                                   random_state=0)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=n_test,
                                                            random_state=0)

        model = LabelPropagation(kernel=topk_rbf)
        model.fit(X_train, y_train)
        assert model.score(X_test, y_test).fetch() >= 0.9
