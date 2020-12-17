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

import mars.tensor as mt
import mars.dataframe as md
from mars.session import new_session
from mars.tests.core import ExecutorForTest

try:
    import lightgbm
    from mars.learn.contrib.lightgbm import LGBMClassifier
    from mars.learn.contrib.lightgbm import predict, predict_proba
except ImportError:
    lightgbm = LGBMClassifier = predict = None


@unittest.skipIf(lightgbm is None, 'LightGBM not installed')
class Test(unittest.TestCase):
    def setUp(self):
        n_rows = 1000
        n_columns = 10
        chunk_size = 20
        rs = mt.random.RandomState(0)
        self.X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
        self.y = rs.rand(n_rows, chunk_size=chunk_size)
        self.X_df = md.DataFrame(self.X)
        x_sparse = np.random.rand(n_rows, n_columns)
        x_sparse[np.arange(n_rows), np.random.randint(n_columns, size=n_rows)] = np.nan
        self.X_sparse = mt.tensor(x_sparse, chunk_size=chunk_size).tosparse(missing=np.nan)

        self.session = new_session().as_default()
        self._old_executor = self.session._sess._executor
        self.executor = self.session._sess._executor = \
            ExecutorForTest('numpy', storage=self.session._sess._context)

    def tearDown(self) -> None:
        self.session._sess._executor = self._old_executor

    def testLocalClassifier(self):
        X, y = self.X, self.y
        y = (y * 10).astype(mt.int32)
        classifier = LGBMClassifier(n_estimators=2)
        classifier.fit(X, y, eval_set=[(X, y)], verbose=True)
        prediction = classifier.predict(X)

        self.assertEqual(prediction.ndim, 1)
        self.assertEqual(prediction.shape[0], len(self.X))

        self.assertIsInstance(prediction, mt.Tensor)

        # test sparse tensor
        X_sparse = self.X_sparse
        classifier = LGBMClassifier(n_estimators=2)
        classifier.fit(X_sparse, y, eval_set=[(X_sparse, y)], verbose=True)
        prediction = classifier.predict(X_sparse)

        self.assertEqual(prediction.ndim, 1)
        self.assertEqual(prediction.shape[0], len(self.X))

        self.assertIsInstance(prediction, mt.Tensor)

        prob = classifier.predict_proba(X)
        self.assertEqual(prob.shape, X.shape)

        prediction_empty = classifier.predict(mt.array([]).reshape((0, X.shape[1])))
        self.assertEqual(prediction_empty.shape, (0,))

        # test dataframe
        X_df = self.X_df
        classifier = LGBMClassifier(n_estimators=2)
        classifier.fit(X_df, y, verbose=True)
        prediction = classifier.predict(X_df)

        self.assertEqual(prediction.ndim, 1)
        self.assertEqual(prediction.shape[0], len(self.X))

        prob = classifier.predict_proba(X_df)

        self.assertEqual(prob.ndim, 2)
        self.assertEqual(prob.shape, (len(self.X), 10))

        # test weight
        weights = [mt.random.rand(X.shape[0]), md.Series(mt.random.rand(X.shape[0]))]
        y_df = md.DataFrame(y)
        for weight in weights:
            classifier = LGBMClassifier(n_estimators=2)
            classifier.fit(X, y_df, sample_weight=weight, verbose=True)
            prediction = classifier.predict(X)

            self.assertEqual(prediction.ndim, 1)
            self.assertEqual(prediction.shape[0], len(self.X))

        # should raise error if weight.ndim > 1
        with self.assertRaises(ValueError):
            LGBMClassifier(n_estimators=2).fit(
                X, y_df, sample_weight=mt.random.rand(1, 1), verbose=True)

        # test binary classifier
        new_y = (self.y > 0.5).astype(mt.int32)
        classifier = LGBMClassifier(n_estimators=2)
        classifier.fit(X, new_y, verbose=True)

        prediction = classifier.predict(X)
        self.assertEqual(prediction.ndim, 1)
        self.assertEqual(prediction.shape[0], len(self.X))

        prediction = classifier.predict_proba(X)
        self.assertEqual(prediction.ndim, 2)
        self.assertEqual(prediction.shape[0], len(self.X))

        # test with existing model
        classifier = lightgbm.LGBMClassifier(n_estimators=2)
        classifier.fit(X, new_y, verbose=True)

        label_result = predict(classifier, X_df)
        self.assertEqual(label_result.ndim, 1)
        self.assertEqual(label_result.shape[0], len(self.X))

        proba_result = predict_proba(classifier, X_df)
        self.assertEqual(proba_result.ndim, 2)
        self.assertEqual(proba_result.shape[0], len(self.X))
