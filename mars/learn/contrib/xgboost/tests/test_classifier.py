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

import mars.tensor as mt
import mars.dataframe as md
from mars.learn.contrib.xgboost import XGBClassifier
from mars.session import new_session
from mars.tests.core import aio_case

try:
    import xgboost
except ImportError:
    xgboost = None


@unittest.skipIf(xgboost is None, 'XGBoost not installed')
@aio_case
class Test(unittest.TestCase):
    def setUp(self):
        n_rows = 1000
        n_columns = 10
        chunk_size = 20
        rs = mt.random.RandomState(0)
        self.X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
        self.y = rs.rand(n_rows, chunk_size=chunk_size)
        self.X_df = md.DataFrame(self.X)

    def testLocalClassifier(self):
        new_session().as_default()

        X, y = self.X, self.y
        y = (y * 10).astype(mt.int32)
        classifier = XGBClassifier(verbosity=1, n_estimators=2)
        classifier.fit(X, y, eval_set=[(X, y)])
        prediction = classifier.predict(X)

        self.assertEqual(prediction.ndim, 1)
        self.assertEqual(prediction.shape[0], len(self.X))

        history = classifier.evals_result()

        self.assertIsInstance(prediction, mt.Tensor)
        self.assertIsInstance(history, dict)

        self.assertEqual(list(history)[0], 'validation_0')
        self.assertEqual(list(history['validation_0'])[0], 'merror')
        self.assertEqual(len(history['validation_0']), 1)
        self.assertEqual(len(history['validation_0']['merror']), 2)

        prob = classifier.predict_proba(X)
        self.assertEqual(prob.shape, X.shape)

        # test dataframe
        X_df = self.X_df
        classifier = XGBClassifier(verbosity=1, n_estimators=2)
        classifier.fit(X_df, y)
        prediction = classifier.predict(X_df)

        self.assertEqual(prediction.ndim, 1)
        self.assertEqual(prediction.shape[0], len(self.X))

        # test weight
        weights = [mt.random.rand(X.shape[0]), md.Series(mt.random.rand(X.shape[0])),
                   md.DataFrame(mt.random.rand(X.shape[0]))]
        y_df = md.DataFrame(self.y)
        for weight in weights:
            classifier = XGBClassifier(verbosity=1, n_estimators=2)
            classifier.fit(X, y_df, sample_weights=weight)
            prediction = classifier.predict(X)

            self.assertEqual(prediction.ndim, 1)
            self.assertEqual(prediction.shape[0], len(self.X))

        # should raise error if weight.ndim > 1
        with self.assertRaises(ValueError):
            XGBClassifier(verbosity=1, n_estimators=2).fit(
                X, y_df, sample_weights=mt.random.rand(1, 1))

        # test binary classifier
        new_y = (self.y > 0.5).astype(mt.int32)
        classifier = XGBClassifier(verbosity=1, n_estimators=2)
        classifier.fit(X, new_y)
        prediction = classifier.predict(X)

        self.assertEqual(prediction.ndim, 1)
        self.assertEqual(prediction.shape[0], len(self.X))

        classifier = XGBClassifier(verbosity=1, n_estimators=2)
        with self.assertRaises(TypeError):
            classifier.fit(X, y, wrong_param=1)
        classifier.fit(X, y)
        with self.assertRaises(TypeError):
            classifier.predict(X, wrong_param=1)
