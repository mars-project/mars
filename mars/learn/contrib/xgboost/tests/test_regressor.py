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
from mars.learn.contrib.xgboost import XGBRegressor
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

    def testLocalRegressor(self):
        new_session().as_default()

        X, y = self.X, self.y
        regressor = XGBRegressor(verbosity=1, n_estimators=2)
        regressor.set_params(tree_method='hist')
        regressor.fit(X, y, eval_set=[(X, y)])
        prediction = regressor.predict(X)

        self.assertEqual(prediction.ndim, 1)
        self.assertEqual(prediction.shape[0], len(self.X))

        history = regressor.evals_result()

        self.assertIsInstance(prediction, mt.Tensor)
        self.assertIsInstance(history, dict)

        self.assertEqual(list(history['validation_0'])[0], 'rmse')
        self.assertEqual(len(history['validation_0']['rmse']), 2)

        # test weight
        weight = mt.random.rand(X.shape[0])
        classifier = XGBRegressor(verbosity=1, n_estimators=2)
        regressor.set_params(tree_method='hist')
        classifier.fit(X, y, sample_weights=weight)
        prediction = classifier.predict(X)

        self.assertEqual(prediction.ndim, 1)
        self.assertEqual(prediction.shape[0], len(self.X))

        # test wrong params
        regressor = XGBRegressor(verbosity=1, n_estimators=2)
        with self.assertRaises(TypeError):
            regressor.fit(X, y, wrong_param=1)
        regressor.fit(X, y)
        with self.assertRaises(TypeError):
            regressor.predict(X, wrong_param=1)
