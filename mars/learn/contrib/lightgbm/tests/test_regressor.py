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

import pandas as pd

import mars.tensor as mt
from mars.session import new_session
from mars.tests.core import ExecutorForTest

try:
    import lightgbm
    from mars.learn.contrib.lightgbm import LGBMRegressor
except ImportError:
    lightgbm = LGBMRegressor = None


@unittest.skipIf(lightgbm is None, 'LightGBM not installed')
class Test(unittest.TestCase):
    def setUp(self):
        n_rows = 1000
        n_columns = 10
        chunk_size = 20
        rs = mt.random.RandomState(0)
        self.X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
        self.y = rs.randint(0, 10, n_rows, chunk_size=chunk_size)

        self.session = new_session().as_default()
        self._old_executor = self.session._sess._executor
        self.executor = self.session._sess._executor = \
            ExecutorForTest('numpy', storage=self.session._sess._context)

    def tearDown(self) -> None:
        self.session._sess._executor = self._old_executor

    def testLocalRegressor(self):
        X, y = self.X, self.y
        regressor = LGBMRegressor(n_estimators=2)
        regressor.fit(X, y, verbose=True)
        prediction = regressor.predict(X)

        self.assertEqual(prediction.ndim, 1)
        self.assertEqual(prediction.shape[0], len(self.X))

        self.assertIsInstance(prediction, mt.Tensor)
        result = prediction.fetch()
        self.assertEqual(prediction.dtype, result.dtype)

        # test weight
        weight = mt.random.rand(X.shape[0])
        regressor = LGBMRegressor(verbosity=1, n_estimators=2)
        regressor.fit(X, y, sample_weight=weight)
        prediction = regressor.predict(X)

        self.assertEqual(prediction.ndim, 1)
        self.assertEqual(prediction.shape[0], len(self.X))
        result = prediction.fetch()
        self.assertEqual(prediction.dtype, result.dtype)

        # test numpy tensor
        try:
            from sklearn.datasets import make_classification
            X_array, y_array = make_classification()
            regressor = LGBMRegressor(n_estimators=2)
            regressor.fit(X_array, y_array, verbose=True)
            prediction = regressor.predict(X_array)

            self.assertEqual(prediction.ndim, 1)
            self.assertEqual(prediction.shape[0], len(X_array))

            X_df = pd.DataFrame(X_array)
            y_df = pd.Series(y_array)
            regressor = LGBMRegressor(n_estimators=2)
            regressor.fit(X_df, y_df, verbose=True)
            prediction = regressor.predict(X_df)

            self.assertEqual(prediction.ndim, 1)
            self.assertEqual(prediction.shape[0], len(X_df))
        except ImportError:
            pass

        # test existing model
        X_np = X.execute(session=self.session).fetch(session=self.session)
        y_np = y.execute(session=self.session).fetch(session=self.session)
        raw_regressor = lightgbm.LGBMRegressor(verbosity=1, n_estimators=2)
        raw_regressor.fit(X_np, y_np)
        prediction = LGBMRegressor(raw_regressor).predict(X)

        self.assertEqual(prediction.ndim, 1)
        self.assertEqual(prediction.shape[0], len(self.X))
