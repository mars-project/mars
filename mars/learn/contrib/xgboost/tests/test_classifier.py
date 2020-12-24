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

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import mars.tensor as mt
import mars.dataframe as md
from mars.session import new_session
from mars.learn.contrib.xgboost import XGBClassifier
from mars.tests.core import ExecutorForTest

try:
    import xgboost
except ImportError:
    xgboost = None


@unittest.skipIf(xgboost is None, 'XGBoost not installed')
class Test(unittest.TestCase):
    def setUp(self):
        n_rows = 1000
        n_columns = 10
        chunk_size = 20
        rs = mt.random.RandomState(0)
        self.X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
        self.y = rs.rand(n_rows, chunk_size=chunk_size)
        self.X_df = md.DataFrame(self.X)

        self.session = new_session().as_default()
        self._old_executor = self.session._sess._executor
        self.executor = self.session._sess._executor = \
            ExecutorForTest('numpy', storage=self.session._sess._context)

    def tearDown(self) -> None:
        self.session._sess._executor = self._old_executor

    def testLocalClassifier(self):
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
        # default metrics may differ, see https://github.com/dmlc/xgboost/pull/6183
        eval_metric = list(history['validation_0'])[0]
        self.assertIn(eval_metric, ('merror', 'mlogloss'))
        self.assertEqual(len(history['validation_0']), 1)
        self.assertEqual(len(history['validation_0'][eval_metric]), 2)

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

        # test predict data with unknown shape
        X2 = X[X[:, 0] > 0.1].astype(mt.int32)
        prediction = classifier.predict(X2)

        self.assertEqual(prediction.ndim, 1)

        classifier = XGBClassifier(verbosity=1, n_estimators=2)
        with self.assertRaises(TypeError):
            classifier.fit(X, y, wrong_param=1)
        classifier.fit(X, y)
        with self.assertRaises(TypeError):
            classifier.predict(X, wrong_param=1)

    def testLocalClassifierFromToParquet(self):
        n_rows = 1000
        n_columns = 10
        rs = np.random.RandomState(0)
        X = rs.rand(n_rows, n_columns)
        y = rs.rand(n_rows)
        df = pd.DataFrame(X, columns=[f'c{i}' for i in range(n_columns)])
        df['id'] = [f'i{i}' for i in range(n_rows)]

        booster = xgboost.train({}, xgboost.DMatrix(X, y),
                                num_boost_round=2)

        with tempfile.TemporaryDirectory() as d:
            m_name = os.path.join(d, 'c.model')
            f_name = os.path.join(d, 'data.parquet')
            r_name = os.path.join(d, 'result.parquet')

            booster.save_model(m_name)

            df.to_parquet(f_name)

            df = md.read_parquet(f_name).set_index('id')
            model = XGBClassifier()
            model.load_model(m_name)
            result = model.predict(df, run=False)
            r = md.DataFrame(result).to_parquet(r_name)

            # tiles to ensure no iterative tiling exists
            r.tiles()
            r.execute()

            ret = pd.read_parquet(r_name).iloc[: 0].to_numpy()
            model2 = xgboost.XGBClassifier()
            model2.load_model(m_name)
            expected = model2.predict(X)
            expected = np.stack([expected, 1 - expected]).argmax()
            np.testing.assert_array_equal(ret, expected)
