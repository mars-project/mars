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
from mars.operands import Fuse
from mars.session import new_session
from mars.tests.core import ExecutorForTest

try:
    import lightgbm
    from mars.learn.contrib.lightgbm import LGBMClassifier
except ImportError:
    lightgbm = LGBMClassifier = None


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
        X_np = X.execute(session=self.session).fetch(session=self.session)
        new_y_np = new_y.execute(session=self.session).fetch(session=self.session)
        raw_classifier = lightgbm.LGBMClassifier(n_estimators=2)
        raw_classifier.fit(X_np, new_y_np, verbose=True)

        classifier = LGBMClassifier(raw_classifier)
        label_result = classifier.predict(X_df)
        self.assertEqual(label_result.ndim, 1)
        self.assertEqual(label_result.shape[0], len(self.X))

        proba_result = classifier.predict_proba(X_df)
        self.assertEqual(proba_result.ndim, 2)
        self.assertEqual(proba_result.shape[0], len(self.X))

    def testLocalClassifierFromToParquet(self):
        n_rows = 1000
        n_columns = 10
        rs = np.random.RandomState(0)
        X = rs.rand(n_rows, n_columns)
        y = (rs.rand(n_rows) > 0.5).astype(np.int32)
        df = pd.DataFrame(X, columns=[f'c{i}' for i in range(n_columns)])

        # test with existing model
        classifier = lightgbm.LGBMClassifier(n_estimators=2)
        classifier.fit(X, y, verbose=True)

        with tempfile.TemporaryDirectory() as d:
            result_dir = os.path.join(d, 'result')
            os.mkdir(result_dir)
            data_dir = os.path.join(d, 'data')
            os.mkdir(data_dir)

            df.iloc[:500].to_parquet(os.path.join(d, 'data', 'data1.parquet'))
            df.iloc[500:].to_parquet(os.path.join(d, 'data', 'data2.parquet'))

            df = md.read_parquet(data_dir)
            model = LGBMClassifier()
            model.load_model(classifier)
            result = model.predict(df, run=False)
            r = md.DataFrame(result).to_parquet(result_dir)

            # tiles to ensure no iterative tiling exists
            g = r.build_graph(tiled=True)
            self.assertTrue(all(isinstance(n.op, Fuse) for n in g))
            self.assertEqual(len(g), 2)
            r.execute()

            ret = md.read_parquet(result_dir).to_pandas().iloc[:, 0].to_numpy()
            expected = classifier.predict(X)
            expected = np.stack([1 - expected, expected]).argmax(axis=0)
            np.testing.assert_array_equal(ret, expected)
