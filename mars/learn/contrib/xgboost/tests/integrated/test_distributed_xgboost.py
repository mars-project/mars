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
import unittest

import numpy as np

import mars.dataframe as md
import mars.tensor as mt
from mars.session import new_session
from mars.learn.contrib.xgboost import XGBClassifier
from mars.learn.tests.integrated.base import LearnIntegrationTestBase

try:
    import xgboost
except ImportError:
    xgboost = None


@unittest.skipIf(xgboost is None, 'xgboost not installed')
class Test(LearnIntegrationTestBase):
    def setUp(self):
        n_rows = 1000
        n_columns = 10
        chunk_size = 20
        rs = mt.random.RandomState(0)
        self.X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
        self.y = rs.rand(n_rows, chunk_size=chunk_size)
        super().setUp()

    def testDistributedXGBClassifier(self):
        service_ep = 'http://127.0.0.1:' + self.web_port
        timeout = 120 if 'CI' in os.environ else -1
        with new_session(service_ep) as sess:
            run_kwargs = {'timeout': timeout}

            X, y = self.X, self.y
            y = (y * 10).astype(mt.int32)
            classifier = XGBClassifier(verbosity=1, n_estimators=2)
            classifier.fit(X, y, eval_set=[(X, y)], session=sess, run_kwargs=run_kwargs)
            prediction = classifier.predict(X, session=sess, run_kwargs=run_kwargs)

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

            X = md.DataFrame(np.random.rand(100, 20), chunk_size=20)
            y = md.DataFrame(np.random.randint(0, 2, (100, 1)), chunk_size=20)
            classifier = XGBClassifier(verbosity=1, n_estimators=2)
            classifier.fit(X, y, session=sess, run_kwargs=run_kwargs)
            prediction = classifier.predict(X, session=sess, run_kwargs=run_kwargs)

            self.assertIsInstance(prediction, md.Series)
            self.assertEqual(prediction.shape[0], len(X))
