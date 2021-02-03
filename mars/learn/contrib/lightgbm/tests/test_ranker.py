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
from mars.session import new_session
from mars.tests.core import ExecutorForTest

try:
    import lightgbm
    from mars.learn.contrib.lightgbm import LGBMRanker
except ImportError:
    lightgbm = LGBMRanker = None


@unittest.skipIf(lightgbm is None, 'LightGBM not installed')
class Test(unittest.TestCase):
    def setUp(self):
        n_rows = 1000
        n_columns = 10
        chunk_size = 20
        rs = mt.random.RandomState(0)
        self.X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
        self.y = rs.rand(n_rows, chunk_size=chunk_size)

        self.session = new_session().as_default()
        self._old_executor = self.session._sess._executor
        self.executor = self.session._sess._executor = \
            ExecutorForTest('numpy', storage=self.session._sess._context)

    def tearDown(self) -> None:
        self.session._sess._executor = self._old_executor

    def testLocalRanker(self):
        X, y = self.X, self.y
        y = (y * 10).astype(mt.int32)
        ranker = LGBMRanker(n_estimators=2)
        ranker.fit(X, y, group=[X.shape[0]], verbose=True)
        prediction = ranker.predict(X)

        self.assertEqual(prediction.ndim, 1)
        self.assertEqual(prediction.shape[0], len(self.X))

        self.assertIsInstance(prediction, mt.Tensor)
        result = prediction.fetch()
        self.assertEqual(prediction.dtype, result.dtype)

        # test weight
        weight = mt.random.rand(X.shape[0])
        ranker = LGBMRanker(verbosity=1, n_estimators=2)
        ranker.fit(X, y, group=[X.shape[0]], sample_weight=weight)
        prediction = ranker.predict(X)

        self.assertEqual(prediction.ndim, 1)
        self.assertEqual(prediction.shape[0], len(self.X))
        result = prediction.fetch()
        self.assertEqual(prediction.dtype, result.dtype)

        # test local model
        X_np = X.execute(session=self.session).fetch(session=self.session)
        y_np = y.execute(session=self.session).fetch(session=self.session)
        raw_ranker = lightgbm.LGBMRanker(verbosity=1, n_estimators=2)
        raw_ranker.fit(X_np, y_np, group=[X.shape[0]])
        prediction = LGBMRanker(raw_ranker).predict(X)

        self.assertEqual(prediction.ndim, 1)
        self.assertEqual(prediction.shape[0], len(self.X))
