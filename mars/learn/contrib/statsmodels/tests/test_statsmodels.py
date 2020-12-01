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
from mars.session import new_session
from mars.tests.core import ExecutorForTest

try:
    import statsmodels
    from mars.learn.contrib.statsmodels import MarsDistributedModel, MarsResults
except ImportError:  # pragma: no cover
    statsmodels = MarsDistributedModel = MarsResults = None


@unittest.skipIf(statsmodels is None, 'statsmodels not installed')
class Test(unittest.TestCase):
    def setUp(self):
        n_rows = 1000
        n_columns = 10
        chunk_size = 20
        rs = mt.random.RandomState(0)
        self.X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
        self.y = rs.rand(n_rows, chunk_size=chunk_size)
        self.X_df = md.DataFrame(self.X, chunk_size=(100, 5))
        self.y_s = md.Series(self.y, chunk_size=100)

        self.session = new_session().as_default()
        self._old_executor = self.session._sess._executor
        self.executor = self.session._sess._executor = \
            ExecutorForTest('numpy', storage=self.session._sess._context)

    def testLocalTrainPredict(self):
        model = MarsDistributedModel(num_partitions=10)
        result = model.fit(self.y, self.X, alpha=0.2, session=self.session)
        self.assertIsInstance(result, MarsResults)
        predict_tensor = result.predict(self.X, session=self.session)
        predicted = predict_tensor.fetch(session=self.session)
        self.assertEqual(predicted.shape, self.y.shape)

        model = MarsDistributedModel(num_partitions=10)
        result = model.fit(self.y_s, self.X_df, alpha=0.2, session=self.session)
        self.assertIsInstance(result, MarsResults)
        predict_tensor = result.predict(self.X_df, session=self.session)
        predicted = predict_tensor.fetch(session=self.session)
        self.assertEqual(predicted.shape, self.y.shape)
