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

from mars.tests.integrated.base import IntegrationTestBase
import mars.tensor as mt
from mars.session import new_session

try:
    import statsmodels
    from mars.learn.contrib.statsmodels import MarsDistributedModel, MarsResults
except ImportError:  # pragma: no cover
    statsmodels = MarsDistributedModel = MarsResults = None


@unittest.skipIf(statsmodels is None, 'statsmodels not installed')
class Test(IntegrationTestBase):
    def setUp(self):
        n_rows = 1000
        n_columns = 10
        chunk_size = 20
        rs = mt.random.RandomState(0)
        X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
        y = rs.rand(n_rows, chunk_size=chunk_size)
        filter = rs.rand(n_rows, chunk_size=chunk_size) < 0.8
        self.X = X[filter]
        self.y = y[filter]
        super().setUp()

    def testDistributedStatsModels(self):
        service_ep = 'http://127.0.0.1:' + self.web_port
        timeout = 120 if 'CI' in os.environ else -1
        with new_session(service_ep) as sess:
            run_kwargs = {'timeout': timeout}

            X, y = self.X, self.y
            y = (y * 10).astype(mt.int32)
            model = MarsDistributedModel(factor=1.2)
            result = model.fit(y, X, alpha=0.2, session=sess, run_kwargs=run_kwargs)
            prediction = result.predict(X, session=sess, run_kwargs=run_kwargs)

            X.execute(session=sess)

            self.assertEqual(prediction.ndim, 1)
            self.assertEqual(prediction.shape[0], len(self.X))
