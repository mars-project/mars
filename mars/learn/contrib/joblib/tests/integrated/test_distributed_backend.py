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

import sys
import unittest

import numpy as np

from mars.learn.contrib.joblib import register_mars_backend
from mars.learn.tests.integrated.base import LearnIntegrationTestBase
import mars.tensor as mt

try:
    import joblib
    import sklearn
    from sklearn.datasets import load_digits
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.svm import SVC
except ImportError:
    joblib = sklearn = None


@unittest.skipIf(sys.platform == 'win32', "plasma don't support windows")
@unittest.skipIf(joblib is None, 'joblib not installed')
class Test(LearnIntegrationTestBase):
    def setUp(self):
        register_mars_backend()

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

    @unittest.skipIf(sklearn is None, 'scikit-learn not installed')
    def testSKLearnSVCTrain(self):
        digits = load_digits()
        param_space = {
            'C': np.logspace(-6, 6, 30),
            'gamma': np.logspace(-8, 8, 30),
            'tol': np.logspace(-4, -1, 30),
            'class_weight': [None, 'balanced'],
        }
        model = SVC(kernel='rbf')
        search = RandomizedSearchCV(model, param_space, cv=5, n_iter=10, verbose=10)

        service_ep = 'http://127.0.0.1:' + self.web_port
        with joblib.parallel_backend('mars', service=service_ep):
            search.fit(digits.data, digits.target)
