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
from mars.tests.integrated.base import IntegrationTestBase
import mars.tensor as mt

try:
    import joblib
    import sklearn
    from sklearn.datasets import load_digits
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.svm import SVC
except ImportError:
    joblib = sklearn = None


@pytest.mark.skipif(sys.platform == 'win32', reason="plasma don't support windows")
@pytest.mark.skipif(joblib is None, reason='joblib not installed')
class Test(IntegrationTestBase):
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
    
    
    @pytest.mark.skipif(sklearn is None, reason='scikit-learn not installed')
    def test_sk_learn_svc_train():
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
