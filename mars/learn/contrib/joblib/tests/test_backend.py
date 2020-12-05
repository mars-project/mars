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

import numpy as np

from mars.session import new_session
from mars.tests.core import ExecutorForTest
from mars.learn.contrib.joblib import register_mars_backend

try:
    import joblib
    import sklearn
    from sklearn.datasets import load_digits
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.svm import SVC
except ImportError:
    joblib = sklearn = None


@unittest.skipIf(joblib is None, 'joblib not installed')
class Test(unittest.TestCase):
    def setUp(self):
        register_mars_backend()

        self.session = new_session().as_default()
        self._old_executor = self.session._sess._executor
        self.executor = self.session._sess._executor = \
            ExecutorForTest('numpy', storage=self.session._sess._context)

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

        with joblib.parallel_backend('mars', n_parallel=16):
            search.fit(digits.data, digits.target)
