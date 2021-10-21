# Copyright 1999-2021 Alibaba Group Holding Ltd.
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

import numpy as np
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

from .. import register_mars_backend

register_mars_backend()


def test_sk_learn_svc_train(setup):
    digits = load_digits()
    param_space = {
        "C": np.logspace(-6, 6, 30),
        "gamma": np.logspace(-8, 8, 30),
        "tol": np.logspace(-4, -1, 30),
        "class_weight": [None, "balanced"],
    }
    model = SVC(kernel="rbf")
    search = RandomizedSearchCV(model, param_space, cv=5, n_iter=5, verbose=10)

    with joblib.parallel_backend("mars", n_parallel=16):
        search.fit(digits.data, digits.target)
