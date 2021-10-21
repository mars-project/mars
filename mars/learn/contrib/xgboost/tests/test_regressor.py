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

import pytest

try:
    import xgboost
except ImportError:
    xgboost = None

from ..... import tensor as mt
from ..regressor import XGBRegressor

n_rows = 1000
n_columns = 10
chunk_size = 200
rs = mt.random.RandomState(0)
X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
y = rs.rand(n_rows, chunk_size=chunk_size)


@pytest.mark.skipif(xgboost is None, reason="XGBoost not installed")
def test_local_regressor(setup):
    regressor = XGBRegressor(verbosity=1, n_estimators=2)
    regressor.set_params(tree_method="hist")
    regressor.fit(X, y, eval_set=[(X, y)])
    prediction = regressor.predict(X)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X)

    history = regressor.evals_result()

    assert isinstance(prediction, mt.Tensor)
    assert isinstance(history, dict)

    assert list(history["validation_0"])[0] == "rmse"
    assert len(history["validation_0"]["rmse"]) == 2

    # test weight
    weight = mt.random.rand(X.shape[0])
    classifier = XGBRegressor(verbosity=1, n_estimators=2)
    regressor.set_params(tree_method="hist")
    classifier.fit(X, y, sample_weight=weight)
    prediction = classifier.predict(X)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X)

    # test wrong params
    regressor = XGBRegressor(verbosity=1, n_estimators=2)
    with pytest.raises(TypeError):
        regressor.fit(X, y, wrong_param=1)
    regressor.fit(X, y)
    with pytest.raises(TypeError):
        regressor.predict(X, wrong_param=1)
