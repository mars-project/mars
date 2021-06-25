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
import pytest

import mars.tensor as mt
import mars.dataframe as md
from mars.learn.contrib.xgboost import train, MarsDMatrix
from mars.tests import setup

try:
    import xgboost
    from xgboost import Booster
except ImportError:
    xgboost = None

setup = setup

n_rows = 1000
n_columns = 10
chunk_size = 200
rs = mt.random.RandomState(0)
X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
y = rs.rand(n_rows, chunk_size=chunk_size)
X_df = md.DataFrame(X)
y_series = md.Series(y)
x_sparse = np.random.rand(n_rows, n_columns)
x_sparse[np.arange(n_rows), np.random.randint(n_columns, size=n_rows)] = np.nan
X_sparse = mt.tensor(x_sparse, chunk_size=chunk_size).tosparse(missing=np.nan)


@pytest.mark.skipif(xgboost is None, reason='XGBoost not installed')
def test_local_train_tensor(setup):
    dtrain = MarsDMatrix(X, y)
    booster = train({}, dtrain, num_boost_round=2)
    assert isinstance(booster, Booster)


@pytest.mark.skipif(xgboost is None, reason='XGBoost not installed')
def test_local_train_sparse_tensor(setup):
    dtrain = MarsDMatrix(X_sparse, y)
    booster = train({}, dtrain, num_boost_round=2)
    assert isinstance(booster, Booster)


@pytest.mark.skipif(xgboost is None, reason='XGBoost not installed')
def test_local_train_dataframe(setup):
    dtrain = MarsDMatrix(X_df, y_series)
    booster = train({}, dtrain, num_boost_round=2)
    assert isinstance(booster, Booster)
