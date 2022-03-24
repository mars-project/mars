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

try:
    import xgboost
    from xgboost import Booster
except ImportError:
    xgboost = None
from ..... import tensor as mt
from ..... import dataframe as md
from .....tests.core import require_ray
from .. import train, MarsDMatrix

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


@pytest.mark.skipif(xgboost is None, reason="XGBoost not installed")
def test_local_train_tensor(setup):
    dtrain = MarsDMatrix(X, y)
    booster = train({}, dtrain, num_boost_round=2)
    assert isinstance(booster, Booster)


@pytest.mark.skipif(xgboost is None, reason="XGBoost not installed")
def test_local_train_sparse_tensor(setup):
    dtrain = MarsDMatrix(X_sparse, y)
    booster = train({}, dtrain, num_boost_round=2)
    assert isinstance(booster, Booster)


@pytest.mark.skipif(xgboost is None, reason="XGBoost not installed")
def test_local_train_dataframe(setup):
    dtrain = MarsDMatrix(X_df, y_series)
    booster = train({}, dtrain, num_boost_round=2)
    assert isinstance(booster, Booster)


@pytest.mark.skipif(xgboost is None, reason="XGBoost not installed")
@pytest.mark.parametrize("chunk_size", [n_rows // 5, n_rows])
def test_train_evals(setup_cluster, chunk_size):
    rs = mt.random.RandomState(0)
    # keep 1 chunk for X and y
    X = rs.rand(n_rows, n_columns, chunk_size=(n_rows, n_columns // 2))
    y = rs.rand(n_rows, chunk_size=n_rows)
    base_margin = rs.rand(n_rows, chunk_size=n_rows)
    dtrain = MarsDMatrix(X, y, base_margin=base_margin)
    eval_x = MarsDMatrix(
        rs.rand(n_rows, n_columns, chunk_size=chunk_size),
        rs.rand(n_rows, chunk_size=chunk_size),
    )
    evals = [(eval_x, "eval_x")]
    eval_result = dict()
    booster = train(
        {}, dtrain, num_boost_round=2, evals=evals, evals_result=eval_result
    )
    assert isinstance(booster, Booster)
    assert len(eval_result) > 0

    with pytest.raises(TypeError):
        train(
            {},
            dtrain,
            num_boost_round=2,
            evals=[("eval_x", eval_x)],
            evals_result=eval_result,
        )


@require_ray
def test_train_on_ray_cluster(ray_start_regular, ray_create_mars_cluster):
    rs = mt.random.RandomState(0)
    # keep 1 chunk for X and y
    X = rs.rand(n_rows, n_columns, chunk_size=(n_rows, n_columns // 2))
    y = rs.rand(n_rows, chunk_size=n_rows)
    base_margin = rs.rand(n_rows, chunk_size=n_rows)
    dtrain = MarsDMatrix(X, y, base_margin=base_margin)
    eval_x = MarsDMatrix(
        rs.rand(n_rows, n_columns, chunk_size=n_rows // 5),
        rs.rand(n_rows, chunk_size=n_rows // 5),
    )
    evals = [(eval_x, "eval_x")]
    eval_result = dict()
    booster = train(
        {}, dtrain, num_boost_round=2, evals=evals, evals_result=eval_result
    )
    assert isinstance(booster, Booster)
    assert len(eval_result) > 0

    with pytest.raises(TypeError):
        train(
            {},
            dtrain,
            num_boost_round=2,
            evals=[("eval_x", eval_x)],
            evals_result=eval_result,
        )
