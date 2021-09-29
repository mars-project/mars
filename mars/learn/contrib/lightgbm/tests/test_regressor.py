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

import pandas as pd
import pytest

from ..... import tensor as mt

try:
    import lightgbm
    from .. import LGBMRegressor
except ImportError:
    lightgbm = LGBMRegressor = None


n_rows = 1000
n_columns = 10
chunk_size = 200
rs = mt.random.RandomState(0)
X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
y = rs.randint(0, 10, n_rows, chunk_size=chunk_size)


@pytest.mark.skipif(lightgbm is None, reason="LightGBM not installed")
def test_local_regressor(setup):
    regressor = LGBMRegressor(n_estimators=2)
    regressor.fit(X, y, verbose=True)
    prediction = regressor.predict(X)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X)

    assert isinstance(prediction, mt.Tensor)
    result = prediction.fetch()
    assert prediction.dtype == result.dtype

    # test weight
    weight = mt.random.rand(X.shape[0])
    regressor = LGBMRegressor(verbosity=1, n_estimators=2)
    regressor.fit(X, y, sample_weight=weight)
    prediction = regressor.predict(X)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X)
    result = prediction.fetch()
    assert prediction.dtype == result.dtype

    # test numpy tensor
    try:
        from sklearn.datasets import make_classification

        X_array, y_array = make_classification()
        regressor = LGBMRegressor(n_estimators=2)
        regressor.fit(X_array, y_array, verbose=True)
        prediction = regressor.predict(X_array)

        assert prediction.ndim == 1
        assert prediction.shape[0] == len(X_array)

        X_df = pd.DataFrame(X_array)
        y_df = pd.Series(y_array)
        regressor = LGBMRegressor(n_estimators=2)
        regressor.fit(X_df, y_df, verbose=True)
        prediction = regressor.predict(X_df)

        assert prediction.ndim == 1
        assert prediction.shape[0] == len(X_df)
    except ImportError:
        pass

    # test existing model
    X_np = X.execute().fetch()
    y_np = y.execute().fetch()
    raw_regressor = lightgbm.LGBMRegressor(verbosity=1, n_estimators=2)
    raw_regressor.fit(X_np, y_np)
    prediction = LGBMRegressor(raw_regressor).predict(X)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X)
