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

from ..... import tensor as mt

try:
    import lightgbm
    from .. import LGBMRanker
except ImportError:
    lightgbm = LGBMRanker = None


n_rows = 1000
n_columns = 10
chunk_size = 200
rs = mt.random.RandomState(0)
X_raw = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
y_raw = rs.rand(n_rows, chunk_size=chunk_size)


@pytest.mark.skipif(lightgbm is None, reason="LightGBM not installed")
def test_local_ranker(setup):
    y = (y_raw * 10).astype(mt.int32)
    ranker = LGBMRanker(n_estimators=2)
    ranker.fit(X_raw, y, group=[X_raw.shape[0]], verbose=True)
    prediction = ranker.predict(X_raw)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X_raw)

    assert isinstance(prediction, mt.Tensor)
    result = prediction.fetch()
    assert prediction.dtype == result.dtype

    # test weight
    weight = mt.random.rand(X_raw.shape[0])
    ranker = LGBMRanker(verbosity=1, n_estimators=2)
    ranker.fit(X_raw, y, group=[X_raw.shape[0]], sample_weight=weight)
    prediction = ranker.predict(X_raw)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X_raw)
    result = prediction.fetch()
    assert prediction.dtype == result.dtype

    # test local model
    X_np = X_raw.execute().fetch()
    y_np = y.execute().fetch()
    raw_ranker = lightgbm.LGBMRanker(verbosity=1, n_estimators=2)
    raw_ranker.fit(X_np, y_np, group=[X_raw.shape[0]])
    prediction = LGBMRanker(raw_ranker).predict(X_raw)

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X_raw)
