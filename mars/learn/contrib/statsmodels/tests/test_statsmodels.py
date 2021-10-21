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
    import statsmodels
    from .. import MarsDistributedModel, MarsResults
except ImportError:  # pragma: no cover
    statsmodels = MarsDistributedModel = MarsResults = None


n_rows = 1000
n_columns = 10
chunk_size = 200
rs = mt.random.RandomState(0)
X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
y = rs.rand(n_rows, chunk_size=chunk_size)
filter = rs.rand(n_rows, chunk_size=chunk_size) < 0.8
X = X[filter]
y = y[filter]


@pytest.mark.skipif(statsmodels is None, reason="statsmodels not installed")
def test_distributed_stats_models(setup):
    y_data = (y * 10).astype(mt.int32)
    model = MarsDistributedModel(factor=1.2)
    result = model.fit(y_data, X, alpha=0.2)
    prediction = result.predict(X)

    X.execute()

    assert prediction.ndim == 1
    assert prediction.shape[0] == len(X)
