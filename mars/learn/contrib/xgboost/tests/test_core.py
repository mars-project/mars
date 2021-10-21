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

if xgboost:
    from ..core import wrap_evaluation_matrices


@pytest.mark.skipif(xgboost is None, reason="XGBoost not installed")
def test_wrap_evaluation_matrices():
    X = mt.random.rand(100, 3)
    y = mt.random.randint(3, size=(100,))

    eval_set = [(mt.random.rand(10, 3), mt.random.randint(3, size=10))]
    with pytest.raises(ValueError):
        # sample_weight_eval_set size wrong
        wrap_evaluation_matrices(0.0, X, y, None, None, eval_set, [], None)

    with pytest.raises(ValueError):
        wrap_evaluation_matrices(0.0, X, y, None, None, None, eval_set, None)

    evals = wrap_evaluation_matrices(0.0, X, y, None, None, eval_set, None, None)[1]
    assert len(evals) > 0
