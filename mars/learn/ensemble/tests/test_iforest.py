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
from sklearn.datasets import load_iris

from .... import tensor as mt
from .._iforest import IsolationForest


@pytest.mark.parametrize("max_samples", [0.5, 1.0, 10])
def test_iforest(setup, max_samples):
    rs = np.random.RandomState(0)
    raw_train = rs.poisson(size=(100, 10))
    t_train = mt.tensor(raw_train, chunk_size=20)
    raw_test = rs.poisson(size=(200, 10))
    t_test = mt.tensor(raw_test, chunk_size=20)

    clf = IsolationForest(random_state=rs, n_estimators=10, max_samples=max_samples)
    clf.fit(t_train).predict(t_test)
    clf.score_samples(t_test)


@pytest.mark.parametrize("contamination", [0.25, "auto"])
def test_iforest_works(setup, contamination):
    rs = np.random.RandomState(0)
    # toy sample (the last two samples are outliers)
    raw = np.array(
        [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [-4, 7]]
    )
    t = mt.tensor(raw, chunk_size=4)

    # Test IsolationForest
    clf = IsolationForest(random_state=rs, contamination=contamination)
    clf.fit(t)
    decision_func = -clf.decision_function(t).execute().fetch()
    pred = clf.predict(t).execute().fetch()
    # assert detect outliers:
    assert np.min(decision_func[-2:]) > np.max(decision_func[:-2])
    np.testing.assert_array_equal(pred, 6 * [1] + 2 * [-1])


def test_iforest_error():
    """Test that it gives proper exception on deficient input."""
    iris = load_iris()
    X = iris.data

    # Test max_samples
    with pytest.raises(ValueError):
        IsolationForest(max_samples=-1).fit(X)
    with pytest.raises(ValueError):
        IsolationForest(max_samples=0.0).fit(X)
    with pytest.raises(ValueError):
        IsolationForest(max_samples=2.0).fit(X)

    with pytest.raises(ValueError):
        IsolationForest(max_samples="foobar").fit(X)
    with pytest.raises(ValueError):
        IsolationForest(max_samples=1.5).fit(X)

    # test X_test n_features match X_train one:
    with pytest.raises(ValueError):
        IsolationForest().fit(X).predict(X[:, 1:])
