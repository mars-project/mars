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
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from ... import tensor as mt
from ..wrappers import ParallelPostFit


def test_parallel_post_fit_basic(setup):
    raw_x, raw_y = make_classification(n_samples=1000)
    X, y = mt.tensor(raw_x, chunk_size=100), mt.tensor(raw_y, chunk_size=100)
    clf = ParallelPostFit(GradientBoostingClassifier())
    clf.fit(X, y)

    assert isinstance(clf.predict(X), mt.Tensor)
    assert isinstance(clf.predict_proba(X), mt.Tensor)

    result = clf.score(X, y)
    expected = clf.estimator.score(X, y)
    assert result.fetch() == expected

    clf = ParallelPostFit(LinearRegression())
    clf.fit(X, y)
    with pytest.raises(
        AttributeError, match="The wrapped estimator (.|\n)* 'predict_proba' method."
    ):
        clf.predict_proba(X)


def test_parallel_post_fit_predict(setup):
    raw_x, raw_y = make_classification(n_samples=1000)
    X, y = mt.tensor(raw_x, chunk_size=100), mt.tensor(raw_y, chunk_size=100)
    base = LogisticRegression(random_state=0, n_jobs=1, solver="lbfgs")
    wrap = ParallelPostFit(LogisticRegression(random_state=0, n_jobs=1, solver="lbfgs"))

    base.fit(X, y)
    wrap.fit(X, y)

    result = wrap.predict(X)
    expected = base.predict(X)
    np.testing.assert_allclose(result, expected)

    result = wrap.predict_proba(X)
    expected = base.predict_proba(X)
    np.testing.assert_allclose(result, expected)

    result = wrap.predict_log_proba(X)
    expected = base.predict_log_proba(X)
    np.testing.assert_allclose(result, expected)


def test_parallel_post_fit_transform(setup):
    raw_x, raw_y = make_classification(n_samples=1000)
    X, y = mt.tensor(raw_x, chunk_size=100), mt.tensor(raw_y, chunk_size=100)
    base = PCA(random_state=0)
    wrap = ParallelPostFit(PCA(random_state=0))

    base.fit(raw_x, raw_y)
    wrap.fit(X, y)

    result = base.transform(X)
    expected = wrap.transform(X)
    np.testing.assert_allclose(result, expected, atol=0.1)


def test_parallel_post_fit_multiclass(setup):
    raw_x, raw_y = make_classification(n_samples=1000)
    X, y = mt.tensor(raw_x, chunk_size=100), mt.tensor(raw_y, chunk_size=100)
    raw_x, raw_y = make_classification(n_classes=3, n_informative=4)
    X, y = mt.tensor(raw_x, chunk_size=50), mt.tensor(raw_y, chunk_size=50)

    clf = ParallelPostFit(
        LogisticRegression(random_state=0, n_jobs=1, solver="lbfgs", multi_class="auto")
    )

    clf.fit(X, y)
    result = clf.predict(X)
    expected = clf.estimator.predict(X)

    np.testing.assert_allclose(result, expected)

    result = clf.predict_proba(X)
    expected = clf.estimator.predict_proba(X)

    np.testing.assert_allclose(result, expected)

    result = clf.predict_log_proba(X)
    expected = clf.estimator.predict_log_proba(X)

    np.testing.assert_allclose(result, expected)
