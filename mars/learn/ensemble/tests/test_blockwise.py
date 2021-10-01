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
from sklearn.linear_model import LogisticRegression

from .... import dataframe as md
from .... import tensor as mt
from .. import BlockwiseVotingClassifier, BlockwiseVotingRegressor


fit_raw_X, fit_raw_y = make_classification()
fit_X, fit_y = mt.tensor(fit_raw_X, chunk_size=25), mt.tensor(fit_raw_y, chunk_size=25)
fit_df_X = md.DataFrame(fit_X)
predict_raw_X, predict_raw_y = make_classification()
predict_X, predict_y = (
    mt.tensor(predict_raw_X, chunk_size=20),
    mt.tensor(predict_raw_y, chunk_size=20),
)
predict_df_X = md.DataFrame(predict_X)


@pytest.mark.parametrize(
    "fit_X, fit_y, predict_X, predict_y",
    [
        (fit_X, fit_y, predict_X, predict_y),
        (fit_raw_X, fit_raw_y, predict_raw_X, predict_raw_y),
        (fit_df_X, fit_raw_y, predict_df_X, predict_raw_y),
    ],
)
def test_blockwise_voting_classifier_hard(setup, fit_X, fit_y, predict_X, predict_y):
    clf = BlockwiseVotingClassifier(LogisticRegression(solver="lbfgs"))
    clf.fit(fit_X, fit_y)
    estimators = clf.estimators_.fetch()
    if not isinstance(fit_X, np.ndarray):
        assert len(estimators) == 4

    clf.predict(predict_X)
    score = clf.score(predict_X, predict_y)
    assert isinstance(score.fetch(), float)

    with pytest.raises(AttributeError, match="hard"):
        clf.predict_proba(predict_X)


@pytest.mark.parametrize(
    "fit_X, fit_y, predict_X, predict_y",
    [
        (fit_X, fit_y, predict_X, predict_y),
        (fit_raw_X, fit_raw_y, predict_raw_X, predict_raw_y),
        (fit_df_X, fit_raw_y, predict_df_X, predict_raw_y),
    ],
)
def test_blockwise_voting_classifier_soft(setup, fit_X, fit_y, predict_X, predict_y):
    clf = BlockwiseVotingClassifier(
        LogisticRegression(solver="lbfgs"),
        voting="soft",
        classes=[0, 1],
    )
    clf.fit(fit_X, fit_y)
    estimators = clf.estimators_.fetch()
    if not isinstance(fit_X, np.ndarray):
        assert len(estimators) == 4

    result = clf.predict(predict_X)
    assert result.dtype == np.dtype("int64")
    assert result.shape == (predict_X.shape[0],)

    result = clf.predict_proba(predict_X)
    assert result.dtype == np.dtype("float64")
    assert result.shape == (predict_X.shape[0], 2)

    score = clf.score(predict_X, predict_y)
    assert isinstance(score.fetch(), float)


@pytest.mark.parametrize(
    "fit_X, fit_y, predict_X, predict_y",
    [
        (fit_X, fit_y, predict_X, predict_y),
        (fit_raw_X, fit_raw_y, predict_raw_X, predict_raw_y),
        (fit_df_X, fit_raw_y, predict_df_X, predict_raw_y),
    ],
)
def test_blockwise_voting_regressor(setup, fit_X, fit_y, predict_X, predict_y):
    est = BlockwiseVotingRegressor(LogisticRegression())
    est.fit(fit_X, fit_y)
    estimators = est.estimators_.fetch()
    if not isinstance(fit_X, np.ndarray):
        assert len(estimators) == 4

    result = est.predict(predict_X)
    assert result.dtype == np.dtype("float64")
    assert result.shape == (predict_X.shape[0],)

    score = est.score(predict_X, predict_y)
    assert isinstance(score.fetch(), float)
