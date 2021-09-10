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

import re

import numpy as np
import pandas as pd
try:
    import sklearn
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve as sklearn_roc_curve, auc as sklearn_auc, \
        accuracy_score as sklearn_accuracy_score
    from sklearn.metrics.tests.test_ranking import make_prediction, _auc
    from sklearn.exceptions import UndefinedMetricWarning
    from sklearn.utils import check_random_state
    from sklearn.utils._testing import assert_warns
    from sklearn.metrics._ranking import _binary_roc_auc_score as sk_binary_roc_auc_score
except ImportError:  # pragma: no cover
    sklearn = None
import pytest


from .... import dataframe as md
from .... import tensor as mt
from .. import roc_curve, auc, accuracy_score
from .._ranking import _binary_roc_auc_score


def test_roc_curve(setup):
    for drop in [True, False]:
        # Test Area under Receiver Operating Characteristic (ROC) curve
        y_true, _, probas_pred = make_prediction(binary=True)
        expected_auc = _auc(y_true, probas_pred)

        fpr, tpr, thresholds = roc_curve(y_true, probas_pred,
                                         drop_intermediate=drop).execute().fetch()
        roc_auc = auc(fpr, tpr).to_numpy()
        np.testing.assert_array_almost_equal(roc_auc, expected_auc, decimal=2)
        np.testing.assert_almost_equal(roc_auc, roc_auc_score(y_true, probas_pred))
        assert fpr.shape == tpr.shape
        assert fpr.shape == thresholds.shape


def test_roc_curve_end_points(setup):
    # Make sure that roc_curve returns a curve start at 0 and ending and
    # 1 even in corner cases
    rng = np.random.RandomState(0)
    y_true = np.array([0] * 50 + [1] * 50)
    y_pred = rng.randint(3, size=100)
    fpr, tpr, thr = roc_curve(y_true, y_pred, drop_intermediate=True).fetch()
    assert fpr[0] == 0
    assert fpr[-1] == 1
    assert fpr.shape == tpr.shape
    assert fpr.shape == thr.shape


def test_roc_returns_consistency(setup):
    # Test whether the returned threshold matches up with tpr
    # make small toy dataset
    y_true, _, probas_pred = make_prediction(binary=True)
    fpr, tpr, thresholds = roc_curve(y_true, probas_pred).fetch()

    # use the given thresholds to determine the tpr
    tpr_correct = []
    for t in thresholds:
        tp = np.sum((probas_pred >= t) & y_true)
        p = np.sum(y_true)
        tpr_correct.append(1.0 * tp / p)

    # compare tpr and tpr_correct to see if the thresholds' order was correct
    np.testing.assert_array_almost_equal(tpr, tpr_correct, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_multi(setup):
    # roc_curve not applicable for multi-class problems
    y_true, _, probas_pred = make_prediction(binary=False)

    with pytest.raises(ValueError):
        roc_curve(y_true, probas_pred)


def test_roc_curve_confidence(setup):
    # roc_curve for confidence scores
    y_true, _, probas_pred = make_prediction(binary=True)

    fpr, tpr, thresholds = roc_curve(y_true, probas_pred - 0.5)
    roc_auc = auc(fpr, tpr).fetch()
    np.testing.assert_array_almost_equal(roc_auc, 0.90, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_hard(setup):
    # roc_curve for hard decisions
    y_true, pred, probas_pred = make_prediction(binary=True)

    # always predict one
    trivial_pred = np.ones(y_true.shape)
    fpr, tpr, thresholds = roc_curve(y_true, trivial_pred)
    roc_auc = auc(fpr, tpr).fetch()
    np.testing.assert_array_almost_equal(roc_auc, 0.50, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape

    # always predict zero
    trivial_pred = np.zeros(y_true.shape)
    fpr, tpr, thresholds = roc_curve(y_true, trivial_pred)
    roc_auc = auc(fpr, tpr).fetch()
    np.testing.assert_array_almost_equal(roc_auc, 0.50, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape

    # hard decisions
    fpr, tpr, thresholds = roc_curve(y_true, pred)
    roc_auc = auc(fpr, tpr).fetch()
    np.testing.assert_array_almost_equal(roc_auc, 0.78, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_one_label(setup):
    y_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # assert there are warnings
    w = UndefinedMetricWarning
    fpr, tpr, thresholds = assert_warns(w, roc_curve, y_true, y_pred)
    # all true labels, all fpr should be nan
    np.testing.assert_array_equal(fpr.fetch(), np.full(len(thresholds), np.nan))
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape

    # assert there are warnings
    fpr, tpr, thresholds = assert_warns(w, roc_curve,
                                        [1 - x for x in y_true],
                                        y_pred)
    # all negative labels, all tpr should be nan
    np.testing.assert_array_equal(tpr.fetch(), np.full(len(thresholds), np.nan))
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_binary_roc_auc_score(setup):
    # Test the area is equal under binary roc_auc_score
    rs = np.random.RandomState(0)
    raw_X = rs.randint(0, 2, size=10)
    raw_Y = rs.rand(10).astype('float32')

    X = mt.tensor(raw_X)
    Y = mt.tensor(raw_Y)

    for max_fpr in (np.random.rand(), None):
        # Calculate the score using both frameworks
        score = _binary_roc_auc_score(X, Y, max_fpr=max_fpr)
        expected_score = sk_binary_roc_auc_score(raw_X, raw_Y, max_fpr=max_fpr)

        # Both the scores should be equal
        np.testing.assert_almost_equal(score, expected_score, decimal=6)

    with pytest.raises(ValueError):
        _binary_roc_auc_score(mt.tensor([0]), Y)

    with pytest.raises(ValueError):
        _binary_roc_auc_score(X, Y, max_fpr=0)


def test_roc_curve_drop_intermediate(setup):
    # Test that drop_intermediate drops the correct thresholds
    y_true = [0, 0, 0, 0, 1, 1]
    y_score = [0., 0.2, 0.5, 0.6, 0.7, 1.0]
    tpr, fpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
    np.testing.assert_array_almost_equal(thresholds.fetch(), [2., 1., 0.7, 0.])

    # Test dropping thresholds with repeating scores
    y_true = [0, 0, 0, 0, 0, 0, 0,
              1, 1, 1, 1, 1, 1]
    y_score = [0., 0.1, 0.6, 0.6, 0.7, 0.8, 0.9,
               0.6, 0.7, 0.8, 0.9, 0.9, 1.0]
    tpr, fpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
    np.testing.assert_array_almost_equal(thresholds.fetch(),
                                         [2.0, 1.0, 0.9, 0.7, 0.6, 0.])


def test_roc_curve_fpr_tpr_increasing(setup):
    # Ensure that fpr and tpr returned by roc_curve are increasing.
    # Construct an edge case with float y_score and sample_weight
    # when some adjacent values of fpr and tpr are actually the same.
    y_true = [0, 0, 1, 1, 1]
    y_score = [0.1, 0.7, 0.3, 0.4, 0.5]
    sample_weight = np.repeat(0.2, 5)
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    assert ((mt.diff(fpr) < 0).sum() == 0).to_numpy()
    assert ((mt.diff(tpr) < 0).sum() == 0).to_numpy()


def test_auc(setup):
    # Test Area Under Curve (AUC) computation
    x = [0, 1]
    y = [0, 1]
    np.testing.assert_array_almost_equal(auc(x, y).fetch(), 0.5)
    x = [1, 0]
    y = [0, 1]
    np.testing.assert_array_almost_equal(auc(x, y).fetch(), 0.5)
    x = [1, 0, 0]
    y = [0, 1, 1]
    np.testing.assert_array_almost_equal(auc(x, y).fetch(), 0.5)
    x = [0, 1]
    y = [1, 1]
    np.testing.assert_array_almost_equal(auc(x, y).fetch(), 1)
    x = [0, 0.5, 1]
    y = [0, 0.5, 1]
    np.testing.assert_array_almost_equal(auc(x, y).fetch(), 0.5)


def test_auc_errors(setup):
    # Incompatible shapes
    with pytest.raises(ValueError):
        auc([0.0, 0.5, 1.0], [0.1, 0.2])

    # Too few x values
    with pytest.raises(ValueError):
        auc([0.0], [0.1])

    # x is not in order
    x = [2, 1, 3, 4]
    y = [5, 6, 7, 8]
    error_message = f"x is neither increasing nor decreasing : {np.array(x)}"
    with pytest.raises(ValueError, match=re.escape(error_message)):
        auc(x, y)


def test_binary_clf_curve_multiclass_error(setup):
    rng = check_random_state(404)
    y_true = rng.randint(0, 3, size=10)
    y_pred = rng.rand(10)
    msg = "multiclass format is not supported"

    with pytest.raises(ValueError, match=msg):
        roc_curve(y_true, y_pred)


def test_dataframe_roc_curve_auc(setup):
    rs = np.random.RandomState(0)
    raw = pd.DataFrame({'a': rs.randint(0, 10, (10,)),
                        'b': rs.rand(10)})

    df = md.DataFrame(raw)
    y = df['a'].to_tensor().astype('int')
    pred = df['b'].to_tensor().astype('float')
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=2)
    m = auc(fpr, tpr)

    sk_fpr, sk_tpr, sk_threshod = sklearn_roc_curve(raw['a'].to_numpy().astype('int'),
                                                    raw['b'].to_numpy().astype('float'),
                                                    pos_label=2)
    expect_m = sklearn_auc(sk_fpr, sk_tpr)
    assert pytest.approx(m.fetch()) == expect_m


def test_dataframe_accuracy_score(setup):
    rs = np.random.RandomState(0)
    raw = pd.DataFrame({'a': rs.randint(0, 10, (10,)),
                        'b': rs.randint(0, 10, (10,))})

    df = md.DataFrame(raw)
    y = df['a'].to_tensor().astype('int')
    pred = df['b'].astype('int')

    score = accuracy_score(y, pred)
    expect = sklearn_accuracy_score(raw['a'].to_numpy().astype('int'),
                                    raw['b'].to_numpy().astype('int'))
    assert pytest.approx(score.fetch()) == expect
