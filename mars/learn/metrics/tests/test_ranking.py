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
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    roc_curve as sklearn_roc_curve,
    auc as sklearn_auc,
    accuracy_score as sklearn_accuracy_score,
)
from sklearn.metrics.tests.test_ranking import make_prediction, _auc
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
)

from .... import dataframe as md
from .... import tensor as mt
from ...utils.extmath import softmax
from .. import roc_curve, auc, accuracy_score, roc_auc_score


def _partial_roc_auc_score(y_true, y_predict, max_fpr):
    """Alternative implementation to check for correctness of `roc_auc_score`
    with `max_fpr` set.
    """

    def _partial_roc(y_true, y_predict, max_fpr):
        fpr, tpr, _ = sklearn_roc_curve(y_true, y_predict)
        new_fpr = fpr[fpr <= max_fpr]
        new_fpr = np.append(new_fpr, max_fpr)
        new_tpr = tpr[fpr <= max_fpr]
        idx_out = np.argmax(fpr > max_fpr)
        idx_in = idx_out - 1
        x_interp = [fpr[idx_in], fpr[idx_out]]
        y_interp = [tpr[idx_in], tpr[idx_out]]
        new_tpr = np.append(new_tpr, np.interp(max_fpr, x_interp, y_interp))
        return (new_fpr, new_tpr)

    new_fpr, new_tpr = _partial_roc(y_true, y_predict, max_fpr)
    partial_auc = sklearn_auc(new_fpr, new_tpr)

    # Formula (5) from McClish 1989
    fpr1 = 0
    fpr2 = max_fpr
    min_area = 0.5 * (fpr2 - fpr1) * (fpr2 + fpr1)
    max_area = fpr2 - fpr1
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))


@pytest.mark.parametrize("drop", [True, False])
def test_roc_curve(setup, drop):
    # Test Area under Receiver Operating Characteristic (ROC) curve
    y_true, _, probas_pred = make_prediction(binary=True)
    expected_auc = _auc(y_true, probas_pred)

    fpr, tpr, thresholds = (
        roc_curve(y_true, probas_pred, drop_intermediate=drop).execute().fetch()
    )
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
    with pytest.warns(w):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # all true labels, all fpr should be nan
    np.testing.assert_array_equal(fpr.fetch(), np.full(len(thresholds), np.nan))
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape

    # assert there are warnings
    with pytest.warns(w):
        fpr, tpr, thresholds = roc_curve([1 - x for x in y_true], y_pred)
    # all negative labels, all tpr should be nan
    np.testing.assert_array_equal(tpr.fetch(), np.full(len(thresholds), np.nan))
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_toydata(setup):
    # Binary classification
    y_true = [0, 1]
    y_score = [0, 1]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 0, 1])
    assert_array_almost_equal(fpr, [0, 1, 1])
    assert_almost_equal(roc_auc, 1.0)

    y_true = [0, 1]
    y_score = [1, 0]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1, 1])
    assert_array_almost_equal(fpr, [0, 0, 1])
    assert_almost_equal(roc_auc, 0.0)

    y_true = [1, 0]
    y_score = [1, 1]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1])
    assert_array_almost_equal(fpr, [0, 1])
    assert_almost_equal(roc_auc, 0.5)

    y_true = [1, 0]
    y_score = [1, 0]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 0, 1])
    assert_array_almost_equal(fpr, [0, 1, 1])
    assert_almost_equal(roc_auc, 1.0)

    y_true = [1, 0]
    y_score = [0.5, 0.5]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1])
    assert_array_almost_equal(fpr, [0, 1])
    assert_almost_equal(roc_auc, 0.5)

    y_true = [0, 0]
    y_score = [0.25, 0.75]
    # assert UndefinedMetricWarning because of no positive sample in y_true
    expected_message = (
        "No positive samples in y_true, true positive value should be meaningless"
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        tpr, fpr, _ = roc_curve(y_true, y_score)

    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0.0, 0.5, 1.0])
    assert_array_almost_equal(fpr, [np.nan, np.nan, np.nan])

    y_true = [1, 1]
    y_score = [0.25, 0.75]
    # assert UndefinedMetricWarning because of no negative sample in y_true
    expected_message = (
        "No negative samples in y_true, false positive value should be meaningless"
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        tpr, fpr, _ = roc_curve(y_true, y_score)

    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [np.nan, np.nan, np.nan])
    assert_array_almost_equal(fpr, [0.0, 0.5, 1.0])

    # Multi-label classification task
    y_true = np.array([[0, 1], [0, 1]])
    y_score = np.array([[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, average="macro")
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, average="weighted")
    assert_almost_equal(roc_auc_score(y_true, y_score, average="samples"), 1.0)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="micro"), 1.0)

    y_true = np.array([[0, 1], [0, 1]])
    y_score = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, average="macro")
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, average="weighted")
    assert_almost_equal(roc_auc_score(y_true, y_score, average="samples"), 0.5)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="micro"), 0.5)

    y_true = np.array([[1, 0], [0, 1]])
    y_score = np.array([[0, 1], [1, 0]])
    assert_almost_equal(roc_auc_score(y_true, y_score, average="macro"), 0)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="weighted"), 0)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="samples"), 0)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="micro"), 0)

    y_true = np.array([[1, 0], [0, 1]])
    y_score = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert_almost_equal(roc_auc_score(y_true, y_score, average="macro"), 0.5)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="weighted"), 0.5)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="samples"), 0.5)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="micro"), 0.5)


def test_roc_curve_drop_intermediate(setup):
    # Test that drop_intermediate drops the correct thresholds
    y_true = [0, 0, 0, 0, 1, 1]
    y_score = [0.0, 0.2, 0.5, 0.6, 0.7, 1.0]
    tpr, fpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
    np.testing.assert_array_almost_equal(thresholds.fetch(), [2.0, 1.0, 0.7, 0.0])

    # Test dropping thresholds with repeating scores
    y_true = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_score = [0.0, 0.1, 0.6, 0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.9, 0.9, 1.0]
    tpr, fpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
    np.testing.assert_array_almost_equal(
        thresholds.fetch(), [2.0, 1.0, 0.9, 0.7, 0.6, 0.0]
    )


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


@pytest.mark.parametrize(
    "y_true, labels",
    [
        (np.array([0, 1, 0, 2]), [0, 1, 2]),
        (np.array([0, 1, 0, 2]), None),
        (["a", "b", "a", "c"], ["a", "b", "c"]),
        (["a", "b", "a", "c"], None),
    ],
)
def test_multiclass_ovo_roc_auc_toydata(setup, y_true, labels):
    # Tests the one-vs-one multiclass ROC AUC algorithm
    # on a small example, representative of an expected use case.
    y_scores = np.array(
        [[0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.35, 0.5, 0.15], [0, 0.2, 0.8]]
    )

    # Used to compute the expected output.
    # Consider labels 0 and 1:
    # positive label is 0, negative label is 1
    score_01 = roc_auc_score([1, 0, 1], [0.1, 0.3, 0.35])
    # positive label is 1, negative label is 0
    score_10 = roc_auc_score([0, 1, 0], [0.8, 0.4, 0.5])
    average_score_01 = (score_01 + score_10) / 2

    # Consider labels 0 and 2:
    score_02 = roc_auc_score([1, 1, 0], [0.1, 0.35, 0])
    score_20 = roc_auc_score([0, 0, 1], [0.1, 0.15, 0.8])
    average_score_02 = (score_02 + score_20) / 2

    # Consider labels 1 and 2:
    score_12 = roc_auc_score([1, 0], [0.4, 0.2])
    score_21 = roc_auc_score([0, 1], [0.3, 0.8])
    average_score_12 = (score_12 + score_21) / 2

    # Unweighted, one-vs-one multiclass ROC AUC algorithm
    ovo_unweighted_score = (average_score_01 + average_score_02 + average_score_12) / 3
    assert_almost_equal(
        roc_auc_score(y_true, y_scores, labels=labels, multi_class="ovo"),
        ovo_unweighted_score,
    )

    # Weighted, one-vs-one multiclass ROC AUC algorithm
    # Each term is weighted by the prevalence for the positive label.
    pair_scores = [average_score_01, average_score_02, average_score_12]
    prevalence = [0.75, 0.75, 0.50]
    ovo_weighted_score = np.average(pair_scores, weights=prevalence)
    assert_almost_equal(
        roc_auc_score(
            y_true, y_scores, labels=labels, multi_class="ovo", average="weighted"
        ),
        ovo_weighted_score,
    )


@pytest.mark.parametrize(
    "y_true, labels",
    [
        (np.array([0, 2, 0, 2]), [0, 1, 2]),
        (np.array(["a", "d", "a", "d"]), ["a", "b", "d"]),
    ],
)
def test_multiclass_ovo_roc_auc_toydata_binary(setup, y_true, labels):
    # Tests the one-vs-one multiclass ROC AUC algorithm for binary y_true
    #
    # on a small example, representative of an expected use case.
    y_scores = np.array(
        [[0.2, 0.0, 0.8], [0.6, 0.0, 0.4], [0.55, 0.0, 0.45], [0.4, 0.0, 0.6]]
    )

    # Used to compute the expected output.
    # Consider labels 0 and 1:
    # positive label is 0, negative label is 1
    score_01 = roc_auc_score([1, 0, 1, 0], [0.2, 0.6, 0.55, 0.4])
    # positive label is 1, negative label is 0
    score_10 = roc_auc_score([0, 1, 0, 1], [0.8, 0.4, 0.45, 0.6])
    ovo_score = (score_01 + score_10) / 2

    assert_almost_equal(
        roc_auc_score(y_true, y_scores, labels=labels, multi_class="ovo"), ovo_score
    )

    # Weighted, one-vs-one multiclass ROC AUC algorithm
    assert_almost_equal(
        roc_auc_score(
            y_true, y_scores, labels=labels, multi_class="ovo", average="weighted"
        ),
        ovo_score,
    )


@pytest.mark.parametrize(
    "y_true, labels",
    [
        (np.array([0, 1, 2, 2]), None),
        (["a", "b", "c", "c"], None),
        ([0, 1, 2, 2], [0, 1, 2]),
        (["a", "b", "c", "c"], ["a", "b", "c"]),
    ],
)
def test_multiclass_ovr_roc_auc_toydata(setup, y_true, labels):
    # Tests the unweighted, one-vs-rest multiclass ROC AUC algorithm
    # on a small example, representative of an expected use case.
    y_scores = np.array(
        [[1.0, 0.0, 0.0], [0.1, 0.5, 0.4], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]]
    )
    # Compute the expected result by individually computing the 'one-vs-rest'
    # ROC AUC scores for classes 0, 1, and 2.
    out_0 = roc_auc_score([1, 0, 0, 0], y_scores[:, 0])
    out_1 = roc_auc_score([0, 1, 0, 0], y_scores[:, 1])
    out_2 = roc_auc_score([0, 0, 1, 1], y_scores[:, 2])
    result_unweighted = (out_0 + out_1 + out_2) / 3.0

    assert_almost_equal(
        roc_auc_score(y_true, y_scores, multi_class="ovr", labels=labels),
        result_unweighted,
    )

    # Tests the weighted, one-vs-rest multiclass ROC AUC algorithm
    # on the same input (Provost & Domingos, 2000)
    result_weighted = out_0 * 0.25 + out_1 * 0.25 + out_2 * 0.5
    assert_almost_equal(
        roc_auc_score(
            y_true, y_scores, multi_class="ovr", labels=labels, average="weighted"
        ),
        result_weighted,
    )


@pytest.mark.parametrize(
    "msg, y_true, labels",
    [
        ("Parameter 'labels' must be unique", np.array([0, 1, 2, 2]), [0, 2, 0]),
        (
            "Parameter 'labels' must be unique",
            np.array(["a", "b", "c", "c"]),
            ["a", "a", "b"],
        ),
        (
            "Number of classes in y_true not equal to the number of columns "
            "in 'y_score'",
            np.array([0, 2, 0, 2]),
            None,
        ),
        (
            "Parameter 'labels' must be ordered",
            np.array(["a", "b", "c", "c"]),
            ["a", "c", "b"],
        ),
        (
            "Number of given labels, 2, not equal to the number of columns in "
            "'y_score', 3",
            np.array([0, 1, 2, 2]),
            [0, 1],
        ),
        (
            "Number of given labels, 2, not equal to the number of columns in "
            "'y_score', 3",
            np.array(["a", "b", "c", "c"]),
            ["a", "b"],
        ),
        (
            "Number of given labels, 4, not equal to the number of columns in "
            "'y_score', 3",
            np.array([0, 1, 2, 2]),
            [0, 1, 2, 3],
        ),
        (
            "Number of given labels, 4, not equal to the number of columns in "
            "'y_score', 3",
            np.array(["a", "b", "c", "c"]),
            ["a", "b", "c", "d"],
        ),
        (
            "'y_true' contains labels not in parameter 'labels'",
            np.array(["a", "b", "c", "e"]),
            ["a", "b", "c"],
        ),
        (
            "'y_true' contains labels not in parameter 'labels'",
            np.array(["a", "b", "c", "d"]),
            ["a", "b", "c"],
        ),
        (
            "'y_true' contains labels not in parameter 'labels'",
            np.array([0, 1, 2, 3]),
            [0, 1, 2],
        ),
    ],
)
@pytest.mark.parametrize("multi_class", ["ovo", "ovr"])
def test_roc_auc_score_multiclass_labels_error(setup, msg, y_true, labels, multi_class):
    y_scores = np.array(
        [[0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.35, 0.5, 0.15], [0, 0.2, 0.8]]
    )

    with pytest.raises(ValueError, match=msg):
        roc_auc_score(y_true, y_scores, labels=labels, multi_class=multi_class)


@pytest.mark.parametrize(
    "msg, kwargs",
    [
        (
            (
                r"average must be one of \('macro', 'weighted'\) for "
                r"multiclass problems"
            ),
            {"average": "samples", "multi_class": "ovo"},
        ),
        (
            (
                r"average must be one of \('macro', 'weighted'\) for "
                r"multiclass problems"
            ),
            {"average": "micro", "multi_class": "ovr"},
        ),
        (
            (
                r"sample_weight is not supported for multiclass one-vs-one "
                r"ROC AUC, 'sample_weight' must be None in this case"
            ),
            {"multi_class": "ovo", "sample_weight": []},
        ),
        (
            (
                r"Partial AUC computation not available in multiclass setting, "
                r"'max_fpr' must be set to `None`, received `max_fpr=0.5` "
                r"instead"
            ),
            {"multi_class": "ovo", "max_fpr": 0.5},
        ),
        (
            (
                r"multi_class='ovp' is not supported for multiclass ROC AUC, "
                r"multi_class must be in \('ovo', 'ovr'\)"
            ),
            {"multi_class": "ovp"},
        ),
        (r"multi_class must be in \('ovo', 'ovr'\)", {}),
    ],
)
def test_roc_auc_score_multiclass_error(setup, msg, kwargs):
    # Test that roc_auc_score function returns an error when trying
    # to compute multiclass AUC for parameters where an output
    # is not defined.
    rng = check_random_state(404)
    y_score = rng.rand(20, 3)
    y_prob = softmax(y_score)
    y_true = rng.randint(0, 3, size=20)
    with pytest.raises(ValueError, match=msg):
        roc_auc_score(y_true, y_prob, **kwargs)


def test_auc_score_non_binary_class(setup):
    # Test that roc_auc_score function returns an error when trying
    # to compute AUC for non-binary class values.
    rng = check_random_state(404)
    y_pred = rng.rand(10)
    # y_true contains only one class value
    y_true = np.zeros(10, dtype="int")
    err_msg = "ROC AUC score is not defined"
    with pytest.raises(ValueError, match=err_msg):
        roc_auc_score(y_true, y_pred)
    y_true = np.ones(10, dtype="int")
    with pytest.raises(ValueError, match=err_msg):
        roc_auc_score(y_true, y_pred)
    y_true = np.full(10, -1, dtype="int")
    with pytest.raises(ValueError, match=err_msg):
        roc_auc_score(y_true, y_pred)

    with warnings.catch_warnings(record=True):
        rng = check_random_state(404)
        y_pred = rng.rand(10)
        # y_true contains only one class value
        y_true = np.zeros(10, dtype="int")
        with pytest.raises(ValueError, match=err_msg):
            roc_auc_score(y_true, y_pred)
        y_true = np.ones(10, dtype="int")
        with pytest.raises(ValueError, match=err_msg):
            roc_auc_score(y_true, y_pred)
        y_true = np.full(10, -1, dtype="int")
        with pytest.raises(ValueError, match=err_msg):
            roc_auc_score(y_true, y_pred)


def test_binary_clf_curve_multiclass_error(setup):
    rng = check_random_state(404)
    y_true = rng.randint(0, 3, size=10)
    y_pred = rng.rand(10)
    msg = "multiclass format is not supported"

    with pytest.raises(ValueError, match=msg):
        roc_curve(y_true, y_pred)


def test_dataframe_roc_curve_auc(setup):
    rs = np.random.RandomState(0)
    raw = pd.DataFrame({"a": rs.randint(0, 10, (10,)), "b": rs.rand(10)})

    df = md.DataFrame(raw)
    y = df["a"].to_tensor().astype("int")
    pred = df["b"].to_tensor().astype("float")
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=2)
    m = auc(fpr, tpr)

    sk_fpr, sk_tpr, sk_threshod = sklearn_roc_curve(
        raw["a"].to_numpy().astype("int"),
        raw["b"].to_numpy().astype("float"),
        pos_label=2,
    )
    expect_m = sklearn_auc(sk_fpr, sk_tpr)
    assert pytest.approx(m.fetch()) == expect_m


def test_dataframe_accuracy_score(setup):
    rs = np.random.RandomState(0)
    raw = pd.DataFrame({"a": rs.randint(0, 10, (10,)), "b": rs.randint(0, 10, (10,))})

    df = md.DataFrame(raw)
    y = df["a"].to_tensor().astype("int")
    pred = df["b"].astype("int")

    score = accuracy_score(y, pred)
    expect = sklearn_accuracy_score(
        raw["a"].to_numpy().astype("int"), raw["b"].to_numpy().astype("int")
    )
    assert pytest.approx(score.fetch()) == expect


def test_partial_roc_auc_score(setup):
    # Check `roc_auc_score` for max_fpr != `None`
    y_true = np.array([0, 0, 1, 1])
    assert roc_auc_score(y_true, y_true, max_fpr=1) == 1
    assert roc_auc_score(y_true, y_true, max_fpr=0.001) == 1
    with pytest.raises(ValueError):
        assert roc_auc_score(y_true, y_true, max_fpr=-0.1)
    with pytest.raises(ValueError):
        assert roc_auc_score(y_true, y_true, max_fpr=1.1)
    with pytest.raises(ValueError):
        assert roc_auc_score(y_true, y_true, max_fpr=0)

    y_scores = np.array([0.1, 0, 0.1, 0.01])
    roc_auc_with_max_fpr_one = roc_auc_score(y_true, y_scores, max_fpr=1)
    unconstrained_roc_auc = roc_auc_score(y_true, y_scores)
    assert roc_auc_with_max_fpr_one == unconstrained_roc_auc
    assert roc_auc_score(y_true, y_scores, max_fpr=0.3) == 0.5

    y_true, y_pred, _ = make_prediction(binary=True)
    for max_fpr in np.linspace(1e-4, 1, 5):
        assert_almost_equal(
            roc_auc_score(y_true, y_pred, max_fpr=max_fpr),
            _partial_roc_auc_score(y_true, y_pred, max_fpr),
        )
