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
from sklearn import datasets, svm
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_array_almost_equal,
)
from sklearn.utils._mocking import MockDataFrame

from .... import execute, fetch
from .... import tensor as mt
from ....lib.sparse import SparseNDArray
from .. import accuracy_score, log_loss
from .._classification import (
    _check_targets,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)

IND = "multilabel-indicator"
MC = "multiclass"
BIN = "binary"
CNT = "continuous"
MMC = "multiclass-multioutput"
MCN = "continuous-multioutput"
# all of length 3
EXAMPLES = [
    (IND, np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]])),
    # must not be considered binary
    (IND, np.array([[0, 1], [1, 0], [1, 1]])),
    (MC, [2, 3, 1]),
    (BIN, [0, 1, 1]),
    (CNT, [0.0, 1.5, 1.0]),
    (MC, np.array([[2], [3], [1]])),
    (BIN, np.array([[0], [1], [1]])),
    (CNT, np.array([[0.0], [1.5], [1.0]])),
    (MMC, np.array([[0, 2], [1, 3], [2, 3]])),
    (MCN, np.array([[0.5, 2.0], [1.1, 3.0], [2.0, 3.0]])),
]
# expected type given input types, or None for error
# (types will be tried in either order)
EXPECTED = {
    (IND, IND): IND,
    (MC, MC): MC,
    (BIN, BIN): BIN,
    (MC, IND): None,
    (BIN, IND): None,
    (BIN, MC): MC,
    # Disallowed types
    (CNT, CNT): None,
    (MMC, MMC): None,
    (MCN, MCN): None,
    (IND, CNT): None,
    (MC, CNT): None,
    (BIN, CNT): None,
    (MMC, CNT): None,
    (MCN, CNT): None,
    (IND, MMC): None,
    (MC, MMC): None,
    (BIN, MMC): None,
    (MCN, MMC): None,
    (IND, MCN): None,
    (MC, MCN): None,
    (BIN, MCN): None,
}


###############################################################################
# Utilities for testing


def make_prediction(dataset=None, binary=False):
    """Make some classification predictions on a toy dataset using a SVC

    If binary is True restrict to a binary classification problem instead of a
    multiclass classification problem
    """

    if dataset is None:
        # import some data to play with
        dataset = datasets.load_iris()

    X = dataset.data
    y = dataset.target

    if binary:
        # restrict to a binary classification task
        X, y = X[y < 2], y[y < 2]

    n_samples, n_features = X.shape
    p = np.arange(n_samples)

    rng = check_random_state(37)
    rng.shuffle(p)
    X, y = X[p], y[p]
    half = int(n_samples / 2)

    # add noisy features to make the problem harder and avoid perfect results
    rng = np.random.RandomState(0)
    X = np.c_[X, rng.randn(n_samples, 200 * n_features)]

    # run classifier, get class probabilities and label predictions
    clf = svm.SVC(kernel="linear", probability=True, random_state=0)
    probas_pred = clf.fit(X[:half], y[:half]).predict_proba(X[half:])

    if binary:
        # only interested in probabilities of the positive case
        # XXX: do we really want a special API for the binary case?
        probas_pred = probas_pred[:, 1]

    y_pred = clf.predict(X[half:])
    y_true = y[half:]
    return y_true, y_pred, probas_pred


@pytest.mark.parametrize("type1, y1", EXAMPLES)
@pytest.mark.parametrize("type2, y2", EXAMPLES)
def test__check_targets(setup, type1, y1, type2, y2):
    # Check that _check_targets correctly merges target types, squeezes
    # output and fails if input lengths differ.
    try:
        expected = EXPECTED[type1, type2]
    except KeyError:
        expected = EXPECTED[type2, type1]
    if expected is None:
        with pytest.raises(ValueError):
            _check_targets(y1, y2).execute()

        if type1 != type2:
            with pytest.raises(ValueError):
                _check_targets(y1, y2).execute()

        else:
            if type1 not in (BIN, MC, IND):
                with pytest.raises(ValueError):
                    _check_targets(y1, y2).execute()

    else:
        merged_type, y1out, y2out = _check_targets(y1, y2).execute().fetch()
        assert merged_type == expected
        if merged_type.startswith("multilabel"):
            assert isinstance(y1out, SparseNDArray)
            assert isinstance(y2out, SparseNDArray)
        else:
            np.testing.assert_array_equal(y1out, np.squeeze(y1))
            np.testing.assert_array_equal(y2out, np.squeeze(y2))
        with pytest.raises(ValueError):
            _check_targets(y1[:-1], y2).execute()


def test_accuracy_score(setup):
    y_pred = [0, 2, 1, 3]
    y_true = [0, 1, 2, 3]

    score = accuracy_score(y_true, y_pred)
    result = score.execute().fetch()
    expected = sklearn_accuracy_score(y_true, y_pred)
    assert pytest.approx(result) == expected

    score = accuracy_score(y_true, y_pred, normalize=False)
    result = score.execute().fetch()
    expected = sklearn_accuracy_score(y_true, y_pred, normalize=False)
    assert pytest.approx(result) == expected

    y_pred = np.array([[0, 1], [1, 1]])
    y_true = np.ones((2, 2))
    score = accuracy_score(y_true, y_pred)
    result = score.execute().fetch()
    expected = sklearn_accuracy_score(y_true, y_pred)
    assert pytest.approx(result) == expected

    sample_weight = [0.7, 0.3]
    score = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    result = score.execute().fetch()
    expected = sklearn_accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    assert pytest.approx(result) == expected

    score = accuracy_score(
        mt.tensor(y_true),
        mt.tensor(y_pred),
        sample_weight=mt.tensor(sample_weight),
        normalize=False,
    )
    result = score.execute().fetch()
    expected = sklearn_accuracy_score(
        y_true, y_pred, sample_weight=sample_weight, normalize=False
    )
    assert pytest.approx(result) == expected


def test_log_loss(setup):
    # binary case with symbolic labels ("no" < "yes")
    y_true = ["no", "no", "no", "yes", "yes", "yes"]
    y_pred = mt.array(
        [[0.5, 0.5], [0.1, 0.9], [0.01, 0.99], [0.9, 0.1], [0.75, 0.25], [0.001, 0.999]]
    )
    loss = log_loss(y_true, y_pred).fetch()
    assert_almost_equal(loss, 1.8817971)

    # multiclass case; adapted from http://bit.ly/RJJHWA
    y_true = [1, 0, 2]
    y_pred = [[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]]
    loss = log_loss(y_true, y_pred, normalize=True).fetch()
    assert_almost_equal(loss, 0.6904911)

    # check that we got all the shapes and axes right
    # by doubling the length of y_true and y_pred
    y_true *= 2
    y_pred *= 2
    loss = log_loss(y_true, y_pred, normalize=False).fetch()
    assert_almost_equal(loss, 0.6904911 * 6, decimal=6)

    # check eps and handling of absolute zero and one probabilities
    y_pred = np.asarray(y_pred) > 0.5
    loss = log_loss(y_true, y_pred, normalize=True, eps=0.1).fetch()
    assert_almost_equal(loss, log_loss(y_true, np.clip(y_pred, 0.1, 0.9)).fetch())

    # raise error if number of classes are not equal.
    y_true = [1, 0, 2]
    y_pred = [[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]]
    with pytest.raises(ValueError):
        log_loss(y_true, y_pred)
    with pytest.raises(ValueError):
        log_loss(y_true, y_pred, labels=[0, 1, 2])

    # case when y_true is a string array object
    y_true = ["ham", "spam", "spam", "ham"]
    y_pred = [[0.2, 0.7], [0.6, 0.5], [0.4, 0.1], [0.7, 0.2]]
    loss = log_loss(y_true, y_pred).fetch()
    assert_almost_equal(loss, 1.0383217, decimal=6)

    # test labels option

    y_true = [2, 2]
    y_pred = [[0.2, 0.7], [0.6, 0.5]]
    y_score = np.array([[0.1, 0.9], [0.1, 0.9]])
    error_str = (
        r"y_true contains only one label \(2\). Please provide "
        r"the true labels explicitly through the labels argument."
    )
    with pytest.raises(ValueError, match=error_str):
        log_loss(y_true, y_pred)
    error_str = (
        r"The labels array needs to contain at least two "
        r"labels for log_loss, got \[1\]."
    )
    with pytest.raises(ValueError, match=error_str):
        log_loss(y_true, y_pred, labels=[1])

    # works when the labels argument is used

    true_log_loss = -np.mean(np.log(y_score[:, 1]))
    calculated_log_loss = log_loss(y_true, y_score, labels=[1, 2]).fetch()
    assert_almost_equal(calculated_log_loss, true_log_loss)

    # ensure labels work when len(np.unique(y_true)) != y_pred.shape[1]
    y_true = [1, 2, 2]
    y_score2 = [[0.2, 0.7, 0.3], [0.6, 0.5, 0.3], [0.3, 0.9, 0.1]]
    loss = log_loss(y_true, y_score2, labels=[1, 2, 3]).fetch()
    assert_almost_equal(loss, 1.0630345, decimal=6)


def test_log_loss_pandas_input(setup):
    # case when input is a pandas series and dataframe gh-5715
    y_tr = np.array(["ham", "spam", "spam", "ham"])
    y_pr = np.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1], [0.7, 0.2]])
    types = [(MockDataFrame, MockDataFrame)]
    try:
        from pandas import Series, DataFrame

        types.append((Series, DataFrame))
    except ImportError:
        pass
    for TrueInputType, PredInputType in types:
        # y_pred dataframe, y_true series
        y_true, y_pred = TrueInputType(y_tr), PredInputType(y_pr)
        loss = log_loss(y_true, y_pred).fetch()
        assert_almost_equal(loss, 1.0383217, decimal=6)


def test_multilabel_confusion_matrix_binary(setup):
    # Test multilabel confusion matrix - binary classification case
    y_true, y_pred, _ = make_prediction(binary=True)
    y_true = mt.tensor(y_true, chunk_size=40)
    y_pred = mt.tensor(y_pred, chunk_size=40)

    def run_test(y_true, y_pred):
        cm = multilabel_confusion_matrix(y_true, y_pred).fetch()
        assert_array_equal(cm, [[[17, 8], [3, 22]], [[22, 3], [8, 17]]])

    run_test(y_true, y_pred)
    run_test(y_true.astype(str), y_pred.astype(str))


def test_multilabel_confusion_matrix_multiclass(setup):
    # Test multilabel confusion matrix - multi-class case
    y_true, y_pred, _ = make_prediction(binary=False)
    y_true = mt.tensor(y_true, chunk_size=40)
    y_pred = mt.tensor(y_pred, chunk_size=40)

    def run_test(y_true, y_pred, string_type=False):
        # compute confusion matrix with default labels introspection
        cm = multilabel_confusion_matrix(y_true, y_pred).fetch()
        assert_array_equal(
            cm, [[[47, 4], [5, 19]], [[38, 6], [28, 3]], [[30, 25], [2, 18]]]
        )

        # compute confusion matrix with explicit label ordering
        labels = ["0", "2", "1"] if string_type else [0, 2, 1]
        cm = multilabel_confusion_matrix(y_true, y_pred, labels=labels).fetch()
        assert_array_equal(
            cm, [[[47, 4], [5, 19]], [[30, 25], [2, 18]], [[38, 6], [28, 3]]]
        )

        # compute confusion matrix with super set of present labels
        labels = ["0", "2", "1", "3"] if string_type else [0, 2, 1, 3]
        cm = multilabel_confusion_matrix(y_true, y_pred, labels=labels).fetch()
        assert_array_equal(
            cm,
            [
                [[47, 4], [5, 19]],
                [[30, 25], [2, 18]],
                [[38, 6], [28, 3]],
                [[75, 0], [0, 0]],
            ],
        )

    run_test(y_true, y_pred)
    run_test(y_true.astype(str), y_pred.astype(str), string_type=True)


def test_multilabel_confusion_matrix_multilabel(setup):
    # Test multilabel confusion matrix - multilabel-indicator case
    from scipy.sparse import csc_matrix, csr_matrix

    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    y_true_csr = csr_matrix(y_true)
    y_pred_csr = csr_matrix(y_pred)
    y_true_csc = csc_matrix(y_true)
    y_pred_csc = csc_matrix(y_pred)

    y_true_t = mt.tensor(y_true)
    y_pred_t = mt.tensor(y_pred)

    # cross test different types
    sample_weight = np.array([2, 1, 3])
    real_cm = [[[1, 0], [1, 1]], [[1, 0], [1, 1]], [[0, 2], [1, 0]]]
    trues = [y_true_t, y_true_csr, y_true_csc]
    preds = [y_pred_t, y_pred_csr, y_pred_csc]

    for y_true_tmp in trues:
        for y_pred_tmp in preds:
            cm = multilabel_confusion_matrix(y_true_tmp, y_pred_tmp).fetch()
            assert_array_equal(cm, real_cm)

    # test support for samplewise
    cm = multilabel_confusion_matrix(y_true_t, y_pred_t, samplewise=True).fetch()
    assert_array_equal(cm, [[[1, 0], [1, 1]], [[1, 1], [0, 1]], [[0, 1], [2, 0]]])

    # test support for labels
    cm = multilabel_confusion_matrix(y_true_t, y_pred_t, labels=[2, 0]).fetch()
    assert_array_equal(cm, [[[0, 2], [1, 0]], [[1, 0], [1, 1]]])

    # test support for labels with samplewise
    cm = multilabel_confusion_matrix(
        y_true_t, y_pred_t, labels=[2, 0], samplewise=True
    ).fetch()
    assert_array_equal(cm, [[[0, 0], [1, 1]], [[1, 1], [0, 0]], [[0, 1], [1, 0]]])

    # test support for sample_weight with sample_wise
    cm = multilabel_confusion_matrix(
        y_true_t, y_pred_t, sample_weight=sample_weight, samplewise=True
    ).fetch()
    assert_array_equal(cm, [[[2, 0], [2, 2]], [[1, 1], [0, 1]], [[0, 3], [6, 0]]])


def test_multilabel_confusion_matrix_errors(setup):
    y_true = mt.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = mt.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])

    # Bad sample_weight
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        multilabel_confusion_matrix(y_true, y_pred, sample_weight=[1, 2])
    with pytest.raises(ValueError, match="should be a 1d array"):
        multilabel_confusion_matrix(
            y_true, y_pred, sample_weight=[[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        )

    # Bad labels
    err_msg = r"All labels must be in \[0, n labels\)"
    with pytest.raises(ValueError, match=err_msg):
        multilabel_confusion_matrix(y_true, y_pred, labels=[-1])
    err_msg = r"All labels must be in \[0, n labels\)"
    with pytest.raises(ValueError, match=err_msg):
        multilabel_confusion_matrix(y_true, y_pred, labels=[3])

    # Using samplewise outside multilabel
    with pytest.raises(ValueError, match="Samplewise metrics"):
        multilabel_confusion_matrix([0, 1, 2], [1, 2, 0], samplewise=True)

    # Bad y_type
    err_msg = "multiclass-multioutput is not supported"
    with pytest.raises(ValueError, match=err_msg):
        multilabel_confusion_matrix([[0, 1, 2], [2, 1, 0]], [[1, 2, 0], [1, 0, 2]])


@pytest.mark.parametrize("average", ["macro", "micro", "weighted", "samples"])
def test_precision_recall_f1_no_labels_check_warnings(setup, average):
    y_true = mt.zeros((20, 3))
    y_pred = mt.zeros_like(y_true)

    func = precision_recall_fscore_support
    with pytest.warns(UndefinedMetricWarning):
        p, r, f, s = func(y_true, y_pred, average=average, beta=1.0)
        p, r, f = fetch(execute(p, r, f))

    assert_almost_equal(p, 0)
    assert_almost_equal(r, 0)
    assert_almost_equal(f, 0)
    assert s is None

    with pytest.warns(UndefinedMetricWarning):
        fbeta = fetch(execute(fbeta_score(y_true, y_pred, average=average, beta=1.0)))

    assert_almost_equal(fbeta, 0)


def test_precision_recall_f1_score_multiclass(setup):
    # Test Precision Recall and F1 Score for multiclass classification task
    y_true, y_pred, _ = make_prediction(binary=False)
    y_true = mt.tensor(y_true, chunk_size=40)
    y_pred = mt.tensor(y_pred, chunk_size=40)

    # compute scores with default labels introspection
    p, r, f, s = fetch(
        execute(precision_recall_fscore_support(y_true, y_pred, average=None))
    )
    assert_array_almost_equal(p, [0.83, 0.33, 0.42], 2)
    assert_array_almost_equal(r, [0.79, 0.09, 0.90], 2)
    assert_array_almost_equal(f, [0.81, 0.15, 0.57], 2)
    assert_array_equal(s, [24, 31, 20])

    # averaging tests
    ps = fetch(execute(precision_score(y_true, y_pred, pos_label=1, average="micro")))
    assert_array_almost_equal(ps, 0.53, 2)

    rs = fetch(execute(recall_score(y_true, y_pred, average="micro")))
    assert_array_almost_equal(rs, 0.53, 2)

    fs = fetch(execute(f1_score(y_true, y_pred, average="micro")))
    assert_array_almost_equal(fs, 0.53, 2)

    ps = fetch(execute(precision_score(y_true, y_pred, average="macro")))
    assert_array_almost_equal(ps, 0.53, 2)

    rs = fetch(execute(recall_score(y_true, y_pred, average="macro")))
    assert_array_almost_equal(rs, 0.60, 2)

    fs = fetch(execute(f1_score(y_true, y_pred, average="macro")))
    assert_array_almost_equal(fs, 0.51, 2)

    ps = fetch(execute(precision_score(y_true, y_pred, average="weighted")))
    assert_array_almost_equal(ps, 0.51, 2)

    rs = fetch(execute(recall_score(y_true, y_pred, average="weighted")))
    assert_array_almost_equal(rs, 0.53, 2)

    fs = fetch(execute(f1_score(y_true, y_pred, average="weighted")))
    assert_array_almost_equal(fs, 0.47, 2)

    with pytest.raises(ValueError):
        precision_score(y_true, y_pred, average="samples")
    with pytest.raises(ValueError):
        recall_score(y_true, y_pred, average="samples")
    with pytest.raises(ValueError):
        f1_score(y_true, y_pred, average="samples")
    with pytest.raises(ValueError):
        fbeta_score(y_true, y_pred, average="samples", beta=0.5)

    # same prediction but with and explicit label ordering
    p, r, f, s = fetch(
        execute(
            precision_recall_fscore_support(
                y_true, y_pred, labels=[0, 2, 1], average=None
            )
        )
    )
    assert_array_almost_equal(p, [0.83, 0.41, 0.33], 2)
    assert_array_almost_equal(r, [0.79, 0.90, 0.10], 2)
    assert_array_almost_equal(f, [0.81, 0.57, 0.15], 2)
    assert_array_equal(s, [24, 20, 31])


@pytest.mark.parametrize("average", ["samples", "micro", "macro", "weighted", None])
def test_precision_refcall_f1_score_multilabel_unordered_labels(setup, average):
    # test that labels need not be sorted in the multilabel case
    y_true = mt.array([[1, 1, 0, 0]])
    y_pred = mt.array([[0, 0, 1, 1]])
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[3, 0, 1, 2], warn_for=[], average=average
    )
    p, r, f = fetch(execute(p, r, f))
    assert_array_equal(p, 0)
    assert_array_equal(r, 0)
    assert_array_equal(f, 0)
    if average is None:
        assert_array_equal(s, [0, 1, 1, 0])


def test_precision_recall_f1_score_binary_averaged(setup):
    y_true = mt.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1], chunk_size=10)
    y_pred = mt.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1], chunk_size=10)

    # compute scores with default labels introspection
    ps, rs, fs, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    ps, rs, fs = fetch(execute(ps, rs, fs))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    p, r, f = fetch(execute(p, r, f))
    assert p == np.mean(ps)
    assert r == np.mean(rs)
    assert f == np.mean(fs)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    p, r, f = fetch(execute(p, r, f))
    support = np.bincount(y_true).execute().fetch()
    assert p == np.average(ps, weights=support)
    assert r == np.average(rs, weights=support)
    assert f == np.average(fs, weights=support)


def test_zero_precision_recall(setup):
    # Check that pathological cases do not bring NaNs

    old_error_settings = np.seterr(all="raise")

    try:
        y_true = mt.array([0, 1, 2, 0, 1, 2], chunk_size=4)
        y_pred = mt.array([2, 0, 1, 1, 2, 0], chunk_size=4)

        assert_almost_equal(
            precision_score(y_true, y_pred, average="macro").execute().fetch(), 0.0, 2
        )
        assert_almost_equal(
            recall_score(y_true, y_pred, average="macro").execute().fetch(), 0.0, 2
        )
        assert_almost_equal(
            f1_score(y_true, y_pred, average="macro").execute().fetch(), 0.0, 2
        )

    finally:
        np.seterr(**old_error_settings)


def test_precision_recall_f_binary_single_class(setup):
    # Test precision, recall and F-scores behave with a single positive or
    # negative class
    # Such a case may occur with non-stratified cross-validation
    assert 1.0 == fetch(execute(precision_score([1, 1], [1, 1])))
    assert 1.0 == fetch(execute(recall_score([1, 1], [1, 1])))
    assert 1.0 == fetch(execute(f1_score([1, 1], [1, 1])))
    assert 1.0 == fetch(execute(fbeta_score([1, 1], [1, 1], beta=0)))

    assert 0.0 == fetch(execute(precision_score([-1, -1], [-1, -1])))
    assert 0.0 == fetch(execute(recall_score([-1, -1], [-1, -1])))
    assert 0.0 == fetch(execute(f1_score([-1, -1], [-1, -1])))
    assert 0.0 == fetch(execute(fbeta_score([-1, -1], [-1, -1], beta=float("inf"))))
    assert fetch(
        execute(fbeta_score([-1, -1], [-1, -1], beta=float("inf")))
    ) == pytest.approx(fetch(execute(fbeta_score([-1, -1], [-1, -1], beta=1e5))))
