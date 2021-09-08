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
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._mocking import MockDataFrame

from .... import tensor as mt
from ....lib.sparse import SparseNDArray
from .. import accuracy_score, log_loss
from .._classification import _check_targets


IND = 'multilabel-indicator'
MC = 'multiclass'
BIN = 'binary'
CNT = 'continuous'
MMC = 'multiclass-multioutput'
MCN = 'continuous-multioutput'
# all of length 3
EXAMPLES = [
    (IND, np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]])),
    # must not be considered binary
    (IND, np.array([[0, 1], [1, 0], [1, 1]])),
    (MC, [2, 3, 1]),
    (BIN, [0, 1, 1]),
    (CNT, [0., 1.5, 1.]),
    (MC, np.array([[2], [3], [1]])),
    (BIN, np.array([[0], [1], [1]])),
    (CNT, np.array([[0.], [1.5], [1.]])),
    (MMC, np.array([[0, 2], [1, 3], [2, 3]])),
    (MCN, np.array([[0.5, 2.], [1.1, 3.], [2., 3.]])),
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


@pytest.mark.parametrize('type1, y1', EXAMPLES)
@pytest.mark.parametrize('type2, y2', EXAMPLES)
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
        merged_type, y1out, y2out = \
            _check_targets(y1, y2).execute().fetch()
        assert merged_type == expected
        if merged_type.startswith('multilabel'):
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

    score = accuracy_score(mt.tensor(y_true), mt.tensor(y_pred),
                           sample_weight=mt.tensor(sample_weight), normalize=False)
    result = score.execute().fetch()
    expected = sklearn_accuracy_score(y_true, y_pred, sample_weight=sample_weight,
                                      normalize=False)
    assert pytest.approx(result) == expected


def test_log_loss(setup):
    # binary case with symbolic labels ("no" < "yes")
    y_true = ["no", "no", "no", "yes", "yes", "yes"]
    y_pred = mt.array([[0.5, 0.5], [0.1, 0.9], [0.01, 0.99],
                       [0.9, 0.1], [0.75, 0.25], [0.001, 0.999]])
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
    y_pred = np.asarray(y_pred) > .5
    loss = log_loss(y_true, y_pred, normalize=True, eps=.1).fetch()
    assert_almost_equal(loss, log_loss(y_true, np.clip(y_pred, .1, .9)).fetch())

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
    error_str = (r'y_true contains only one label \(2\). Please provide '
                 r'the true labels explicitly through the labels argument.')
    with pytest.raises(ValueError, match=error_str):
        log_loss(y_true, y_pred)
    error_str = (r'The labels array needs to contain at least two '
                 r'labels for log_loss, got \[1\].')
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
