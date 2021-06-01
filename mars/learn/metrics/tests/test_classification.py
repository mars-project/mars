# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
try:
    import sklearn
    from sklearn.metrics import accuracy_score as sklearn_accuracy_score
except ImportError:  # pragma: no cover
    sklearn = None

import mars.tensor as mt
from mars.config import option_context
from mars.learn.metrics import accuracy_score
from mars.learn.metrics._classification import _check_targets
from mars.lib.sparse import SparseNDArray
from mars.tests import new_test_session


@pytest.fixture(scope='module')
def setup():
    sess = new_test_session(default=True)
    with option_context({'show_progress': False}):
        try:
            yield sess
        finally:
            sess.stop_server()


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


@pytest.mark.skipif(sklearn is None, reason='scikit-learn not installed')
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
