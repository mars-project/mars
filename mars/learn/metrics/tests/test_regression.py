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

from itertools import product

import numpy as np
import pytest
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_array_almost_equal,
)

from .... import tensor as mt
from .. import r2_score
from .._regresssion import _check_reg_targets


def test__check_reg_targets(setup):
    # All of length 3
    EXAMPLES = [
        ("continuous", [1, 2, 3], 1),
        ("continuous", [[1], [2], [3]], 1),
        ("continuous-multioutput", [[1, 1], [2, 2], [3, 1]], 2),
        ("continuous-multioutput", [[5, 1], [4, 2], [3, 1]], 2),
        ("continuous-multioutput", [[1, 3, 4], [2, 2, 2], [3, 1, 1]], 3),
    ]

    for (type1, y1, n_out1), (type2, y2, n_out2) in product(EXAMPLES, repeat=2):
        if type1 == type2 and n_out1 == n_out2:
            y_type, y_check1, y_check2, multioutput = _check_reg_targets(y1, y2, None)
            assert type1 == y_type
            if type1 == "continuous":
                assert_array_equal(y_check1, np.reshape(y1, (-1, 1)))
                assert_array_equal(y_check2, np.reshape(y2, (-1, 1)))
            else:
                assert_array_equal(y_check1, y1)
                assert_array_equal(y_check2, y2)
        else:
            with pytest.raises(ValueError):
                _check_reg_targets(y1, y2, None)


def test__check_reg_targets_exception(setup):
    invalid_multioutput = "this_value_is_not_valid"
    expected_message = (
        "Allowed 'multioutput' string values are.+"
        "You provided multioutput={!r}".format(invalid_multioutput)
    )
    with pytest.raises(ValueError, match=expected_message):
        _check_reg_targets([1, 2, 3], [[1], [2], [3]], invalid_multioutput)

    with pytest.raises(ValueError):
        _check_reg_targets([1, 2], [[1], [2]], multioutput=[0.4, 0.6])
    with pytest.raises(ValueError):
        _check_reg_targets([[1, 2], [3, 4]], [[1, 2], [3, 4]], multioutput=[0.4])


def test_r2_score(setup, n_samples=50):
    y_true = mt.arange(n_samples)
    y_pred = y_true + 1

    assert_almost_equal(r2_score(y_true, y_pred).fetch(), 0.995, 2)

    y_true = mt.array([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]])
    y_pred = mt.array([[0, 0, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]])

    error = r2_score(y_true, y_pred, multioutput="variance_weighted")
    assert_almost_equal(error.fetch(), 1.0 - 5.0 / 2)
    error = r2_score(y_true, y_pred, multioutput="uniform_average")
    assert_almost_equal(error.fetch(), -0.875)

    assert_almost_equal(r2_score([0.0, 1], [0.0, 1]).fetch(), 1.00, 2)
    assert_almost_equal(
        r2_score([0.0, 1], [0.0, 1], sample_weight=[0.5, 0.5]).fetch(), 1.00, 2
    )

    y_true = [[1, 2], [2.5, -1], [4.5, 3], [5, 7]]
    y_pred = [[1, 1], [2, -1], [5, 4], [5, 6.5]]

    r = r2_score(y_true, y_pred, multioutput="raw_values")

    assert_array_almost_equal(r, [0.95, 0.93], decimal=2)

    # mean_absolute_error and mean_squared_error are equal because
    # it is a binary problem.
    y_true = [[0, 0]] * 4
    y_pred = [[1, 1]] * 4
    r = r2_score(y_true, y_pred, multioutput="raw_values")
    assert_array_almost_equal(r, [0.0, 0.0], decimal=2)

    r = r2_score([[0, -1], [0, 1]], [[2, 2], [1, 1]], multioutput="raw_values")
    assert_array_almost_equal(r, [0, -3.5], decimal=2)
    assert (
        np.mean(r.fetch())
        == r2_score(
            [[0, -1], [0, 1]], [[2, 2], [1, 1]], multioutput="uniform_average"
        ).fetch()
    )

    # Checking for the condition in which both numerator and denominator is
    # zero.
    y_true = [[1, 3], [-1, 2]]
    y_pred = [[1, 4], [-1, 1]]
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    assert_array_almost_equal(r2, [1.0, -3.0], decimal=2)
    assert (
        np.mean(r2.fetch())
        == r2_score(y_true, y_pred, multioutput="uniform_average").fetch()
    )

    y_true = [[1, 2], [2.5, -1], [4.5, 3], [5, 7]]
    y_pred = [[1, 1], [2, -1], [5, 4], [5, 6.5]]

    rw = r2_score(y_true, y_pred, multioutput=[0.4, 0.6])

    assert_almost_equal(rw.fetch(), 0.94, decimal=2)

    y_true = [0]
    y_pred = [1]
    warning_msg = "not well-defined with less than two samples."

    # Trigger the warning
    with pytest.warns(UndefinedMetricWarning, match=warning_msg):
        score = r2_score(y_true, y_pred)
        assert np.isnan(score)
