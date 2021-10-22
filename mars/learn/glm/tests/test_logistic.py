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

import pytest
from sklearn.datasets import load_iris

from .._logistic import _check_solver, _check_multi_class, LogisticRegression


# general data load
X, y = load_iris(return_X_y=True)


def test_check_solver(setup):
    all_solvers = ["SGD"]
    for solver in all_solvers:
        checked_solver = _check_solver(solver)
        assert checked_solver == solver

    invalid_solver = "Newton"
    error_msg = re.escape(
        "Logistic Regression supports only solvers in %s, "
        "got %s." % (all_solvers, invalid_solver)
    )

    with pytest.raises(ValueError, match=error_msg):
        _check_solver(invalid_solver)


def test_check_multi_class(setup):
    all_multi_class = ["auto", "multinomial", "ovr"]
    solver = "SGD"

    for multi_class in all_multi_class:
        checked_multi_class = _check_multi_class(multi_class, solver, 2)
        assert checked_multi_class == "multinomial"

    error_msg = re.escape(
        "Solver %s does not support "
        "an ovr backend with number of classes "
        "larger than 2." % solver
    )
    with pytest.raises(ValueError, match=error_msg):
        _check_multi_class("ovr", solver, 3)

    invalid_multi_class = "multiovr"
    error_msg = re.escape(
        "multi_class should be 'multinomial', "
        "'ovr' or 'auto'. Got %s." % invalid_multi_class
    )
    with pytest.raises(ValueError, match=error_msg):
        _check_multi_class(invalid_multi_class, solver, 3)


def test_invalid_penalty(setup):
    error_msg = re.escape("Only support L2 penalty.")

    with pytest.raises(NotImplementedError, match=error_msg):
        model = LogisticRegression(penalty="l1")
        model.fit(X, y)


def test_invalid_C(setup):
    invalid_C = -1
    error_msg = re.escape("Penalty term must be positive; got (C=%r)" % invalid_C)

    with pytest.raises(ValueError, match=error_msg):
        model = LogisticRegression(C=invalid_C)
        model.fit(X, y)


def test_invalid_max_iter(setup):
    invalid_max_iter = -1
    error_msg = re.escape(
        "Maximum number of iteration must be positive;"
        " got (max_iter=%r)" % invalid_max_iter
    )

    with pytest.raises(ValueError, match=error_msg):
        model = LogisticRegression(max_iter=invalid_max_iter)
        model.fit(X, y)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_logistic_regression_no_converge(setup, fit_intercept):
    # quite slow in local tests, so set max_iter=1
    # suggested max_iter >= 10
    model = LogisticRegression(fit_intercept=fit_intercept, max_iter=1)
    model.fit(X, y)
    model.predict(X)
    model.score(X, y)
    model.predict_proba(X)
    model.predict_log_proba(X)

    error_msg = re.escape(
        "X has %d features per sample; expecting %d"
        % (X.shape[1], model.coef_.shape[1] - 1)
    )
    model.coef_ = model.coef_[:, :-1]
    with pytest.raises(ValueError, match=error_msg):
        model.predict(X)
