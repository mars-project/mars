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

import numbers

from sklearn.utils.validation import _deprecate_positional_args

from ... import tensor as mt
from ..linear_model._base import LinearClassifierMixin
from ..base import BaseEstimator
from ..utils.validation import check_is_fitted
from ..utils.multiclass import check_classification_targets
from ..utils.extmath import softmax
from ._optimizers import gradient_descent


def _check_solver(solver):
    all_solvers = ["SGD"]
    if solver not in all_solvers:
        raise ValueError(
            "Logistic Regression supports only solvers in %s, got"
            " %s." % (all_solvers, solver)
        )
    return solver


def _check_multi_class(multi_class, solver, n_classes):
    if multi_class == "auto":
        return "multinomial"
    if multi_class == "ovr":
        if n_classes == 2:
            return "multinomial"
        else:
            raise ValueError(
                "Solver %s does not support "
                "an ovr backend with number of classes "
                "larger than 2." % solver
            )
    if multi_class == "multinomial":
        return "multinomial"

    raise ValueError(
        "multi_class should be 'multinomial', " "'ovr' or 'auto'. Got %s." % multi_class
    )


class LogisticRegression(LinearClassifierMixin, BaseEstimator):
    """
    Logistic Regression (aka logit, MaxEnt) classifier.

    In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
    scheme if the 'multi_class' option is set to 'ovr', and uses the
    cross-entropy loss if the 'multi_class' option is set to 'multinomial'.
    (Currently the 'multinomial' option is supported only by the 'lbfgs',
    'sag', 'saga' and 'newton-cg' solvers.)

    This class implements regularized logistic regression using the
    'liblinear' library, 'newton-cg', 'sag', 'saga' and 'lbfgs' solvers. **Note
    that regularization is applied by default**. It can handle both dense
    and sparse input. Use C-ordered arrays or CSR matrices containing 64-bit
    floats for optimal performance; any other input format will be converted
    (and copied).

    The 'newton-cg', 'sag', and 'lbfgs' solvers support only L2 regularization
    with primal formulation, or no regularization. The 'liblinear' solver
    supports both L1 and L2 regularization, with a dual formulation only for
    the L2 penalty. The Elastic-Net regularization is only supported by the
    'saga' solver.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
        only supported by the 'saga' solver. If 'none' (not supported by the
        liblinear solver), no regularization is applied.

        .. versionadded:: 0.19
           l1 penalty with SAGA solver (allowing 'multinomial' + L1)

    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. See :term:`Glossary <random_state>` for details.

    solver : SGD (stochastic gradient descent)

    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.

    multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.22
            Default changed from 'ovr' to 'auto' in 0.22.

    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    Attributes
    ----------

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `coef_` corresponds
        to outcome 1 (True) and `-coef_` corresponds to outcome 0 (False).

    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.

        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape (1,) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `intercept_`
        corresponds to outcome 1 (True) and `-intercept_` corresponds to
        outcome 0 (False).

    See Also
    --------
    SGDClassifier : Incrementally trained logistic regression (when given
        the parameter ``loss="log"``).
    LogisticRegressionCV : Logistic regression with built-in cross validation.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegression(random_state=0).fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    """

    @_deprecate_positional_args
    def __init__(
        self,
        penalty="l2",
        fit_intercept=False,
        C=100,
        batch_size=20,
        learning_rate=0.1,
        random_state=None,
        solver="SGD",
        max_iter=300,
        multi_class="auto",
        verbose=0,
    ):

        self.penalty = penalty
        self.fit_intercept = fit_intercept
        self.C = C
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.
        """
        # ========== Pre-check =============
        if self.penalty not in ["l2"]:
            raise NotImplementedError("Only support L2 penalty.")

        solver = _check_solver(self.solver)

        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)" % self.C)

        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError(
                "Maximum number of iteration must be positive;"
                " got (max_iter=%r)" % self.max_iter
            )

        _dtype = [mt.float64, mt.float32]

        X, y = self._validate_data(X, y, accept_sparse="csr", dtype=_dtype, order="C")

        check_classification_targets(y)

        self.classes_ = mt.unique(y)
        n_classes = self.classes_.execute().shape[0]
        multi_class = _check_multi_class(self.multi_class, solver, n_classes)

        # ========== Fit solver ============
        # Only support stochastic gradient descent for now
        if multi_class == "multinomial":
            if solver == "SGD":
                self.coef_ = gradient_descent(
                    X,
                    y,
                    learning_rate=self.learning_rate,
                    reg=(1 / self.C),
                    max_epochs=self.max_iter,
                    batch_size=self.batch_size,
                    fit_intercept=self.fit_intercept,
                    verbose=self.verbose,
                )
                self.coef_ = self.coef_.T

        # ========== Post processing =======
        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]
        else:
            self.intercept_ = mt.zeros(n_classes)

        return self

    def predict_proba(self, X):
        """
        Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        Else use a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self)
        decision = self.decision_function(X)

        return softmax(decision, copy=False).execute()

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return mt.log(self.predict_proba(X)).execute()
