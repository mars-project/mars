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

from typing import Callable, Union

import numpy as np
from sklearn.base import (
    MetaEstimatorMixin,
    BaseEstimator as SklearnBaseEstimator,
    RegressorMixin as SklearnRegressorMixin,
    ClassifierMixin as SklearnClassifierMixin,
)

from .. import remote as mr
from .. import tensor as mt
from ..core import ENTITY_TYPE
from .base import BaseEstimator, RegressorMixin, ClassifierMixin
from .metrics import get_scorer
from .utils import copy_learned_attributes, check_array


def _wrap(estimator: SklearnBaseEstimator, method, X, y, **kwargs):
    X = X.fetch() if isinstance(X, ENTITY_TYPE) else X
    y = y.fetch() if isinstance(y, ENTITY_TYPE) else y
    return getattr(estimator, method)(X, y, **kwargs)


class ParallelPostFit(BaseEstimator, MetaEstimatorMixin):
    """
    Meta-estimator for parallel predict and transform.

    Parameters
    ----------
    estimator : Estimator
        The underlying estimator that is fit.

    scoring : string or callable, optional
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique)
        strings or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a
        single value. Metric functions returning a list/array of values
        can be wrapped into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        .. warning::

           If None, the estimator's default scorer (if available) is used.
           Most scikit-learn estimators will convert large Mars tensors to
           a single NumPy array, which may exhaust the memory of your worker.
           You probably want to always specify `scoring`.

    Notes
    -----

    .. warning::

       This class is not appropriate for parallel or distributed *training*
       on large datasets. For that, see :class:`Incremental`, which provides
       distributed (but sequential) training. If you're doing distributed
       hyperparameter optimization on larger-than-memory datasets, see
       :class:`mars.learn.model_selection.IncrementalSearch`.

    This estimator does not parallelize the training step. This simply calls
    the underlying estimators's ``fit`` method called and copies over the
    learned attributes to ``self`` afterwards.

    It is helpful for situations where your training dataset is relatively
    small (fits on a single machine) but you need to predict or transform
    a much larger dataset. ``predict``, ``predict_proba`` and ``transform``
    will be done in parallel (potentially distributed if you've connected
    to a Mars cluster).

    Note that many scikit-learn estimators already predict and transform in
    parallel. This meta-estimator may still be useful in those cases when your
    dataset is larger than memory, as the distributed scheduler will ensure the
    data isn't all read into memory at once.

    See Also
    --------
    Incremental
    mars.learn.model_selection.IncrementalSearch

    Examples
    --------
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> from sklearn.datasets import make_classification
    >>> import mars.tensor as mt
    >>> from mars.learn.wrappers import ParallelPostFit

    Make a small 1,000 sample 2 training dataset and fit normally.

    >>> X, y = make_classification(n_samples=1000, random_state=0)
    >>> clf = ParallelPostFit(estimator=GradientBoostingClassifier(),
    ...                       scoring='accuracy')
    >>> clf.fit(X, y)
    ParallelPostFit(estimator=GradientBoostingClassifier(...))

    >>> clf.classes_
    array([0, 1])

    Transform and predict return Mars outputs for Mars inputs.

    >>> X_big, y_big = make_classification(n_samples=100000,
                                           random_state=0)
    >>> X_big, y_big = mt.tensor(X_big), mt.tensor(y_big)
    >>> clf.predict(X_big)
    array([1, 0, 0, ..., 1, 0, 0])

    Which can be computed in parallel.

    >>> clf.predict_proba(X_big)
    array([[0.01780031, 0.98219969],
           [0.62199242, 0.37800758],
           [0.89059934, 0.10940066],
           ...,
           [0.03249968, 0.96750032],
           [0.951434  , 0.048566  ],
           [0.99527114, 0.00472886]])
    """

    def __init__(
        self,
        estimator: SklearnBaseEstimator = None,
        scoring: Union[str, Callable] = None,
    ):
        self.estimator = estimator
        self.scoring = scoring

    def _make_fit(self, method):
        def _fit(X, y=None, **kwargs):
            result = (
                mr.spawn(_wrap, args=(self.estimator, method, X, y), kwargs=kwargs)
                .execute()
                .fetch()
            )

            copy_learned_attributes(result, self)
            copy_learned_attributes(result, self.estimator)
            return self

        return _fit

    def fit(self, X, y=None, **kwargs):
        """
        Fit the underlying estimator.

        Parameters
        ----------
        X, y : array-like
        **kwargs
            Additional fit-kwargs for the underlying estimator.

        Returns
        -------
        self : object
        """
        return self._make_fit("fit")(X, y=y, **kwargs)

    def partial_fit(self, X, y=None, **kwargs):  # pragma: no cover
        return self._make_fit("partial_fit")(X, y=y, **kwargs)

    def _check_method(self, method):
        """
        Check if self.estimator has 'method'.

        Raises
        ------
        AttributeError
        """
        estimator = self.estimator
        if not hasattr(estimator, method):
            msg = "The wrapped estimator '{}' does not have a '{}' method.".format(
                estimator, method
            )
            raise AttributeError(msg)
        return getattr(estimator, method)

    def transform(self, X):
        """
        Transform block or partition-wise for Mars inputs.

        For Mars inputs, a Mars tensor is returned. For other
        inputs (NumPy array, pandas dataframe, scipy sparse matrix), the
        regular return value is returned.

        If the underlying estimator does not have a ``transform`` method, then
        an ``AttributeError`` is raised.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        transformed : array-like
        """
        self._check_method("transform")
        X = check_array(X)
        dtype = self.estimator.transform(np.zeros((1, X.shape[1]), dtype=X.dtype)).dtype
        return X.map_chunk(self.estimator.transform, dtype=dtype)

    def score(self, X, y):
        """
        Returns the score on the given data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
            return self.estimator.score(X, y)
        """

        scoring = self.scoring
        X = check_array(X)
        y = check_array(y, ensure_2d=False)

        if not scoring:
            if type(self.estimator).score in (
                RegressorMixin.score,
                SklearnRegressorMixin.score,
            ):  # pragma: no cover
                scoring = "r2"
            elif type(self.estimator).score in (
                ClassifierMixin.score,
                SklearnClassifierMixin.score,
            ):
                scoring = "accuracy"
        else:  # pragma: no cover
            scoring = self.scoring

        if scoring:
            scorer = get_scorer(scoring)
            return scorer(self, X, y).execute()
        else:  # pragma: no cover
            return mr.spawn(self.estimator.score, args=(X, y)).execute().fetch()

    def predict(self, X, execute=True):
        """
        Predict for X.

        For Mars inputs, a Mars tensor is returned. For other
        inputs (NumPy array, pandas dataframe, scipy sparse matrix), the
        regular return value is returned.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        y : array-like
        """

        self._check_method("predict")
        X = check_array(X)

        result = X.map_chunk(self.estimator.predict, dtype="int", shape=X.shape[:1])
        if execute:
            result.execute()
        return result

    def predict_proba(self, X, execute=True):
        """
        Probability estimates.

        For Mars inputs, a Mars tensor is returned. For other
        inputs (NumPy array, pandas dataframe, scipy sparse matrix), the
        regular return value is returned.

        If the underlying estimator does not have a ``predict_proba``
        method, then an ``AttributeError`` is raised.

        Parameters
        ----------
        X : array or dataframe

        Returns
        -------
        y : array-like
        """
        self._check_method("predict_proba")
        X = check_array(X)
        result = X.map_chunk(
            self.estimator.predict_proba,
            dtype="float",
            shape=(X.shape[0], len(self.estimator.classes_)),
        )
        if execute:
            result.execute()
        return result

    def predict_log_proba(self, X, execute=True):
        """
        Log of probability estimates.

        For Mars inputs, a Mars tensor is returned. For other
        inputs (NumPy array, pandas dataframe, scipy sparse matrix), the
        regular return value is returned.

        If the underlying estimator does not have a ``predict_proba``
        method, then an ``AttributeError`` is raised.

        Parameters
        ----------
        X : array or dataframe

        Returns
        -------
        y : array-like
        """

        self._check_method("predict_log_proba")
        result = mt.log(self.predict_proba(X, execute=False))
        if execute:
            result.execute()
        return result
