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
import warnings
from typing import Union

import numpy as np
from sklearn.base import OutlierMixin
from sklearn.tree import ExtraTreeRegressor
from sklearn.utils import (
    check_array as sklearn_check_array,
    gen_batches as sklearn_gen_batches,
)

from ... import tensor as mt
from ...deploy.oscar.session import execute
from ...lib.sparse import issparse
from ...tensor.utils import check_random_state
from ..utils import get_chunk_n_rows, convert_to_tensor_or_dataframe
from ..utils.validation import _num_samples, check_is_fitted
from ._bagging import BaseBagging, BaggingPredictionOperand, PredictionType


def _average_path_length(n_samples_leaf):
    """
    The average path length in a n_samples iTree, which is equal to
    the average path length of an unsuccessful BST search since the
    latter has the same structure as an isolation tree.
    Parameters
    ----------
    n_samples_leaf : array-like of shape (n_samples,)
        The number of training samples in each test sample leaf, for
        each estimators.

    Returns
    -------
    average_path_length : ndarray of shape (n_samples,)
    """

    n_samples_leaf = sklearn_check_array(n_samples_leaf, ensure_2d=False)

    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.0
    average_path_length[mask_2] = 1.0
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)


def _tree_decision_function(instance, estimator):
    n_samples = _num_samples(instance)

    # We get as many rows as possible within our working_memory budget
    # (defined by sklearn.get_config()['working_memory']) to store
    # self._max_features in each row during computation.
    #
    # Note:
    #  - this will get at least 1 row, even if 1 row of score will
    #    exceed working_memory.
    #  - this does only account for temporary memory usage while loading
    #    the data needed to compute the scores -- the returned scores
    #    themselves are 1D.

    chunk_n_rows = get_chunk_n_rows(
        row_bytes=16 * instance.shape[1], max_n_rows=n_samples
    )
    slices = sklearn_gen_batches(n_samples, chunk_n_rows)

    scores = np.zeros(n_samples, order="f")

    for sl in slices:
        # compute score on the slices of test samples:
        scores[sl] = _compute_score_samples(instance[sl], estimator)

    return scores


def _compute_score_samples(instance, estimator):
    leaves_index = estimator.apply(instance)
    node_indicator = estimator.decision_path(instance)
    n_samples_leaf = estimator.tree_.n_node_samples[leaves_index]

    return (
        np.ravel(node_indicator.sum(axis=1))
        + _average_path_length(n_samples_leaf)
        - 1.0
    )


class IsolationForest(OutlierMixin, BaseBagging):
    """
    Isolation Forest Algorithm.

    Return the anomaly score of each sample using the IsolationForest algorithm

    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.

    Read more in the :ref:`User Guide <isolation_forest>`.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators in the ensemble.

    max_samples : "auto", int or float, default="auto"
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.

            - If 'auto', the threshold is determined as in the
              original paper.
            - If float, the contamination should be in the range (0, 0.5].

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    Attributes
    ----------
    base_estimator_ : ExtraTreeRegressor instance
        The child estimator template used to create the collection of
        fitted sub-estimators.

    estimators_ : list of ExtraTreeRegressor instances
        The collection of fitted sub-estimators.

    estimators_features_ : list of ndarray
        The subset of drawn features for each base estimator.

    max_samples_ : int
        The actual number of samples.

    offset_ : float
        Offset used to define the decision function from the raw scores. We
        have the relation: ``decision_function = score_samples - offset_``.
        ``offset_`` is defined as follows. When the contamination parameter is
        set to "auto", the offset is equal to -0.5 as the scores of inliers are
        close to 0 and the scores of outliers are close to -1. When a
        contamination parameter different than "auto" is provided, the offset
        is defined in such a way we obtain the expected number of outliers
        (samples with decision function < 0) in training.

    Notes
    -----
    The implementation is based on an ensemble of ExtraTreeRegressor. The
    maximum depth of each tree is set to ``ceil(log_2(n))`` where
    :math:`n` is the number of samples used to build the tree
    (see (Liu et al., 2008) for more details).

    References
    ----------
    .. [1] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
           Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.
    .. [2] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation-based
           anomaly detection." ACM Transactions on Knowledge Discovery from
           Data (TKDD) 6.1 (2012): 3.

    See Also
    ----------
    sklearn.covariance.EllipticEnvelope : An object for detecting outliers in a
        Gaussian distributed dataset.
    sklearn.svm.OneClassSVM : Unsupervised Outlier Detection.
        Estimate the support of a high-dimensional distribution.
        The implementation is based on libsvm.
    sklearn.neighbors.LocalOutlierFactor : Unsupervised Outlier Detection
        using Local Outlier Factor (LOF).

    Examples
    --------
    >>> from mars.learn.ensemble import IsolationForest
    >>> X = [[-1.1], [0.3], [0.5], [100]]
    >>> clf = IsolationForest(random_state=0).fit(X)
    >>> clf.predict([[0.1], [0], [90]])
    array([ 1,  1, -1])
    """

    contamination: Union[str, float]

    def __init__(
        self,
        *,
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=None,
        warm_start=False,
    ):
        super().__init__(
            base_estimator=ExtraTreeRegressor(
                max_features=1, splitter="random", random_state=random_state
            ),
            # here above max_features has no links with self.max_features
            bootstrap=bootstrap,
            bootstrap_features=False,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            warm_start=warm_start,
            random_state=random_state,
        )
        self.contamination = contamination

    def fit(
        self, X, y=None, sample_weight=None, session=None, run_kwargs=None
    ) -> "IsolationForest":
        """
        Fit estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : IsolationForest
            Fitted estimator.
        """
        run_kwargs = run_kwargs or dict()
        X = convert_to_tensor_or_dataframe(X)
        if issparse(X):  # pragma: no cover
            raise NotImplementedError

        if np.isnan(X.shape[0]):
            execute(X, session=session, **run_kwargs)

        rnd = check_random_state(self.random_state)
        y = rnd.uniform(size=X.shape[0])

        # ensure that max_sample is in [1, n_samples]:
        n_samples = X.shape[0]

        if self.contamination != "auto":
            if not (0.0 < self.contamination <= 0.5):
                raise ValueError(
                    "contamination must be in (0, 0.5], got: %f" % self.contamination
                )

        if isinstance(self.max_samples, str):
            if self.max_samples == "auto":
                max_samples = min(256, n_samples)
            else:
                raise ValueError(
                    "max_samples (%s) is not supported."
                    'Valid choices are: "auto", int or'
                    "float" % self.max_samples
                )

        elif isinstance(self.max_samples, numbers.Integral):
            if self.max_samples > n_samples:
                warnings.warn(
                    "max_samples (%s) is greater than the "
                    "total number of samples (%s). max_samples "
                    "will be set to n_samples for estimation."
                    % (self.max_samples, n_samples)
                )
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # float
            if not 0.0 < self.max_samples <= 1.0:
                raise ValueError(
                    "max_samples must be in (0, 1], got %r" % self.max_samples
                )
            max_samples = int(self.max_samples * X.shape[0])

        self.max_samples_ = max_samples
        max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
        super()._fit(
            X,
            y,
            sample_weight=sample_weight,
            max_samples=max_samples,
            estimator_params=dict(max_samples=max_samples, max_depth=max_depth),
        )

        if self.contamination == "auto":
            # 0.5 plays a special role as described in the original paper.
            # we take the opposite as we consider the opposite of their score.
            self.offset_ = -0.5
            return self

        # else, define offset_ wrt contamination parameter
        self.offset_ = execute(
            mt.percentile(self._score_samples(X), 100.0 * self.contamination),
            session=session,
            **(run_kwargs or dict()),
        ).fetch(session=session, **(run_kwargs or dict()))

        return self

    def predict(self, X, session=None, run_kwargs=None):
        """
        Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            For each observation, tells whether or not (+1 or -1) it should
            be considered as an inlier according to the fitted model.
        """
        check_is_fitted(self)
        is_inlier = mt.ones(X.shape[0], dtype=int)
        is_inlier[self._decision_function(X) < 0] = -1
        return execute(is_inlier, session=session, **(run_kwargs or dict()))

    def _decision_function(self, X):
        return self._score_samples(X) - self.offset_

    def decision_function(self, X, session=None, run_kwargs=None):
        """
        Average anomaly score of X of the base classifiers.

        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.

        The measure of normality of an observation given a tree is the depth
        of the leaf containing this observation, which is equivalent to
        the number of splittings required to isolate this point. In case of
        several observations n_left in the leaf, the average path length of
        a n_left samples isolation tree is added.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal. Negative scores represent outliers,
            positive scores represent inliers.
        """
        # We subtract self.offset_ to make 0 be the threshold value for being
        # an outlier:

        decision_func = self._decision_function(X)
        return execute(decision_func, session=session, **(run_kwargs or dict()))

    def _score_samples(self, X):
        check_is_fitted(self)
        X = convert_to_tensor_or_dataframe(X)
        predict_op = BaggingPredictionOperand(
            prediction_type=PredictionType.DECISION_FUNCTION,
            decision_function=_tree_decision_function,
            calc_means=False,
        )
        depths = predict_op(X, self.estimators_, self.estimator_features_).sum(axis=1)
        denominator = self.estimators_.shape[0] * _average_path_length(
            [self.max_samples_]
        )
        return -(
            2
            ** (
                # For a single training sample, denominator and depth are 0.
                # Therefore, we set the score manually to 1.
                -mt.divide(
                    depths,
                    denominator,
                    out=mt.ones_like(depths),
                    where=denominator != 0,
                )
            )
        )

    def score_samples(self, X, session=None, run_kwargs=None):
        """
        Opposite of the anomaly score defined in the original paper.

        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.

        The measure of normality of an observation given a tree is the depth
        of the leaf containing this observation, which is equivalent to
        the number of splittings required to isolate this point. In case of
        several observations n_left in the leaf, the average path length of
        a n_left samples isolation tree is added.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal.
        """
        scores = self._score_samples(X)
        return execute(scores, session=session, **(run_kwargs or dict()))
