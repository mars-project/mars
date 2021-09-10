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

from ... import opcodes as OperandDef
from ... import tensor as mt
from ...core import ENTITY_TYPE, recursive_tile
from ...core.context import get_context
from ...serialization.serializables import AnyField, BoolField, KeyField
from ...tensor.core import TensorOrder
from ..operands import LearnOperand, LearnOperandMixin, OutputType
from ..preprocessing import LabelBinarizer
from ..utils import check_array, check_consistent_length
from ._check_targets import _check_targets


class AccuracyScore(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.ACCURACY_SCORE

    _y_true = AnyField('y_true')
    _y_pred = AnyField('y_pred')
    _normalize = BoolField('normalize')
    _sample_weight = AnyField('sample_weight')
    _type_true = KeyField('type_true')

    def __init__(self, y_true=None, y_pred=None, normalize=None,
                 sample_weight=None, type_true=None, **kw):
        super().__init__(_y_true=y_true, _y_pred=y_pred,
                         _normalize=normalize, _sample_weight=sample_weight,
                         _type_true=type_true, **kw)
        self.output_types = [OutputType.tensor]

    @property
    def y_true(self):
        return self._y_true

    @property
    def y_pred(self):
        return self._y_pred

    @property
    def normalize(self):
        return self._normalize

    @property
    def sample_weight(self):
        return self._sample_weight

    @property
    def type_true(self):
        return self._type_true

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        if self._y_true is not None:
            self._y_true = next(inputs_iter)
        if self._y_pred is not None:
            self._y_pred = next(inputs_iter)
        if self._type_true is not None:
            self._type_true = next(inputs_iter)
        if isinstance(self._sample_weight, ENTITY_TYPE):
            self._sample_weight = next(inputs_iter)

    def __call__(self, y_true, y_pred):
        type_true, y_true, y_pred = _check_targets(y_true, y_pred)
        self._type_true = type_true
        inputs = [y_true, y_pred, type_true]
        if isinstance(self._sample_weight, ENTITY_TYPE):
            inputs.append(self._sample_weight)

        dtype = np.dtype(float) if self._normalize else np.result_type(y_true, y_pred)
        return self.new_tileable(inputs, dtype=dtype,
                                 shape=(), order=TensorOrder.C_ORDER)

    @classmethod
    def tile(cls, op):
        # make sure type_true executed first
        chunks = [op.type_true.chunks[0]]
        yield chunks

        ctx = get_context()
        type_true = ctx.get_chunks_result([chunks[0].key])[0]

        y_true, y_pred = op.y_true, op.y_pred
        if type_true.item().startswith('multilabel'):
            differing_labels = mt.count_nonzero(y_true - y_pred, axis=1)
            score = mt.equal(differing_labels, 0)
        else:
            score = mt.equal(y_true, y_pred)

        result = _weighted_sum(score, op.sample_weight, op.normalize)
        return [(yield from recursive_tile(result))]


def _weighted_sum(sample_score, sample_weight, normalize=False):
    if normalize:
        return mt.average(sample_score, weights=sample_weight)
    elif sample_weight is not None:
        return mt.dot(sample_score, sample_weight)
    else:
        return sample_score.sum()


def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None,
                   session=None, run_kwargs=None):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Read more in the :ref:`User Guide <accuracy_score>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator tensor / sparse tensor
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator tensor / sparse tensor
        Predicted labels, as returned by a classifier.

    normalize : bool, optional (default=True)
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    score : float
        If ``normalize == True``, return the fraction of correctly
        classified samples (float), else returns the number of correctly
        classified samples (int).

        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.

    See also
    --------
    jaccard_score, hamming_loss, zero_one_loss

    Notes
    -----
    In binary and multiclass classification, this function is equal
    to the ``jaccard_score`` function.

    Examples
    --------
    >>> from mars.learn.metrics import accuracy_score
    >>> y_pred = [0, 2, 1, 3]
    >>> y_true = [0, 1, 2, 3]
    >>> accuracy_score(y_true, y_pred).execute()
    0.5
    >>> accuracy_score(y_true, y_pred, normalize=False).execute()
    2

    In the multilabel case with binary label indicators:

    >>> import mars.tensor as mt
    >>> accuracy_score(mt.array([[0, 1], [1, 1]]), mt.ones((2, 2))).execute()
    0.5
    """

    # Compute accuracy for each possible representation
    op = AccuracyScore(y_true=y_true, y_pred=y_pred, normalize=normalize,
                       sample_weight=sample_weight)
    score = op(y_true, y_pred)
    return score.execute(session=session, **(run_kwargs or dict()))


def log_loss(y_true, y_pred, *, eps=1e-15, normalize=True, sample_weight=None,
             labels=None):
    r"""Log loss, aka logistic loss or cross-entropy loss.

    This is the loss function used in (multinomial) logistic regression
    and extensions of it such as neural networks, defined as the negative
    log-likelihood of a logistic model that returns ``y_pred`` probabilities
    for its training data ``y_true``.
    The log loss is only defined for two or more labels.
    For a single sample with true label :math:`y \in \{0,1\}` and
    and a probability estimate :math:`p = \operatorname{Pr}(y = 1)`, the log
    loss is:

    .. math::
        L_{\log}(y, p) = -(y \log (p) + (1 - y) \log (1 - p))

    Read more in the :ref:`User Guide <log_loss>`.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels for n_samples samples.

    y_pred : array-like of float, shape = (n_samples, n_classes) or (n_samples,)
        Predicted probabilities, as returned by a classifier's
        predict_proba method. If ``y_pred.shape = (n_samples,)``
        the probabilities provided are assumed to be that of the
        positive class. The labels in ``y_pred`` are assumed to be
        ordered alphabetically, as done by
        :class:`preprocessing.LabelBinarizer`.

    eps : float, default=1e-15
        Log loss is undefined for p=0 or p=1, so probabilities are
        clipped to max(eps, min(1 - eps, p)).

    normalize : bool, default=True
        If true, return the mean loss per sample.
        Otherwise, return the sum of the per-sample losses.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    labels : array-like, default=None
        If not provided, labels will be inferred from y_true. If ``labels``
        is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
        assumed to be binary and are inferred from ``y_true``.

    Returns
    -------
    loss : float

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).

    Examples
    --------
    >>> from mars.learn.metrics import log_loss
    >>> log_loss(["spam", "ham", "ham", "spam"],
    ...          [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
    0.21616...

    References
    ----------
    C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer,
    p. 209.
    """
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_pred, y_true, sample_weight)

    lb = LabelBinarizer()

    if labels is not None:
        lb.fit(labels)
    else:
        lb.fit(y_true)

    if len(lb.classes_) == 1:
        if labels is None:
            raise ValueError('y_true contains only one label ({0}). Please '
                             'provide the true labels explicitly through the '
                             'labels argument.'.format(lb.classes_[0].fetch()))
        else:
            raise ValueError('The labels array needs to contain at least two '
                             'labels for log_loss, '
                             'got {0}.'.format(lb.classes_.fetch()))

    transformed_labels = lb.transform(y_true)

    if transformed_labels.shape[1] == 1:
        transformed_labels = mt.append(1 - transformed_labels,
                                       transformed_labels, axis=1)

    # Clipping
    y_pred = mt.clip(y_pred, eps, 1 - eps)

    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if y_pred.ndim == 1:  # pragma: no cover
        y_pred = y_pred[:, mt.newaxis]
    if y_pred.shape[1] == 1:  # pragma: no cover
        y_pred = mt.append(1 - y_pred, y_pred, axis=1)

    # Check if dimensions are consistent.
    transformed_labels = check_array(transformed_labels)
    if len(lb.classes_) != y_pred.shape[1]:
        if labels is None:
            raise ValueError("y_true and y_pred contain different number of "
                             "classes {0}, {1}. Please provide the true "
                             "labels explicitly through the labels argument. "
                             "Classes found in "
                             "y_true: {2}".format(transformed_labels.shape[1],
                                                  y_pred.shape[1],
                                                  lb.classes_.fetch()))
        else:
            raise ValueError('The number of classes in labels is different '
                             'from that in y_pred. Classes found in '
                             'labels: {0}'.format(lb.classes_.fetch()))

    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, mt.newaxis]
    loss = -(transformed_labels * mt.log(y_pred)).sum(axis=1)

    return _weighted_sum(loss, sample_weight, normalize).execute()
