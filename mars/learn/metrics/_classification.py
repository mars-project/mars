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
