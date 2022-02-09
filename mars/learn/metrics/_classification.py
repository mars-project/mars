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

import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning

from ... import execute, fetch
from ... import opcodes as OperandDef
from ... import tensor as mt
from ...core import ENTITY_TYPE, recursive_tile
from ...core.context import get_context
from ...serialization.serializables import AnyField, BoolField, KeyField
from ...tensor.core import TensorOrder
from ..operands import LearnOperand, LearnOperandMixin, OutputType
from ..preprocessing import LabelBinarizer, LabelEncoder
from ..utils import check_array, check_consistent_length, column_or_1d
from ..utils.multiclass import unique_labels
from ..utils.sparsefuncs import count_nonzero
from ._check_targets import _check_targets


class AccuracyScore(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.ACCURACY_SCORE

    _y_true = AnyField("y_true")
    _y_pred = AnyField("y_pred")
    _normalize = BoolField("normalize")
    _sample_weight = AnyField("sample_weight")
    _type_true = KeyField("type_true")

    def __init__(
        self,
        y_true=None,
        y_pred=None,
        normalize=None,
        sample_weight=None,
        type_true=None,
        **kw
    ):
        super().__init__(
            _y_true=y_true,
            _y_pred=y_pred,
            _normalize=normalize,
            _sample_weight=sample_weight,
            _type_true=type_true,
            **kw
        )
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
        return self.new_tileable(
            inputs, dtype=dtype, shape=(), order=TensorOrder.C_ORDER
        )

    @classmethod
    def tile(cls, op):
        # make sure type_true executed first
        chunks = [op.type_true.chunks[0]]
        yield chunks

        ctx = get_context()
        type_true = ctx.get_chunks_result([chunks[0].key])[0]

        y_true, y_pred = op.y_true, op.y_pred
        if type_true.item().startswith("multilabel"):
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


def accuracy_score(
    y_true, y_pred, normalize=True, sample_weight=None, session=None, run_kwargs=None
):
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
    op = AccuracyScore(
        y_true=y_true, y_pred=y_pred, normalize=normalize, sample_weight=sample_weight
    )
    score = op(y_true, y_pred)
    return score.execute(session=session, **(run_kwargs or dict()))


def log_loss(
    y_true, y_pred, *, eps=1e-15, normalize=True, sample_weight=None, labels=None
):
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
            raise ValueError(
                "y_true contains only one label ({0}). Please "
                "provide the true labels explicitly through the "
                "labels argument.".format(lb.classes_[0].fetch())
            )
        else:
            raise ValueError(
                "The labels array needs to contain at least two "
                "labels for log_loss, "
                "got {0}.".format(lb.classes_.fetch())
            )

    transformed_labels = lb.transform(y_true)

    if transformed_labels.shape[1] == 1:
        transformed_labels = mt.append(
            1 - transformed_labels, transformed_labels, axis=1
        )

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
            raise ValueError(
                "y_true and y_pred contain different number of "
                "classes {0}, {1}. Please provide the true "
                "labels explicitly through the labels argument. "
                "Classes found in "
                "y_true: {2}".format(
                    transformed_labels.shape[1], y_pred.shape[1], lb.classes_.fetch()
                )
            )
        else:
            raise ValueError(
                "The number of classes in labels is different "
                "from that in y_pred. Classes found in "
                "labels: {0}".format(lb.classes_.fetch())
            )

    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, mt.newaxis]
    loss = -(transformed_labels * mt.log(y_pred)).sum(axis=1)

    return _weighted_sum(loss, sample_weight, normalize).execute()


def multilabel_confusion_matrix(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    labels=None,
    samplewise=False,
    session=None,
    run_kwargs=None
):
    """
    Compute a confusion matrix for each class or sample.

    Compute class-wise (default) or sample-wise (samplewise=True) multilabel
    confusion matrix to evaluate the accuracy of a classification, and output
    confusion matrices for each class or sample.

    In multilabel confusion matrix :math:`MCM`, the count of true negatives
    is :math:`MCM_{:,0,0}`, false negatives is :math:`MCM_{:,1,0}`,
    true positives is :math:`MCM_{:,1,1}` and false positives is
    :math:`MCM_{:,0,1}`.

    Multiclass data will be treated as if binarized under a one-vs-rest
    transformation. Returned confusion matrices will be in the order of
    sorted unique labels in the union of (y_true, y_pred).

    Read more in the :ref:`User Guide <multilabel_confusion_matrix>`.

    Parameters
    ----------
    y_true : {array-like, sparse matrix} of shape (n_samples, n_outputs) or \
            (n_samples,)
        Ground truth (correct) target values.

    y_pred : {array-like, sparse matrix} of shape (n_samples, n_outputs) or \
            (n_samples,)
        Estimated targets as returned by a classifier.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    labels : array-like of shape (n_classes,), default=None
        A list of classes or column indices to select some (or to force
        inclusion of classes absent from the data).

    samplewise : bool, default=False
        In the multilabel case, this calculates a confusion matrix per sample.

    Returns
    -------
    multi_confusion : ndarray of shape (n_outputs, 2, 2)
        A 2x2 confusion matrix corresponding to each output in the input.
        When calculating class-wise multi_confusion (default), then
        n_outputs = n_labels; when calculating sample-wise multi_confusion
        (samplewise=True), n_outputs = n_samples. If ``labels`` is defined,
        the results will be returned in the order specified in ``labels``,
        otherwise the results will be returned in sorted order by default.

    See Also
    --------
    confusion_matrix : Compute confusion matrix to evaluate the accuracy of a
        classifier.

    Notes
    -----
    The `multilabel_confusion_matrix` calculates class-wise or sample-wise
    multilabel confusion matrices, and in multiclass tasks, labels are
    binarized under a one-vs-rest way; while
    :func:`~sklearn.metrics.confusion_matrix` calculates one confusion matrix
    for confusion between every two classes.

    Examples
    --------
    Multiclass case:

    >>> import mars.tensor as mt
    >>> from mars.learn.metrics import multilabel_confusion_matrix
    >>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    >>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    >>> multilabel_confusion_matrix(y_true, y_pred,
    ...                             labels=["ant", "bird", "cat"])
    array([[[3, 1],
            [0, 2]],
    <BLANKLINE>
           [[5, 0],
            [1, 0]],
    <BLANKLINE>
           [[2, 1],
            [1, 2]]])

    Multilabel-indicator case not implemented yet.
    """
    exec_kw = dict(session=session, **(run_kwargs or dict()))

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    execute(y_type, y_true, y_pred, **exec_kw)
    y_type = y_type.fetch()

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
    check_consistent_length(y_true, y_pred, sample_weight, **exec_kw)

    if y_type not in ("binary", "multiclass", "multilabel-indicator"):
        raise ValueError("%s is not supported" % y_type)

    present_labels = unique_labels(y_true, y_pred)
    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        labels = mt.tensor(labels)
        n_labels = labels.shape[0]
        # todo simplify this when mt.setdiff1d is implemented.
        labels = labels.rechunk(((np.nan,),)).map_chunk(
            lambda l, pl: np.hstack([l, np.setdiff1d(pl, l, assume_unique=True)]),
            args=(present_labels,),
            dtype=labels.dtype,
            shape=(np.nan,),
        )

    if y_true.ndim == 1:
        if samplewise:
            raise ValueError(
                "Samplewise metrics are not available outside of "
                "multilabel classification."
            )

        le = LabelEncoder()
        le.fit(labels, execute=False)
        y_true = le.transform(y_true, execute=False)
        y_pred = le.transform(y_pred, execute=False)
        sorted_labels = le.classes_

        # labels are now from 0 to len(labels) - 1 -> use bincount
        tp = y_true == y_pred
        tp_bins = y_true[tp]
        execute(labels, y_true, y_pred, tp_bins, **exec_kw)
        if sample_weight is not None:
            tp_bins_weights = mt.asarray(sample_weight)[tp]
        else:
            tp_bins_weights = None

        if tp_bins.shape[0]:
            tp_sum = mt.bincount(
                tp_bins, weights=tp_bins_weights, minlength=labels.shape[0]
            )
        else:
            # Pathological case
            true_sum = pred_sum = tp_sum = mt.zeros(labels.shape[0])
        if y_pred.shape[0]:
            pred_sum = mt.bincount(
                y_pred, weights=sample_weight, minlength=labels.shape[0]
            )
        if y_true.shape[0]:
            true_sum = mt.bincount(
                y_true, weights=sample_weight, minlength=labels.shape[0]
            )

        # Retain only selected labels
        indices = mt.searchsorted(sorted_labels, labels[:n_labels])
        tp_sum = tp_sum[indices]
        true_sum = true_sum[indices]
        pred_sum = pred_sum[indices]

    else:
        sum_axis = 1 if samplewise else 0

        def _check_labels(labels, present_labels):
            # All labels are index integers for multilabel.
            # Select labels:
            if not np.array_equal(labels, present_labels):
                if np.max(labels) > np.max(present_labels):
                    raise ValueError(
                        "All labels must be in [0, n labels) for "
                        "multilabel targets. "
                        "Got %d > %d" % (np.max(labels), np.max(present_labels))
                    )
                if np.min(labels) < 0:
                    raise ValueError(
                        "All labels must be in [0, n labels) for "
                        "multilabel targets. "
                        "Got %d < 0" % np.min(labels)
                    )
            return labels

        labels = labels.map_chunk(
            _check_labels,
            args=(present_labels,),
            dtype=labels.dtype,
            shape=labels.shape,
        )

        if n_labels is not None:
            y_true = y_true[:, labels[:n_labels]]
            y_pred = y_pred[:, labels[:n_labels]]

        # calculate weighted counts
        true_and_pred = mt.multiply(y_true, y_pred)
        tp_sum = count_nonzero(
            true_and_pred, axis=sum_axis, sample_weight=sample_weight
        )
        pred_sum = count_nonzero(y_pred, axis=sum_axis, sample_weight=sample_weight)
        true_sum = count_nonzero(y_true, axis=sum_axis, sample_weight=sample_weight)

    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum

    # we need to obtain correct shape of y_true for further computation
    executables = (fp, fn, tp, y_true)
    execute(*executables, **exec_kw)

    if sample_weight is not None and samplewise:
        sample_weight = mt.asarray(sample_weight)
        tp = mt.asarray(tp)
        fp = mt.asarray(fp)
        fn = mt.asarray(fn)
        tn = sample_weight * y_true.shape[1] - tp - fp - fn
    elif sample_weight is not None:
        tn = sum(sample_weight) - tp - fp - fn
    elif samplewise:
        tn = y_true.shape[1] - tp - fp - fn
    else:
        tn = y_true.shape[0] - tp - fp - fn

    ret = mt.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)
    return ret.execute(**exec_kw)


def _check_zero_division(zero_division):  # pragma: no cover
    if isinstance(zero_division, str) and zero_division == "warn":
        return
    elif isinstance(zero_division, (int, float)) and zero_division in [0, 1]:
        return
    raise ValueError(
        "Got zero_division={0}." ' Must be one of ["warn", 0, 1]'.format(zero_division)
    )


def _warn_prf(average, modifier, msg_start, result_size):  # pragma: no cover
    axis0, axis1 = "sample", "label"
    if average == "samples":
        axis0, axis1 = axis1, axis0
    msg = (
        "{0} ill-defined and being set to 0.0 {{0}} "
        "no {1} {2}s. Use `zero_division` parameter to control"
        " this behavior.".format(msg_start, modifier, axis0)
    )
    if result_size == 1:
        msg = msg.format("due to")
    else:
        msg = msg.format("in {0}s with".format(axis1))
    warnings.warn(msg, UndefinedMetricWarning, stacklevel=2)


def _prf_divide(
    numerator, denominator, metric, modifier, average, warn_for, zero_division="warn"
):  # pragma: no cover
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements equal to
    0 or 1 (according to ``zero_division``). Plus, if
    ``zero_division != "warn"`` raises a warning.

    The metric, modifier and average arguments are used only for determining
    an appropriate warning.
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    # if ``zero_division=1``, set those with denominator == 0 equal to 1
    result[mask] = 0.0 if zero_division in ["warn", 0] else 1.0

    # the user will be removing warnings if zero_division is set to something
    # different than its default value. If we are computing only f-score
    # the warning will be raised only if precision and recall are ill-defined
    if zero_division != "warn" or metric not in warn_for:
        return result

    # build appropriate warning
    # E.g. "Precision and F-score are ill-defined and being set to 0.0 in
    # labels with no predicted samples. Use ``zero_division`` parameter to
    # control this behavior."

    if metric in warn_for and "f-score" in warn_for:
        msg_start = "{0} and F-score are".format(metric.title())
    elif metric in warn_for:
        msg_start = "{0} is".format(metric.title())
    elif "f-score" in warn_for:
        msg_start = "F-score is"
    else:
        return result

    _warn_prf(average, modifier, msg_start, len(result))

    return result


def _check_set_wise_labels(
    y_true, y_pred, average, labels, pos_label, session=None, run_kwargs=None
):  # pragma: no cover
    """Validation associated with set-wise metrics

    Returns identified labels
    """
    exec_kwargs = dict(session=session, **(run_kwargs or dict()))
    average_options = (None, "micro", "macro", "weighted", "samples")
    if average not in average_options and average != "binary":
        raise ValueError("average has to be one of " + str(average_options))

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    present_labels = unique_labels(y_true, y_pred)
    execute(y_type, y_true, y_pred, **exec_kwargs)
    y_type = y_type.fetch(**exec_kwargs)

    if average == "binary":
        if y_type == "binary":
            t_pos_in_labels = mt.any(mt.isin(present_labels, pos_label))
            execute(t_pos_in_labels, present_labels, **exec_kwargs)
            pos_in_labels = t_pos_in_labels.fetch(**exec_kwargs)
            if pos_in_labels:
                if present_labels.shape[0] >= 2:
                    raise ValueError(
                        "pos_label=%r is not a valid label: "
                        "%r" % (pos_label, present_labels)
                    )
            labels = [pos_label]
        else:
            average_options = list(average_options)
            if y_type == "multiclass":
                average_options.remove("samples")
            raise ValueError(
                "Target is %s but average='binary'. Please "
                "choose another average setting, one of %r." % (y_type, average_options)
            )
    elif pos_label not in (None, 1):
        warnings.warn(
            "Note that pos_label (set to %r) is ignored when "
            "average != 'binary' (got %r). You may use "
            "labels=[pos_label] to specify a single positive class."
            % (pos_label, average),
            UserWarning,
        )
    return labels


def precision_recall_fscore_support(
    y_true,
    y_pred,
    *,
    beta=1.0,
    labels=None,
    pos_label=1,
    average=None,
    warn_for=("precision", "recall", "f-score"),
    sample_weight=None,
    zero_division="warn",
    session=None,
    run_kwargs=None
):
    """Compute precision, recall, F-measure and support for each class

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.

    The F-beta score weights recall more than precision by a factor of
    ``beta``. ``beta == 1.0`` means recall and precision are equally important.

    The support is the number of occurrences of each class in ``y_true``.

    If ``pos_label is None`` and in binary classification, this function
    returns the average precision, recall and F-measure if ``average``
    is one of ``'micro'``, ``'macro'``, ``'weighted'`` or ``'samples'``.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    beta : float, 1.0 by default
        The strength of recall versus precision in the F-score.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    pos_label : str or int, 1 by default
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : string, [None (default), 'binary', 'micro', 'macro', 'samples', \
                       'weighted']
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division:
           - recall: when there are no positive labels
           - precision: when there are no positive predictions
           - f-score: both

        If set to "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    recall : float (if average is not None) or array of float, , shape =\
        [n_unique_labels]

    fbeta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    support : None (if average is not None) or array of int, shape =\
        [n_unique_labels]
        The number of occurrences of each label in ``y_true``.

    References
    ----------
    .. [1] `Wikipedia entry for the Precision and recall
           <https://en.wikipedia.org/wiki/Precision_and_recall>`_

    .. [2] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_

    .. [3] `Discriminative Methods for Multi-labeled Classification Advances
           in Knowledge Discovery and Data Mining (2004), pp. 22-30 by Shantanu
           Godbole, Sunita Sarawagi
           <http://www.godbole.net/shantanu/pubs/multilabelsvm-pakdd04.pdf>`_

    Examples
    --------
    >>> import numpy as np
    >>> from mars.learn.metrics import precision_recall_fscore_support
    >>> y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
    >>> y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
    >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
    (0.22..., 0.33..., 0.26..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='micro')
    (0.33..., 0.33..., 0.33..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
    (0.22..., 0.33..., 0.26..., None)

    It is possible to compute per-label precisions, recalls, F1-scores and
    supports instead of averaging:

    >>> precision_recall_fscore_support(y_true, y_pred, average=None,
    ... labels=['pig', 'dog', 'cat'])
    (array([0.        , 0.        , 0.66...]),
     array([0., 0., 1.]), array([0. , 0. , 0.8]),
     array([2, 2, 2]))

    Notes
    -----
    When ``true positive + false positive == 0``, precision is undefined;
    When ``true positive + false negative == 0``, recall is undefined.
    In such cases, by default the metric will be set to 0, as will f-score,
    and ``UndefinedMetricWarning`` will be raised. This behavior can be
    modified with ``zero_division``.
    """
    exec_kw = dict(session=session, **(run_kwargs or dict()))

    _check_zero_division(zero_division)
    if beta < 0:
        raise ValueError("beta should be >=0 in the F-beta score")
    labels = _check_set_wise_labels(
        y_true,
        y_pred,
        average,
        labels,
        pos_label,
        session=session,
        run_kwargs=run_kwargs,
    )

    # Calculate tp_sum, pred_sum, true_sum ###
    samplewise = average == "samples"
    MCM = multilabel_confusion_matrix(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        labels=labels,
        samplewise=samplewise,
        session=session,
        run_kwargs=run_kwargs,
    )
    tp_sum = MCM[:, 1, 1]
    pred_sum = tp_sum + MCM[:, 0, 1]
    true_sum = tp_sum + MCM[:, 1, 0]

    if average == "micro":
        tp_sum = mt.array([tp_sum.sum()])
        pred_sum = mt.array([pred_sum.sum()])
        true_sum = mt.array([true_sum.sum()])

    execute(true_sum, **exec_kw)

    # Finally, we have all our sufficient statistics. Divide! #
    beta2 = beta**2

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision = _prf_divide(
        tp_sum, pred_sum, "precision", "predicted", average, warn_for, zero_division
    )
    recall = _prf_divide(
        tp_sum, true_sum, "recall", "true", average, warn_for, zero_division
    )

    # warn for f-score only if zero_division is warn, it is in warn_for
    # and BOTH prec and rec are ill-defined
    if zero_division == "warn" and ("f-score",) == warn_for:
        any_pred_sum_zero = (
            (pred_sum[true_sum == 0] == 0).any().execute(**exec_kw).fetch(**exec_kw)
        )
        if any_pred_sum_zero:
            _warn_prf(average, "true nor predicted", "F-score is", len(true_sum))

    # if tp == 0 F will be 1 only if all predictions are zero, all labels are
    # zero, and zero_division=1. In all other case, 0
    if np.isposinf(beta):
        f_score = recall
    else:
        denom = beta2 * precision + recall

        denom[denom == 0.0] = 1  # avoid division by 0
        f_score = (1 + beta2) * precision * recall / denom

    # Average the results
    if average == "weighted":
        weights = true_sum
        sum_weights, sum_pred_sum = fetch(
            execute(weights.sum(), pred_sum.sum(), **exec_kw), **exec_kw
        )
        if sum_weights == 0:
            zero_division_value = 0.0 if zero_division in ["warn", 0] else 1.0
            # precision is zero_division if there are no positive predictions
            # recall is zero_division if there are no positive labels
            # fscore is zero_division if all labels AND predictions are
            # negative
            return (
                mt.scalar(zero_division_value if sum_pred_sum == 0 else 0),
                mt.scalar(zero_division_value),
                mt.scalar(zero_division_value if sum_pred_sum == 0 else 0),
                None,
            )

    elif average == "samples":
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        assert average != "binary" or len(precision) == 1
        precision = mt.average(precision, weights=weights)
        recall = mt.average(recall, weights=weights)
        f_score = mt.average(f_score, weights=weights)
        true_sum = None  # return no support

    return precision, recall, f_score, true_sum


def precision_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn"
):
    """Compute the precision

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The best value is 1 and the worst value is 0.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    pos_label : str or int, 1 by default
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted']
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Precision of the positive class in binary classification or weighted
        average of the precision of each class for the multiclass task.

    See also
    --------
    precision_recall_fscore_support, multilabel_confusion_matrix

    Examples
    --------
    >>> from mars.learn.metrics import precision_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> precision_score(y_true, y_pred, average='macro')
    0.22...
    >>> precision_score(y_true, y_pred, average='micro')
    0.33...
    >>> precision_score(y_true, y_pred, average='weighted')
    0.22...
    >>> precision_score(y_true, y_pred, average=None)
    array([0.66..., 0.        , 0.        ])
    >>> y_pred = [0, 0, 0, 0, 0, 0]
    >>> precision_score(y_true, y_pred, average=None)
    array([0.33..., 0.        , 0.        ])
    >>> precision_score(y_true, y_pred, average=None, zero_division=1)
    array([0.33..., 1.        , 1.        ])

    Notes
    -----
    When ``true positive + false positive == 0``, precision returns 0 and
    raises ``UndefinedMetricWarning``. This behavior can be
    modified with ``zero_division``.

    """
    p, _, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=("precision",),
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    return p


def recall_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn"
):
    """Compute the recall

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    pos_label : str or int, 1 by default
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted']
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    recall : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Recall of the positive class in binary classification or weighted
        average of the recall of each class for the multiclass task.

    See also
    --------
    precision_recall_fscore_support, balanced_accuracy_score,
    multilabel_confusion_matrix

    Examples
    --------
    >>> from mars.learn.metrics import recall_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> recall_score(y_true, y_pred, average='macro')
    0.33...
    >>> recall_score(y_true, y_pred, average='micro')
    0.33...
    >>> recall_score(y_true, y_pred, average='weighted')
    0.33...
    >>> recall_score(y_true, y_pred, average=None)
    array([1., 0., 0.])
    >>> y_true = [0, 0, 0, 0, 0, 0]
    >>> recall_score(y_true, y_pred, average=None)
    array([0.5, 0. , 0. ])
    >>> recall_score(y_true, y_pred, average=None, zero_division=1)
    array([0.5, 1. , 1. ])

    Notes
    -----
    When ``true positive + false negative == 0``, recall returns 0 and raises
    ``UndefinedMetricWarning``. This behavior can be modified with
    ``zero_division``.
    """
    _, r, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=("recall",),
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    return r


def f1_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn"
):
    """Compute the F1 score, also known as balanced F-score or F-measure

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    In the multi-class and multi-label case, this is the average of
    the F1 score of each class with weighting depending on the ``average``
    parameter.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    pos_label : str or int, 1 by default
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted']
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division, i.e. when all
        predictions and labels are negative. If set to "warn", this acts as 0,
        but warnings are also raised.

    Returns
    -------
    f1_score : float or array of float, shape = [n_unique_labels]
        F1 score of the positive class in binary classification or weighted
        average of the F1 scores of each class for the multiclass task.

    See also
    --------
    fbeta_score, precision_recall_fscore_support, jaccard_score,
    multilabel_confusion_matrix

    References
    ----------
    .. [1] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_

    Examples
    --------
    >>> from mars.learn.metrics import f1_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> f1_score(y_true, y_pred, average='macro')
    0.26...
    >>> f1_score(y_true, y_pred, average='micro')
    0.33...
    >>> f1_score(y_true, y_pred, average='weighted')
    0.26...
    >>> f1_score(y_true, y_pred, average=None)
    array([0.8, 0. , 0. ])
    >>> y_true = [0, 0, 0, 0, 0, 0]
    >>> y_pred = [0, 0, 0, 0, 0, 0]
    >>> f1_score(y_true, y_pred, zero_division=1)
    1.0...

    Notes
    -----
    When ``true positive + false positive == 0``, precision is undefined;
    When ``true positive + false negative == 0``, recall is undefined.
    In such cases, by default the metric will be set to 0, as will f-score,
    and ``UndefinedMetricWarning`` will be raised. This behavior can be
    modified with ``zero_division``.
    """
    return fbeta_score(
        y_true,
        y_pred,
        beta=1,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def fbeta_score(
    y_true,
    y_pred,
    *,
    beta,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn"
):
    """Compute the F-beta score

    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0.

    The `beta` parameter determines the weight of recall in the combined
    score. ``beta < 1`` lends more weight to precision, while ``beta > 1``
    favors recall (``beta -> 0`` considers only precision, ``beta -> +inf``
    only recall).

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    beta : float
        Determines the weight of recall in the combined score.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    pos_label : str or int, 1 by default
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted']
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division, i.e. when all
        predictions and labels are negative. If set to "warn", this acts as 0,
        but warnings are also raised.

    Returns
    -------
    fbeta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        F-beta score of the positive class in binary classification or weighted
        average of the F-beta score of each class for the multiclass task.

    See also
    --------
    precision_recall_fscore_support, multilabel_confusion_matrix

    References
    ----------
    .. [1] R. Baeza-Yates and B. Ribeiro-Neto (2011).
           Modern Information Retrieval. Addison Wesley, pp. 327-328.

    .. [2] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_

    Examples
    --------
    >>> from mars.learn.metrics import fbeta_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> fbeta_score(y_true, y_pred, average='macro', beta=0.5)
    0.23...
    >>> fbeta_score(y_true, y_pred, average='micro', beta=0.5)
    0.33...
    >>> fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
    0.23...
    >>> fbeta_score(y_true, y_pred, average=None, beta=0.5)
    array([0.71..., 0.        , 0.        ])

    Notes
    -----
    When ``true positive + false positive == 0`` or
    ``true positive + false negative == 0``, f-score returns 0 and raises
    ``UndefinedMetricWarning``. This behavior can be
    modified with ``zero_division``.
    """

    _, _, f, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        beta=beta,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=("f-score",),
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    return f
