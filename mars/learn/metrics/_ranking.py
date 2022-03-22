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
from functools import partial

import numpy as np

from ... import execute as _execute, fetch as _fetch
from ... import tensor as mt
from ...utils import cache_tileables
from ..preprocessing import label_binarize
from ..utils._encode import _encode, _unique
from ..utils.checks import assert_all_finite
from ..utils.core import sort_by
from ..utils.multiclass import type_of_target
from ..utils.validation import check_array, check_consistent_length, column_or_1d
from ._base import _average_binary_score, _average_multiclass_ovo_score


def auc(x, y, session=None, run_kwargs=None):
    """Compute Area Under the Curve (AUC) using the trapezoidal rule

    This is a general function, given points on a curve.  For computing the
    area under the ROC-curve, see :func:`roc_auc_score`.  For an alternative
    way to summarize a precision-recall curve, see
    :func:`average_precision_score`.

    Parameters
    ----------
    x : tensor, shape = [n]
        x coordinates. These must be either monotonic increasing or monotonic
        decreasing.
    y : tensor, shape = [n]
        y coordinates.

    Returns
    -------
    auc : tensor, with float value

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.learn import metrics
    >>> y = mt.array([1, 1, 2, 2])
    >>> pred = mt.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    >>> metrics.auc(fpr, tpr)
    0.75

    See also
    --------
    roc_auc_score : Compute the area under the ROC curve
    average_precision_score : Compute average precision from prediction scores
    precision_recall_curve :
        Compute precision-recall pairs for different probability thresholds
    """
    check_consistent_length(x, y)
    x = column_or_1d(x)
    y = column_or_1d(y)

    if x.shape[0] < 2:
        raise ValueError(
            "At least 2 points are needed to compute"
            f" area under curve, but x.shape = {x.shape}"
        )

    direction = 1
    dx = mt.diff(x)
    any_dx_lt_0 = mt.any(dx < 0)
    all_dx_le_0 = mt.all(dx <= 0)
    mt.ExecutableTuple([x, any_dx_lt_0, all_dx_le_0]).execute(
        session=session, **(run_kwargs or dict())
    )
    if any_dx_lt_0.fetch(session=session):
        if all_dx_le_0.fetch(session=session):
            direction = -1
        else:
            x_data = x.fetch(session=session)
            raise ValueError(f"x is neither increasing nor decreasing : {x_data}.")

    area = direction * mt.trapz(y, x)
    return area.execute(session=session, **(run_kwargs or dict()))


def _binary_clf_curve(
    y_true, y_score, pos_label=None, sample_weight=None, session=None, run_kwargs=None
):
    """Calculate true and false positives per binary classification threshold.

    Parameters
    ----------
    y_true : tensor, shape = [n_samples]
        True targets of binary classification

    y_score : tensor, shape = [n_samples]
        Estimated probabilities or decision function

    pos_label : int or str, default=None
        The label of the positive class

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    fps : tensor, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).

    tps : tensor, shape = [n_thresholds <= len(mt.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).

    thresholds : tensor, shape = [n_thresholds]
        Decreasing score values.
    """
    y_type = type_of_target(y_true).to_numpy(session=session, **(run_kwargs or dict()))
    if not (y_type == "binary" or (y_type == "multiclass" and pos_label is not None)):
        raise ValueError(f"{y_type} format is not supported")

    check_consistent_length(
        y_true, y_score, sample_weight, session=session, **(run_kwargs or dict())
    )
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    y_true = assert_all_finite(y_true, check_only=False)
    y_score = assert_all_finite(y_score, check_only=False)

    cache_tileables(y_true, y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a FutureWarning by calling np.array_equal(a, b)
    # when elements in the two arrays are not comparable.
    classes = mt.unique(y_true, aggregate_size=1).to_numpy(
        session=session, **(run_kwargs or dict())
    )
    if pos_label is None and (
        classes.dtype.kind in ("O", "U", "S")
        or not (
            np.array_equal(classes, [0, 1])
            or np.array_equal(classes, [-1, 1])
            or np.array_equal(classes, [0])
            or np.array_equal(classes, [-1])
            or np.array_equal(classes, [1])
        )
    ):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError(
            f"y_true takes value in {{{classes_repr}}} and "
            "pos_label is not specified: either make y_true "
            "take value in {{0, 1}} or {{-1, 1}} or "
            "pass pos_label explicitly."
        )
    elif pos_label is None:
        pos_label = 1.0

    # make y_true a boolean vector
    y_true = y_true == pos_label

    # sort scores and corresponding truth values
    # original implementation adopted from sklearn:
    # """
    # desc_score_indices = mt.argsort(y_score, kind="mergesort")[::-1]
    # y_score = y_score[desc_score_indices]
    # y_true = y_true[desc_score_indices]
    # if sample_weight is not None:
    #     weight = sample_weight[desc_score_indices]
    # else:
    #     weight = 1.0
    # """
    # since fancy indexing is a heavy operation, we try to use DataFrame to sort
    to_sort = [y_score, y_true]
    if sample_weight is not None:
        to_sort.append(sample_weight)
    to_sort = sort_by(to_sort, y_score, ascending=False)
    y_score, y_true = to_sort[:2]
    if sample_weight is not None:
        weight = to_sort[-1]
    else:
        weight = 1.0

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = mt.where(mt.diff(y_score))[0]
    threshold_idxs = mt.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    # raw tps from sklearn implementation
    # we try to perform only one fancy index
    # tps = (y_true * weight).cumsum()[threshold_idxs]
    temp_tps = (y_true * weight).cumsum()
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        # fps = ((1 - y_true) * weight).cumsum()[threshold_idxs]
        temp_fps = ((1 - y_true) * weight).cumsum()
        tps, fps, thresholds = mt.stack([temp_tps, temp_fps, y_score])[
            :, threshold_idxs
        ]

    else:
        tps, thresholds = mt.stack([temp_tps, y_score])[:, threshold_idxs]
        fps = 1 + threshold_idxs - tps
    return _execute([fps, tps, thresholds], session=session, **(run_kwargs or dict()))


def _binary_roc_auc_score(
    y_true, y_score, sample_weight=None, max_fpr=None, session=None, run_kwargs=None
):
    """Binary roc auc score."""

    from numpy import interp

    if len(mt.unique(y_true).execute()) != 2:
        raise ValueError(
            "Only one class present in y_true. ROC AUC score "
            "is not defined in that case."
        )

    fpr, tpr, _ = roc_curve(
        y_true,
        y_score,
        sample_weight=sample_weight,
        session=session,
        run_kwargs=run_kwargs,
    )
    fpr, tpr = mt.ExecutableTuple([fpr, tpr]).fetch(session=session)

    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr, session=session, run_kwargs=run_kwargs).fetch(
            session=session
        )
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError(f"Expected max_fpr in range (0, 1], got: {max_fpr}")

    # Add a single point at max_fpr by linear interpolation
    stop = (
        mt.searchsorted(fpr, max_fpr, "right")
        .execute(session=session, **(run_kwargs or dict()))
        .fetch(session=session)
    )
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = list(tpr[:stop])
    tpr.append(interp(max_fpr, x_interp, y_interp))
    fpr = list(fpr[:stop])
    fpr.append(max_fpr)
    partial_auc = auc(fpr, tpr, session=session, run_kwargs=run_kwargs)

    # McClish correction: standardize result to be 0.5 if non-discriminant
    # and 1 if maximal
    min_area = 0.5 * max_fpr**2
    max_area = max_fpr
    return 0.5 * (
        1 + (partial_auc.fetch(session=session) - min_area) / (max_area - min_area)
    )


def roc_auc_score(
    y_true,
    y_score,
    *,
    average="macro",
    sample_weight=None,
    max_fpr=None,
    multi_class="raise",
    labels=None,
    session=None,
    run_kwargs=None,
):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    from prediction scores.

    Note: this implementation can be used with binary, multiclass and
    multilabel classification, but some restrictions apply (see Parameters).

    Read more in the :ref:`User Guide <roc_metrics>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        True labels or binary label indicators. The binary and multiclass cases
        expect labels with shape (n_samples,) while the multilabel case expects
        binary label indicators with shape (n_samples, n_classes).

    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores.

        * In the binary case, it corresponds to an array of shape
          `(n_samples,)`. Both probability estimates and non-thresholded
          decision values can be provided. The probability estimates correspond
          to the **probability of the class with the greater label**,
          i.e. `estimator.classes_[1]` and thus
          `estimator.predict_proba(X, y)[:, 1]`. The decision values
          corresponds to the output of `estimator.decision_function(X, y)`.
          See more information in the :ref:`User guide <roc_auc_binary>`;
        * In the multiclass case, it corresponds to an array of shape
          `(n_samples, n_classes)` of probability estimates provided by the
          `predict_proba` method. The probability estimates **must**
          sum to 1 across the possible classes. In addition, the order of the
          class scores must correspond to the order of ``labels``,
          if provided, or else to the numerical or lexicographical order of
          the labels in ``y_true``. See more information in the
          :ref:`User guide <roc_auc_multiclass>`;
        * In the multilabel case, it corresponds to an array of shape
          `(n_samples, n_classes)`. Probability estimates are provided by the
          `predict_proba` method and the non-thresholded decision values by
          the `decision_function` method. The probability estimates correspond
          to the **probability of the class with the greater label for each
          output** of the classifier. See more information in the
          :ref:`User guide <roc_auc_multilabel>`.

    average : {'micro', 'macro', 'samples', 'weighted'} or None, \
            default='macro'
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:
        Note: multiclass ROC AUC currently only handles the 'macro' and
        'weighted' averages.

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

        Will be ignored when ``y_true`` is binary.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    max_fpr : float > 0 and <= 1, default=None
        If not ``None``, the standardized partial AUC [2]_ over the range
        [0, max_fpr] is returned. For the multiclass case, ``max_fpr``,
        should be either equal to ``None`` or ``1.0`` as AUC ROC partial
        computation currently is not supported for multiclass.

    multi_class : {'raise', 'ovr', 'ovo'}, default='raise'
        Only used for multiclass targets. Determines the type of configuration
        to use. The default value raises an error, so either
        ``'ovr'`` or ``'ovo'`` must be passed explicitly.

        ``'ovr'``:
            Stands for One-vs-rest. Computes the AUC of each class
            against the rest [3]_ [4]_. This
            treats the multiclass case in the same way as the multilabel case.
            Sensitive to class imbalance even when ``average == 'macro'``,
            because class imbalance affects the composition of each of the
            'rest' groupings.
        ``'ovo'``:
            Stands for One-vs-one. Computes the average AUC of all
            possible pairwise combinations of classes [5]_.
            Insensitive to class imbalance when
            ``average == 'macro'``.

    labels : array-like of shape (n_classes,), default=None
        Only used for multiclass targets. List of labels that index the
        classes in ``y_score``. If ``None``, the numerical or lexicographical
        order of the labels in ``y_true`` is used.

    Returns
    -------
    auc : float

    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

    .. [2] `Analyzing a portion of the ROC curve. McClish, 1989
            <https://www.ncbi.nlm.nih.gov/pubmed/2668680>`_

    .. [3] Provost, F., Domingos, P. (2000). Well-trained PETs: Improving
           probability estimation trees (Section 6.2), CeDER Working Paper
           #IS-00-04, Stern School of Business, New York University.

    .. [4] `Fawcett, T. (2006). An introduction to ROC analysis. Pattern
            Recognition Letters, 27(8), 861-874.
            <https://www.sciencedirect.com/science/article/pii/S016786550500303X>`_

    .. [5] `Hand, D.J., Till, R.J. (2001). A Simple Generalisation of the Area
            Under the ROC Curve for Multiple Class Classification Problems.
            Machine Learning, 45(2), 171-186.
            <http://link.springer.com/article/10.1023/A:1010920819831>`_

    See Also
    --------
    average_precision_score : Area under the precision-recall curve.
    roc_curve : Compute Receiver operating characteristic (ROC) curve.
    RocCurveDisplay.from_estimator : Plot Receiver Operating Characteristic
        (ROC) curve given an estimator and some data.
    RocCurveDisplay.from_predictions : Plot Receiver Operating Characteristic
        (ROC) curve given the true and predicted values.

    Examples
    --------
    Binary case:

    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from mars.learn.metrics import roc_auc_score
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    >>> roc_auc_score(y, clf.predict_proba(X)[:, 1])
    0.99...
    >>> roc_auc_score(y, clf.decision_function(X))
    0.99...

    Multiclass case:

    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegression(solver="liblinear").fit(X, y)
    >>> roc_auc_score(y, clf.predict_proba(X), multi_class='ovr')
    0.99...

    Multilabel case:

    >>> import numpy as np
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from sklearn.multioutput import MultiOutputClassifier
    >>> X, y = make_multilabel_classification(random_state=0)
    >>> clf = MultiOutputClassifier(clf).fit(X, y)
    >>> # get a list of n_output containing probability arrays of shape
    >>> # (n_samples, n_classes)
    >>> y_pred = clf.predict_proba(X)
    >>> # extract the positive columns for each output
    >>> y_pred = np.transpose([pred[:, 1] for pred in y_pred])
    >>> roc_auc_score(y, y_pred, average=None)
    array([0.82..., 0.86..., 0.94..., 0.85... , 0.94...])
    >>> from sklearn.linear_model import RidgeClassifierCV
    >>> clf = RidgeClassifierCV().fit(X, y)
    >>> roc_auc_score(y, clf.decision_function(X), average=None)
    array([0.81..., 0.84... , 0.93..., 0.87..., 0.94...])
    """

    cache_tileables(y_true, y_score)

    y_type = type_of_target(y_true)
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_score = check_array(y_score, ensure_2d=False)
    _execute([y_type, y_true, y_score], session=session, **(run_kwargs or dict()))
    y_type = y_type.fetch(session=session)

    def execute(*args):
        result = [None] * len(args)
        to_execute = dict()
        for i, arg in enumerate(args):
            if hasattr(arg, "op"):
                to_execute[i] = arg
            else:
                result[i] = arg
        if to_execute:
            _execute(*to_execute.values(), session=session, **(run_kwargs or dict()))
        for i, e in to_execute.items():
            if e.isscalar():
                e = e.fetch(session=session)
            result[i] = e
        return result[0] if len(result) == 1 else result

    if y_type == "multiclass" or (
        y_type == "binary" and y_score.ndim == 2 and y_score.shape[1] > 2
    ):
        # do not support partial ROC computation for multiclass
        if max_fpr is not None and max_fpr != 1.0:
            raise ValueError(
                "Partial AUC computation not available in "
                "multiclass setting, 'max_fpr' must be"
                " set to `None`, received `max_fpr={0}` "
                "instead".format(max_fpr)
            )
        if multi_class == "raise":
            raise ValueError("multi_class must be in ('ovo', 'ovr')")
        return execute(
            _multiclass_roc_auc_score(
                y_true, y_score, labels, multi_class, average, sample_weight
            )
        )
    elif y_type == "binary":
        labels = mt.unique(y_true).execute(session=session, **(run_kwargs or dict()))
        y_true = label_binarize(y_true, classes=labels, execute=False)[:, 0]
        cache_tileables(y_true)
        return execute(
            _average_binary_score(
                partial(_binary_roc_auc_score, max_fpr=max_fpr),
                y_true,
                y_score,
                average,
                sample_weight=sample_weight,
            )
        )
    else:  # multilabel-indicator
        return execute(
            _average_binary_score(
                partial(_binary_roc_auc_score, max_fpr=max_fpr),
                y_true,
                y_score,
                average,
                sample_weight=sample_weight,
            )
        )


def _multiclass_roc_auc_score(
    y_true,
    y_score,
    labels,
    multi_class,
    average,
    sample_weight,
    session=None,
    run_kwargs=None,
):
    # validation of the input y_score
    if not mt.allclose(1, y_score.sum(axis=1)).to_numpy(
        session=session, **(run_kwargs or dict())
    ):  # pragma: no cover
        raise ValueError(
            "Target scores need to be probabilities for multiclass "
            "roc_auc, i.e. they should sum up to 1.0 over classes"
        )

    # validation for multiclass parameter specifications
    average_options = ("macro", "weighted")
    if average not in average_options:
        raise ValueError(
            "average must be one of {0} for multiclass problems".format(average_options)
        )

    multiclass_options = ("ovo", "ovr")
    if multi_class not in multiclass_options:
        raise ValueError(
            "multi_class='{0}' is not supported "
            "for multiclass ROC AUC, multi_class must be "
            "in {1}".format(multi_class, multiclass_options)
        )

    if labels is not None:
        labels = column_or_1d(labels).to_numpy(
            session=session, **(run_kwargs or dict())
        )
        classes = _unique(labels).to_numpy(session=session, **(run_kwargs or dict()))
        if len(classes) != len(labels):
            raise ValueError("Parameter 'labels' must be unique")
        if not np.array_equal(classes, labels):
            raise ValueError("Parameter 'labels' must be ordered")
        if len(classes) != y_score.shape[1]:
            raise ValueError(
                "Number of given labels, {0}, not equal to the number "
                "of columns in 'y_score', {1}".format(len(classes), y_score.shape[1])
            )
        if len(
            mt.setdiff1d(y_true, classes).execute(
                session=session, **(run_kwargs or dict())
            )
        ):
            raise ValueError("'y_true' contains labels not in parameter 'labels'")
    else:
        classes = _unique(y_true).execute(session=session, **(run_kwargs or dict()))
        if len(classes) != y_score.shape[1]:
            raise ValueError(
                "Number of classes in y_true not equal to the number of "
                "columns in 'y_score'"
            )

    if multi_class == "ovo":
        if sample_weight is not None:
            raise ValueError(
                "sample_weight is not supported "
                "for multiclass one-vs-one ROC AUC, "
                "'sample_weight' must be None in this case."
            )
        y_true_encoded = _encode(y_true, uniques=classes)
        # Hand & Till (2001) implementation (ovo)
        return _average_multiclass_ovo_score(
            _binary_roc_auc_score,
            y_true_encoded,
            y_score,
            average=average,
            session=session,
            run_kwargs=run_kwargs,
        )
    else:
        # ovr is same as multi-label
        y_true_multilabel = label_binarize(y_true, classes=classes, execute=False)
        return _average_binary_score(
            _binary_roc_auc_score,
            y_true_multilabel,
            y_score,
            average,
            sample_weight=sample_weight,
            session=session,
            run_kwargs=run_kwargs,
        )


def roc_curve(
    y_true,
    y_score,
    pos_label=None,
    sample_weight=None,
    drop_intermediate=True,
    session=None,
    run_kwargs=None,
):
    """Compute Receiver operating characteristic (ROC)

    Note: this implementation is restricted to the binary classification task.

    Read more in the :ref:`User Guide <roc_metrics>`.

    Parameters
    ----------

    y_true : tensor, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : tensor, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    drop_intermediate : boolean, optional (default=True)
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.

        .. versionadded:: 0.17
           parameter *drop_intermediate*.

    Returns
    -------
    fpr : tensor, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].

    tpr : tensor, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].

    thresholds : tensor, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `max(y_score) + 1`.

    See also
    --------
    roc_auc_score : Compute the area under the ROC curve

    Notes
    -----
    Since the thresholds are sorted from low to high values, they
    are reversed upon returning them to ensure they correspond to both ``fpr``
    and ``tpr``, which are sorted in reversed order during their calculation.

    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

    .. [2] Fawcett T. An introduction to ROC analysis[J]. Pattern Recognition
           Letters, 2006, 27(8):861-874.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.learn import metrics
    >>> y = mt.array([1, 1, 2, 2])
    >>> scores = mt.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    >>> fpr
    array([0. , 0. , 0.5, 0.5, 1. ])
    >>> tpr
    array([0. , 0.5, 0.5, 1. , 1. ])
    >>> thresholds
    array([1.8 , 0.8 , 0.4 , 0.35, 0.1 ])

    """
    from sklearn.exceptions import UndefinedMetricWarning

    cache_tileables(y_true, y_score)

    fps, tps, thresholds = _binary_clf_curve(
        y_true,
        y_score,
        pos_label=pos_label,
        sample_weight=sample_weight,
        session=session,
        run_kwargs=run_kwargs,
    )

    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here mt.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = mt.where(
            mt.r_[True, mt.logical_or(mt.diff(fps, 2), mt.diff(tps, 2)), True]
        )[0]
        # original implementation of sklearn:
        # """
        # fps = fps[optimal_idxs]
        # tps = tps[optimal_idxs]
        # thresholds = thresholds[optimal_idxs]
        # """
        # however, it's really a heavy operation to perform fancy index,
        # thus we put them together
        stacked = mt.stack([fps, tps, thresholds])
        fps, tps, thresholds = stacked[:, optimal_idxs]

    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    tps = mt.r_[0, tps]
    fps = mt.r_[0, fps]
    thresholds = mt.r_[thresholds[0] + 1, thresholds]

    last_fps = fps[-1]
    last_tps = tps[-1]
    _execute(
        [tps, fps, last_fps, last_tps, thresholds],
        session=session,
        **(run_kwargs or dict()),
    )
    last_fps, last_tps = _fetch([last_fps, last_tps], session=session)

    if last_fps <= 0:
        warnings.warn(
            "No negative samples in y_true, "
            "false positive value should be meaningless",
            UndefinedMetricWarning,
        )
        fpr = mt.repeat(mt.nan, fps.shape)
    else:
        fpr = fps / last_fps

    if last_tps <= 0:
        warnings.warn(
            "No positive samples in y_true, "
            "true positive value should be meaningless",
            UndefinedMetricWarning,
        )
        tpr = mt.repeat(mt.nan, tps.shape)
    else:
        tpr = tps / last_tps

    ret = mt.ExecutableTuple([fpr, tpr, thresholds]).execute(
        session=session, **(run_kwargs or dict())
    )
    return ret
