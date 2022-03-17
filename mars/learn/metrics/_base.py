# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

from itertools import combinations

from ... import tensor as mt
from ..utils import check_array, check_consistent_length
from ..utils.multiclass import type_of_target


def _average_binary_score(
    binary_metric,
    y_true,
    y_score,
    average,
    sample_weight=None,
    session=None,
    run_kwargs=None,
):
    average_options = (None, "micro", "macro", "weighted", "samples")
    if average not in average_options:  # pragma: no cover
        raise ValueError("average has to be one of {0}".format(average_options))

    y_type = type_of_target(y_true).to_numpy(session=session, **(run_kwargs or dict()))
    if y_type not in ("binary", "multilabel-indicator"):  # pragma: no cover
        raise ValueError("{0} format is not supported".format(y_type))

    if y_type == "binary":
        return binary_metric(y_true, y_score, sample_weight=sample_weight)

    check_consistent_length(
        y_true, y_score, sample_weight, session=session, run_kwargs=run_kwargs
    )
    y_true = check_array(y_true)
    y_score = check_array(y_score)

    not_average_axis = 1
    score_weight = sample_weight
    average_weight = None

    if average == "micro":
        if score_weight is not None:  # pragma: no cover
            score_weight = mt.repeat(score_weight, y_true.shape[1])
        y_true = y_true.ravel()
        y_score = y_score.ravel()

    elif average == "weighted":
        if score_weight is not None:  # pragma: no cover
            average_weight = mt.sum(
                mt.multiply(y_true, mt.reshape(score_weight, (-1, 1))), axis=0
            )
        else:
            average_weight = mt.sum(y_true, axis=0)
        if mt.isclose(average_weight.sum(), 0.0).to_numpy(
            session=session, **(run_kwargs or dict())
        ):
            return 0

    elif average == "samples":
        # swap average_weight <-> score_weight
        average_weight = score_weight
        score_weight = None
        not_average_axis = 0

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))

    n_classes = y_score.shape[not_average_axis]
    score = mt.zeros((n_classes,))
    for c in range(n_classes):
        y_true_c = y_true.take([c], axis=not_average_axis).ravel()
        y_score_c = y_score.take([c], axis=not_average_axis).ravel()
        score[c] = binary_metric(y_true_c, y_score_c, sample_weight=score_weight)

    # Average the results
    if average is not None:
        if average_weight is not None:
            # Scores with 0 weights are forced to be 0, preventing the average
            # score from being affected by 0-weighted NaN elements.
            average_weight = mt.asarray(average_weight)
            score[average_weight == 0] = 0
        return mt.average(score, weights=average_weight)
    else:
        return score


def _average_multiclass_ovo_score(
    binary_metric, y_true, y_score, average="macro", session=None, run_kwargs=None
):
    check_consistent_length(y_true, y_score, session=session, run_kwargs=run_kwargs)

    y_true_unique = mt.unique(y_true).to_numpy()
    n_classes = y_true_unique.shape[0]
    n_pairs = n_classes * (n_classes - 1) // 2
    pair_scores = mt.empty(n_pairs)

    is_weighted = average == "weighted"
    prevalence = mt.empty(n_pairs) if is_weighted else None

    # Compute scores treating a as positive class and b as negative class,
    # then b as positive class and a as negative class
    for ix, (a, b) in enumerate(combinations(y_true_unique, 2)):
        a_mask = y_true == a
        b_mask = y_true == b
        ab_mask = mt.logical_or(a_mask, b_mask)

        if is_weighted:
            prevalence[ix] = mt.average(ab_mask)

        a_true = a_mask[ab_mask]
        b_true = b_mask[ab_mask]

        a_true_score = binary_metric(a_true, y_score[ab_mask, a])
        b_true_score = binary_metric(b_true, y_score[ab_mask, b])
        pair_scores[ix] = (a_true_score + b_true_score) / 2

    return mt.average(pair_scores, weights=prevalence)
