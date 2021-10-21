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

from sklearn.metrics import make_scorer

from . import accuracy_score, log_loss, r2_score


accuracy_score = make_scorer(accuracy_score)
r2_score = make_scorer(r2_score)
neg_log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)


SCORERS = dict(
    r2=r2_score,
    accuracy=accuracy_score,
    neg_log_loss=neg_log_loss_scorer,
)


def get_scorer(score_func: Union[str, Callable], **kwargs) -> Callable:
    """
    Get a scorer from string

    Parameters
    ----------
    score_func : str | callable
        scoring method as string. If callable it is returned as is.

    Returns
    -------
    scorer : callable
        The scorer.
    """
    if isinstance(score_func, str):
        try:
            scorer = SCORERS[score_func]
        except KeyError:
            raise ValueError(
                "{} is not a valid scoring value. "
                "Valid options are {}".format(score_func, sorted(SCORERS))
            )
        return scorer
    else:
        return make_scorer(score_func, **kwargs)
