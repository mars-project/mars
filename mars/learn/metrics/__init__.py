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

from .pairwise import euclidean_distances, pairwise_distances, pairwise_distances_topk
from ._classification import (
    accuracy_score,
    log_loss,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)
from ._ranking import roc_curve, auc, roc_auc_score
from ._regresssion import r2_score
from ._scorer import get_scorer
