# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

try:
    from sklearn.neighbors.ball_tree import BallTree as SkBallTree
except ImportError:  # pragma: no cover
    SkBallTree = None

from ... import opcodes as OperandDef
from ...utils import require_not_none
from .tree import TreeBase, TreeQueryBase


@require_not_none(SkBallTree)
class _BallTree(TreeBase):
    _op_type_ = OperandDef.BALL_TREE_TRAIN
    _tree_type = SkBallTree


@require_not_none(SkBallTree)
class BallTreeQuery(TreeQueryBase):
    _op_type_ = OperandDef.BALL_TREE_QUERY
    _tree_type = SkBallTree


@require_not_none(SkBallTree)
def ball_tree_query(tree, data, n_neighbors, return_distance):
    op = BallTreeQuery(tree=tree, n_neighbors=n_neighbors,
                       return_distance=return_distance)
    ret = op(data)
    if not return_distance:
        return ret[0]
    return ret


@require_not_none(SkBallTree)
def BallTree(X, leaf_size, metric=None, **metric_params):
    metric_func = None
    if callable(metric):
        metric_func, metric = metric, None
    op = _BallTree(leaf_size=leaf_size, metric=metric, metric_func=metric_func,
                   **metric_params)
    return op(X)
