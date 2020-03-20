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
    from sklearn.neighbors import KDTree as SklearnKDTree
except ImportError:  # pragma: no cover
    SklearnKDTree = None

from ... import opcodes as OperandDef
from ...utils import require_not_none
from .tree import TreeObject, TreeBase, TreeQueryBase


class KDTree(TreeObject):
    pass


@require_not_none(SklearnKDTree)
class _KDTree(TreeBase):
    _op_type_ = OperandDef.KD_TREE_TRAIN
    _tree_type = SklearnKDTree

    def __call__(self, a):
        result = super().__call__(a)
        return KDTree(result.data)


@require_not_none(SklearnKDTree)
class KDTreeQuery(TreeQueryBase):
    _op_type_ = OperandDef.KD_TREE_QUERY
    _tree_type = SklearnKDTree


@require_not_none(SklearnKDTree)
def kd_tree_query(tree, data, n_neighbors, return_distance):
    op = KDTreeQuery(tree=tree, n_neighbors=n_neighbors,
                     return_distance=return_distance)
    ret = op(data)
    if not return_distance:
        return ret[0]
    return ret


@require_not_none(SklearnKDTree)
def create_kd_tree(X, leaf_size, metric=None, **metric_params):
    # kd_tree cannot accept callable metric
    assert not callable(metric)
    op = _KDTree(leaf_size=leaf_size, metric=metric,
                 **metric_params)
    return op(X)
