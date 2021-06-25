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


from typing import List, Type

from ....core import OperandType, ChunkGraph
from ..core import Optimizer, OptimizationRule, OptimizationRecords


class ChunkOptimizer(Optimizer):
    """
    Tileable Optimizer
    """


def register_chunk_optimization_rule(op_types: List[Type[OperandType]]):
    def wrap(rule_type: Type[OptimizationRule]):
        ChunkOptimizer.register_rule(op_types, rule_type)
        return rule_type
    return wrap


def optimize(chunk_graph: ChunkGraph) -> OptimizationRecords:
    return ChunkOptimizer.optimize(chunk_graph)
