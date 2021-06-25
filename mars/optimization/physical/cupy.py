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

from ...tensor import arithmetic
from ...tensor.fuse import TensorCpFuseChunk
from ...utils import lazy_import
from .core import RuntimeOptimizer, register_optimizer


cp = lazy_import('cupy', globals=globals(), rename='cp')
CP_INSTALLED = cp is not None

CP_ELEMENTWISE_OP = {
    arithmetic.TensorSubtract,
    arithmetic.TensorMultiply,
    arithmetic.TensorTrueDiv,
    arithmetic.TensorSqrt
}
CP_OP = CP_ELEMENTWISE_OP


@register_optimizer
class CupyRuntimeOptimizer(RuntimeOptimizer):
    engine = 'cupy'

    @classmethod
    def is_available(cls) -> bool:
        return CP_INSTALLED

    def optimize(self):
        fuses = []
        explored = set()

        graph = self._graph
        for node in graph.topological_iter():
            if type(node.op) not in CP_OP:
                continue
            if node in explored:
                continue
            if graph.count_predecessors(node) != 1:
                continue
            if node in graph.results:
                continue

            selected = [node]
            # add successors
            cur_node = graph.successors(node)[0]
            while graph.count_predecessors(cur_node) == 1 and \
                    type(cur_node.op) in CP_OP:
                selected.append(cur_node)
                if graph.count_successors(cur_node) != 1 or \
                        cur_node in graph.results:
                    break
                else:
                    cur_node = graph.successors(cur_node)[0]
            if len(selected) > 1:
                explored.update(selected)
                fuses.append(list(selected))

        return self._fuse_nodes(fuses, TensorCpFuseChunk)
