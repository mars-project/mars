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
import functools
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Type, Set

from ...core import OperandType, EntityType, enter_mode, Entity
from ...core.graph import EntityGraph
from ...utils import implements


class OptimizationRecordType(Enum):
    replace = 0
    new = 1
    delete = 2


@dataclass
class OptimizationRecord:
    original_entity: EntityType = None
    new_entity: EntityType = None
    record_type: OptimizationRecordType = None


class OptimizationRecords:
    _records: List[OptimizationRecord]
    _original_entity_to_records: Dict[EntityType, OptimizationRecord]

    def __init__(self):
        self._records = list()
        self._original_entity_to_records = dict()
        self._optimized_entity_to_records = dict()

    def append_record(self, record: OptimizationRecord):
        self._records.append(record)
        if record.record_type in (
            OptimizationRecordType.replace,
            OptimizationRecordType.delete,
        ):
            self._original_entity_to_records[record.original_entity] = record
        if record.record_type in (
            OptimizationRecordType.new,
            OptimizationRecordType.replace,
        ):
            self._optimized_entity_to_records[record.new_entity] = record

    def get_optimization_result(
        self, original_entity: EntityType, default: Optional[EntityType] = None
    ) -> EntityType:
        entity = original_entity
        if entity not in self._original_entity_to_records:
            return default
        while entity in self._original_entity_to_records:
            record = self._original_entity_to_records[entity]
            if record.record_type == OptimizationRecordType.replace:
                entity = record.new_entity
            else:
                assert record.record_type == OptimizationRecordType.delete
                return None
        return entity

    def get_original_entity(
        self, optimized_entity: EntityType, default: Optional[EntityType] = None
    ) -> EntityType:
        entity = optimized_entity
        if entity not in self._optimized_entity_to_records:
            return default
        while entity in self._optimized_entity_to_records:
            record = self._optimized_entity_to_records[entity]
            if record.record_type == OptimizationRecordType.replace:
                entity = record.original_entity
            else:
                assert record.record_type == OptimizationRecordType.new
                return None
        return entity


class OptimizationRule(ABC):
    def __init__(
        self,
        graph: EntityGraph,
        records: OptimizationRecords,
        optimizer_cls: Type["Optimizer"],
    ):
        self._graph = graph
        self._records = records
        self._optimizer_cls = optimizer_cls
        self._cached_rule = functools.lru_cache(maxsize=None)(
            lambda _rule_type: _rule_type(
                self._graph, self._records, self._optimizer_cls
            )
        )

    @abstractmethod
    def apply(self) -> bool:
        """
        Apply the rule to the graph.

        Returns
        -------
        bool
            If the graph got optimized.
        """
        pass

    def _replace_node(self, original_node: EntityType, new_node: EntityType):
        predecessors = self._graph.predecessors(original_node)
        successors = self._graph.successors(original_node)
        self._graph.remove_node(original_node)
        self._graph.add_node(new_node)
        for pred in predecessors:
            self._graph.add_edge(pred, new_node)
        for succ in successors:
            self._graph.add_edge(new_node, succ)

    def _replace_subgraph(
        self,
        graph: Optional[EntityGraph],
        nodes_to_remove: Optional[Set[EntityType]],
        new_results: Optional[List[Entity]] = None,
    ):
        """
        Replace the subgraph from the self._graph represented by a list of nodes with input graph.
        It will delete the nodes in removed_nodes with all linked edges first, and then add (or update if it's still
        existed in self._graph) the nodes and edges of the input graph.

        Parameters
        ----------
        graph : EntityGraph, optional
            The input graph. If it's none, no new node and edge will be added.
        nodes_to_remove : Set[EntityType], optional
            The nodes to be removed. All the edges connected with them are removed as well.
        new_results : List[Entity], optional, default None
            The new results to be replaced to the original by their keys.

        Raises
        ------
        ValueError
            1. If the input key of the removed node's successor can't be found in the subgraph.
            2. Or some of the nodes of the subgraph are in removed ones.
            3. Or some of the removed nodes are also in the results.
            4. Or the key of the new result can't be found in the original results.
        """
        affected_successors = set()
        output_to_node = dict()
        nodes_to_remove = nodes_to_remove or set()
        new_results = new_results or list()
        result_indices = {
            result.key: idx for idx, result in enumerate(self._graph.results)
        }

        if graph is not None:
            # Add the output key -> node of the subgraph
            for node in graph.iter_nodes():
                if node in nodes_to_remove:
                    raise ValueError(f"The node {node} is in the removed set")
                for output in node.outputs:
                    output_to_node[output.key] = node

        # Add the output key -> node of the original graph
        for node in self._graph.iter_nodes():
            if node not in nodes_to_remove:
                for output in node.outputs:
                    output_to_node[output.key] = node

        # Check if the updated result is valid
        for result in new_results:
            if result.key not in result_indices:
                raise ValueError(f"Unknown result {result} to replace")
            if result.key not in output_to_node:
                raise ValueError(f"The result {result} is missing in the updated graph")

        for node in nodes_to_remove:
            for affected_successor in self._graph.iter_successors(node):
                if affected_successor not in nodes_to_remove:
                    affected_successors.add(affected_successor)
        # Check whether affected successors' inputs are in subgraph
        for affected_successor in affected_successors:
            for inp in affected_successor.inputs:
                if inp.key not in output_to_node:
                    raise ValueError(
                        f"The output {inp} of node {affected_successor} is missing in the subgraph"
                    )
        # Here all the pre-check are passed, we start to replace the subgraph
        for node in nodes_to_remove:
            self._graph.remove_node(node)

        if graph is None:
            return

        for node in graph.iter_nodes():
            self._graph.add_node(node)

        for node in itertools.chain(graph.iter_nodes(), affected_successors):
            for inp in node.inputs:
                pred_node = output_to_node[inp.key]
                self._graph.add_edge(pred_node, node)

        for result in new_results:
            self._graph.results[result_indices[result.key]] = result


class OperandBasedOptimizationRule(OptimizationRule):
    """
    Optimization rule that optimize certain operands of the graph in topological way.
    """

    _rule_type_to_op_types: Dict[
        Type[OptimizationRule], Set[Type[OperandType]]
    ] = defaultdict(set)

    @implements(OptimizationRule.apply)
    def apply(self) -> bool:
        visited = set()
        optimized = False
        for entity in list(self._graph.topological_iter()):
            op = entity.op
            if op in visited:
                continue
            visited.add(op)

            if entity not in self._graph:  # pragma: no cover
                # maybe removed during optimization
                continue
            op_types = self._rule_type_to_op_types[type(self)]
            if isinstance(op, tuple(op_types)) and self.match_operand(op):
                optimized = True
                self.apply_to_operand(op)

        return optimized

    @abstractmethod
    def apply_to_operand(self, op: OperandType) -> None:
        """
        Apply this rule to the given operand.

        Parameters
        ----------
        op : OperandType
            Operand.
        """
        pass

    @abstractmethod
    def match_operand(self, op: OperandType) -> bool:
        """
        If this operand matches this rule.

        Parameters
        ----------
        op : OperandType
            Operand.

        Returns
        -------
        bool
            If this operand matches this rule.
        """
        pass

    @classmethod
    def register_operand(cls, op_type: Type[OperandType]):
        cls._rule_type_to_op_types[cls].add(op_type)
        for derived in op_type.__subclasses__():
            cls._rule_type_to_op_types[cls].add(derived)


class Optimizer(ABC):
    _rule_types: List[Type[OptimizationRule]]

    @classmethod
    def register_rule(cls, rule_type: Type[OptimizationRule]):
        if not hasattr(cls, "_rule_types"):
            cls._rule_types = []
        cls._rule_types.append(rule_type)

    @classmethod
    def _replace_inputs(cls, graph: EntityGraph, records: OptimizationRecords):
        for node in graph:
            for succ in graph.successors(node):
                input_optimized = False
                new_inputs = []
                for inp in succ.inputs:
                    optimized = records.get_optimization_result(inp)
                    if optimized is None:
                        optimized = inp
                    if optimized is not inp:
                        input_optimized = True
                    new_inputs.append(optimized)
                if input_optimized:
                    succ.inputs = new_inputs

    @classmethod
    @enter_mode(build=True)
    def optimize(cls, graph: EntityGraph) -> OptimizationRecords:
        """
        Optimize a graph.

        Parameters
        ----------
        graph : EntityGraph
            Tileable or chunk graph.

        Returns
        -------
        optimization_records : OptimizationRecords
            Optimization records.
        """
        records = OptimizationRecords()
        cached_rule = functools.lru_cache(maxsize=None)(
            lambda _rule_type: _rule_type(graph, records, cls)
        )

        for rule_type in cls._rule_types:
            rule = cached_rule(rule_type)
            if rule.apply():
                cls._replace_inputs(graph, records)
                new_results = []
                for result in graph.results:
                    new_results.append(
                        records.get_optimization_result(result, default=result)
                    )
                graph.results = new_results

        return records
