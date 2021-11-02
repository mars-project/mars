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

import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Type

from ...core import OperandType, ChunkType, EntityType, enter_mode
from ...core.graph import EntityGraph
from ...core.operand import Operand


class OptimizationRecordType(Enum):
    replace = 0
    new = 1
    delete = 2


@dataclass
class OptimizationRecord:
    original_chunk: ChunkType = None
    new_chunk: ChunkType = None
    record_type: OptimizationRecordType = None


class OptimizationRecords:
    _records: List[OptimizationRecord]
    _original_chunk_to_records: Dict[ChunkType, OptimizationRecord]

    def __init__(self):
        self._records = list()
        self._original_chunk_to_records = dict()
        self._optimized_chunk_to_records = dict()

    def append_record(self, record: OptimizationRecord):
        self._records.append(record)
        if record.record_type in (
            OptimizationRecordType.replace,
            OptimizationRecordType.delete,
        ):
            self._original_chunk_to_records[record.original_chunk] = record
        if record.record_type in (
            OptimizationRecordType.new,
            OptimizationRecordType.replace,
        ):
            self._optimized_chunk_to_records[record.new_chunk] = record

    def get_optimization_result(self, original_chunk: ChunkType) -> ChunkType:
        chunk = original_chunk
        if chunk not in self._original_chunk_to_records:
            return
        while chunk in self._original_chunk_to_records:
            record = self._original_chunk_to_records[chunk]
            if record.record_type == OptimizationRecordType.replace:
                chunk = record.new_chunk
            else:
                assert record.record_type == OptimizationRecordType.delete
                return None
        return chunk

    def get_original_chunk(self, optimized_chunk: ChunkType) -> ChunkType:
        chunk = optimized_chunk
        if chunk not in self._optimized_chunk_to_records:
            return
        while chunk in self._optimized_chunk_to_records:
            record = self._optimized_chunk_to_records[chunk]
            if record.record_type == OptimizationRecordType.replace:
                chunk = record.original_chunk
            else:
                assert record.record_type == OptimizationRecordType.new
                return None
        return chunk


class OptimizationRule(ABC):
    _instances: Dict[
        Tuple[Type["OptimizationRule"], EntityGraph, OptimizationRecords],
        "OptimizationRule",
    ] = dict()
    _preds_to_remove = weakref.WeakKeyDictionary()

    def __init__(
        self,
        graph: EntityGraph,
        records: OptimizationRecords,
        optimizer_cls: Type["Optimizer"],
    ):
        self._graph = graph
        self._records = records
        self._optimizer_cls = optimizer_cls

    def __new__(
        cls,
        graph: EntityGraph,
        records: OptimizationRecords,
        optimizer_cls: Type["Optimizer"],
    ):
        if (cls, graph, records) in cls._instances:
            return cls._instances[cls, graph, records]
        inst = cls._instances[cls, graph, records] = object.__new__(cls)
        return inst

    @abstractmethod
    def match(self, op: OperandType) -> bool:
        """
        If this operand matches this rule.

        Parameters
        ----------
        op : OperandType
            Operand.

        Returns
        -------
        matched : bool
            Matched rule or not.
        """

    @abstractmethod
    def apply(self, op: OperandType):
        """
        Apply rule to an operand.

        Parameters
        ----------
        op : OperandType
            Operand
        """

    def _replace_node(self, original_node: EntityType, new_node: EntityType):
        predecessors = self._graph.predecessors(original_node)
        successors = self._graph.successors(original_node)
        self._graph.remove_node(original_node)
        self._graph.add_node(new_node)
        for pred in predecessors:
            self._graph.add_edge(pred, new_node)
        for succ in successors:
            self._graph.add_edge(new_node, succ)

    @classmethod
    def _add_collapsable_predecessor(cls, node: EntityType, predecessor: EntityType):
        if predecessor not in cls._preds_to_remove:
            cls._preds_to_remove[predecessor] = {node}
        else:
            cls._preds_to_remove[predecessor].add(node)

    def _remove_collapsable_predecessors(self, node: EntityType):
        node = self._records.get_optimization_result(node) or node
        preds_opt_to_remove = []
        for pred in self._graph.predecessors(node):
            pred_original = self._records.get_original_chunk(pred) or pred
            pred_opt = self._records.get_optimization_result(pred) or pred
            if pred_opt in self._graph.results or pred_original in self._graph.results:
                continue
            affect_succ = self._preds_to_remove.get(pred_original) or []
            affect_succ_opt = [
                self._records.get_optimization_result(s) or s for s in affect_succ
            ]
            if all(s in affect_succ_opt for s in self._graph.successors(pred)):
                preds_opt_to_remove.append((pred_original, pred_opt))

        for pred_original, pred_opt in preds_opt_to_remove:
            self._graph.remove_node(pred_opt)
            self._records.append_record(
                OptimizationRecord(pred_original, None, OptimizationRecordType.delete)
            )


class Optimizer(ABC):
    _rules: List[Type[OptimizationRule]]
    _op_to_rules: Dict[Type[OperandType], List[Type[OptimizationRule]]]

    @classmethod
    def register_rule(
        cls, operand_types: List[Type[OperandType]], rule: Type[OptimizationRule]
    ):
        if not hasattr(cls, "_rules"):
            cls._rules = []
        cls._rules.append(rule)

        if not hasattr(cls, "_op_to_rules"):
            cls._op_to_rules = defaultdict(list)
        for operand_type in operand_types:
            cls._op_to_rules[operand_type].append(rule)

    @classmethod
    def get_rule_types(
        cls, operand_type: Type[OperandType]
    ) -> List[Type[OptimizationRule]]:
        rule_types = cls._op_to_rules.get(operand_type, None)
        if rule_types is None:
            for op_cls in operand_type.__mro__:
                if op_cls is Operand:
                    break
                rule_types = cls._op_to_rules.get(op_cls)
                if rule_types is not None:
                    break
            cls._op_to_rules[operand_type] = rule_types or []
        return rule_types

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
        optimized = False
        for rule_type in cls._rules:
            visited = set()
            for entity in list(graph.topological_iter()):
                op = entity.op
                if op in visited:
                    continue
                visited.add(op)

                rule_types = cls.get_rule_types(type(op)) or []
                if rule_type not in rule_types:
                    continue

                rule = rule_type(graph, records, cls)
                if entity not in graph:  # pragma: no cover
                    # maybe removed during optimization
                    continue
                if rule.match(op):
                    optimized = True
                    rule.apply(op)
        if optimized:
            cls._replace_inputs(graph, records)
        return records
