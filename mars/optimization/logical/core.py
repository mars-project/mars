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

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Type

from ...core import OperandType, ChunkType, EntityType, enter_mode
from ...core.graph import EntityGraph


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
        if record.record_type in (OptimizationRecordType.replace,
                                  OptimizationRecordType.delete):
            self._original_chunk_to_records[record.original_chunk] = record
        if record.record_type in (OptimizationRecordType.new,
                                  OptimizationRecordType.replace):
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
    _instances: \
        Dict[Tuple[Type["OptimizationRule"], EntityGraph,
                   OptimizationRecords], "OptimizationRule"] = dict()

    def __init__(self,
                 graph: EntityGraph,
                 records: OptimizationRecords,
                 optimizer_cls: Type["Optimizer"]):
        self._graph = graph
        self._records = records
        self._optimizer_cls = optimizer_cls

    def __new__(cls,
                graph: EntityGraph,
                records: OptimizationRecords,
                optimizer_cls: Type["Optimizer"]):
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

    def _replace_node(self,
                      original_node: EntityType,
                      new_node: EntityType):
        predecessors = self._graph.predecessors(original_node)
        successors = self._graph.successors(original_node)
        self._graph.remove_node(original_node)
        self._graph.add_node(new_node)
        for pred in predecessors:
            self._graph.add_edge(pred, new_node)
        for succ in successors:
            self._graph.add_edge(new_node, succ)


class Optimizer(ABC):
    _rules: Dict[Type[OperandType], List[Type[OptimizationRule]]]

    @classmethod
    def register_rule(cls,
                      operand_types: List[Type[OperandType]],
                      rule: Type[OptimizationRule]):
        if not hasattr(cls, '_rules'):
            cls._rules = defaultdict(list)
        for operand_type in operand_types:
            cls._rules[operand_type].append(rule)

    @classmethod
    def get_rule_types(cls,
                       operand_type: Type[OperandType]) \
            -> List[Type[OptimizationRule]]:
        return cls._rules.get(operand_type)

    @classmethod
    def _replace_inputs(cls,
                        graph: EntityGraph,
                        records: OptimizationRecords):
        for node in graph:
            for succ in graph.successors(node):
                new_inputs = []
                input_opt = False
                pred_iter = graph.iter_predecessors(succ)
                inp_to_pred = dict()
                for inp in succ.inputs:
                    if inp not in inp_to_pred:
                        pred = next(pred_iter)
                        inp_to_pred[inp] = pred
                    else:
                        pred = inp_to_pred[inp]
                    optimized = records.get_optimization_result(pred)
                    if optimized is None:
                        optimized = pred
                    if optimized is not inp:
                        input_opt = True
                    new_inputs.append(optimized)
                if input_opt:
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

        visited = set()
        for entity in list(graph.topological_iter()):
            op = entity.op
            if op in visited:
                continue
            visited.add(op)

            rule_types = cls._rules.get(type(op))
            if not rule_types:
                continue

            rules = [rule_type(graph, records, cls)
                     for rule_type in rule_types]
            for rule in rules:
                if entity not in graph:  # pragma: no cover
                    # maybe removed during optimization
                    continue
                if rule.match(op):
                    rule.apply(op)
        cls._replace_inputs(graph, records)
        return records
