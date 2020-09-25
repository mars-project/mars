#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import logging
from collections import deque
from enum import Enum
from io import StringIO

from .serialize import Serializable
from .serialize.core cimport ValueType, ProviderType, ListField, Int8Field

logger = logging.getLogger(__name__)

cdef class DirectedGraph:
    cdef:
        dict _nodes
        dict _predecessors
        dict _successors

    def __init__(self):
        self._nodes = dict()
        self._predecessors = dict()
        self._successors = dict()

    def __iter__(self):
        return iter(self._nodes)

    def __contains__(self, n):
        return n in self._nodes

    def __len__(self):
        return len(self._nodes)

    def __getitem__(self, n):
        return self._successors[n]

    def contains(self, node):
        return node in self._nodes

    def add_node(self, node, node_attr=None, **node_attrs):
        if node_attr is None:
            node_attr = node_attrs
        else:
            try:
                node_attr.update(node_attrs)
            except AttributeError:
                raise TypeError('The node_attr argument must be a dictionary')
        self._add_node(node, node_attr)

    cdef inline _add_node(self, node, node_attr=None):
        if node_attr is None:
            node_attr = dict()
        if node not in self._nodes:
            self._nodes[node] = node_attr
            self._successors[node] = dict()
            self._predecessors[node] = dict()
        else:
            self._nodes[node].update(node_attr)

    def remove_node(self, node):
        if node not in self._nodes:
            raise KeyError(f'Node {node} does not exist in the directed graph')

        del self._nodes[node]

        for succ in self._successors[node]:
            del self._predecessors[succ][node]
        del self._successors[node]

        for pred in self._predecessors[node]:
            del self._successors[pred][node]
        del self._predecessors[node]

    def add_edge(self, u, v, edge_attr=None, **edge_attrs):
        if edge_attr is None:
            edge_attr = edge_attrs
        else:
            try:
                edge_attr.update(edge_attrs)
            except AttributeError:
                raise TypeError('The edge_attr argument must be a dictionary')
        self._add_edge(u, v, edge_attr)

    cdef inline _add_edge(self, u, v, edge_attr=None):
        cdef:
            dict u_succ, v_pred

        if u not in self._nodes:
            raise KeyError(f'Node {u} does not exist in the directed graph')
        if v not in self._nodes:
            raise KeyError(f'Node {v} does not exist in the directed graph')

        if edge_attr is None:
            edge_attr = dict()

        u_succ = self._successors[u]
        if v in u_succ:
            u_succ[v].update(edge_attr)
        else:
            u_succ[v] = edge_attr

        v_pred = self._predecessors[v]
        if u not in v_pred:
            # `update` is not necessary, as they point to the same object
            v_pred[u] = edge_attr

    def remove_edge(self, u, v):
        try:
            del self._successors[u][v]
            del self._predecessors[v][u]
        except KeyError:
            raise KeyError(f'Edge {u}->{v} does not exist in the directed graph')

    def has_successor(self, u, v):
        return (u in self._successors) and (v in self._successors[u])

    def has_predecessor(self, u, v):
        return (u in self._predecessors) and (v in self._predecessors[u])

    def iter_nodes(self, data=False):
        if data:
            return iter(self._nodes.items())
        return iter(self._nodes)

    def iter_successors(self, n):
        try:
            return iter(self._successors[n])
        except KeyError:
            raise KeyError(f'Node {n} does not exist in the directed graph')

    cpdef list successors(self, n):
        return list(self._successors[n])

    def iter_predecessors(self, n):
        try:
            return iter(self._predecessors[n])
        except KeyError:
            raise KeyError(f'Node {n} does not exist in the directed graph')

    cpdef list predecessors(self, n):
        return list(self._predecessors[n])

    def count_successors(self, n):
        return len(self._successors[n])

    def count_predecessors(self, n):
        return len(self._predecessors[n])

    def iter_indep(self, reverse=False):
        cdef dict preds
        preds = self._predecessors if not reverse else self._successors
        for n, p in preds.items():
            if len(p) == 0:
                yield n

    cpdef int count_indep(self, reverse=False):
        cdef:
            dict preds
            int result = 0
        preds = self._predecessors if not reverse else self._successors
        for n, p in preds.items():
            if len(p) == 0:
                result += 1
        return result

    def dfs(self, start=None, visit_predicate=None, successors=None, reverse=False):
        cdef:
            set visited = set()
            list stack
            bint visit_all = False

        if reverse:
            pred_fun, succ_fun = self.successors, self.predecessors
        else:
            pred_fun, succ_fun = self.predecessors, self.successors

        if start:
            if not isinstance(start, (list, tuple)):
                start = [start]
            stack = list(start)
        else:
            stack = list(self.iter_indep(reverse=reverse))

        def _default_visit_predicate(n, visited):
            cdef list preds
            preds = pred_fun(n)
            return not preds or all(pred in visited for pred in preds)

        successors = successors or succ_fun
        visit_all = (visit_predicate == 'all')
        visit_predicate = visit_predicate or _default_visit_predicate

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            preds = self.predecessors(node)
            if visit_all or visit_predicate(node, visited):
                yield node
                visited.add(node)
                stack.extend(n for n in successors(node) if n not in visited)
            else:
                stack.append(node)
                stack.extend(n for n in preds if n not in visited)

    def bfs(self, start=None, visit_predicate=None, successors=None, reverse=False):
        cdef:
            object queue
            object node
            set visited = set()
            bint visit_all = False

        if reverse:
            pred_fun, succ_fun = self.successors, self.predecessors
        else:
            pred_fun, succ_fun = self.predecessors, self.successors

        if start is not None:
            if not isinstance(start, (list, tuple)):
                start = [start]
            queue = deque(start)
        else:
            queue = deque(self.iter_indep(reverse=reverse))

        def _default_visit_predicate(n, visited):
            preds = pred_fun(n)
            return not preds or all(pred in visited for pred in preds)

        successors = successors or succ_fun
        visit_all = (visit_predicate == 'all')
        visit_predicate = visit_predicate or _default_visit_predicate

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            preds = pred_fun(node)
            if visit_all or visit_predicate(node, visited):
                yield node
                visited.add(node)
                queue.extend(n for n in successors(node) if n not in visited)
            else:
                queue.append(node)
                queue.extend(n for n in preds if n not in visited)

    def copy(self):
        cdef DirectedGraph graph = type(self)()
        for n in self:
            if n not in graph._nodes:
                graph._add_node(n)
            for succ in self.iter_successors(n):
                if succ not in graph._nodes:
                    graph._add_node(succ)
                graph._add_edge(n, succ)
        return graph

    def copyto(self, DirectedGraph other_graph):
        if other_graph is self:
            return

        other_graph._nodes = self._nodes.copy()
        other_graph._predecessors = self._predecessors.copy()
        other_graph._successors = self._successors.copy()

    def build_undirected(self):
        cdef DirectedGraph graph = DirectedGraph()
        for n in self:
            if n not in graph._nodes:
                graph._add_node(n)
            for succ in self._successors[n]:
                if succ not in graph._nodes:
                    graph._add_node(succ)
                graph._add_edge(n, succ)
                graph._add_edge(succ, n)
        return graph

    def build_reversed(self):
        cdef DirectedGraph graph = type(self)()
        for n in self:
            if n not in graph._nodes:
                graph._add_node(n)
            for succ in self._successors[n]:
                if succ not in graph._nodes:
                    graph._add_node(succ)
                graph._add_edge(succ, n)
        return graph

    def serialize(self):
        cdef:
            set visited = set()
            list nodes = []

        from .core import ENTITY_TYPE, TILEABLE_TYPE, Entity, TileableEntity

        level = None

        def add_obj(c):
            # we only record the op key in serialized chunk or tensor,
            # so the op should be added as a serializable node
            if c.op not in visited:
                nodes.append(c.op)
                visited.add(c.op)
            if c not in visited:
                nodes.append(c)
                visited.add(c)

        for node in self.iter_nodes():
            if isinstance(node, TILEABLE_TYPE):
                node = node.data if isinstance(node, TileableEntity) else node
                if level is None:
                    level = SerializableGraph.Level.ENTITY
                for c in (node.chunks or ()):
                    add_obj(c.data)
                add_obj(node)
            elif isinstance(node, ENTITY_TYPE):
                node = node.data if isinstance(node, Entity) else node
                add_obj(node)
                if getattr(node, 'composed', None):
                    for c in node.composed:
                        nodes.append(c.op)
            else:
                raise TypeError(f'Unknown node type to serialize: {type(node)}')

        s_graph = SerializableGraph(_nodes=nodes)
        s_graph.level = level if level is not None else SerializableGraph.Level.CHUNK
        return s_graph

    @classmethod
    def _repr_in_dot(cls, val):
        if isinstance(val, bool):
            return 'true' if val else 'false'
        if isinstance(val, str):
            return f'"{val}"'
        return val

    def to_dot(self, graph_attrs=None, node_attrs=None, trunc_key=5, result_chunk_keys=None):
        sio = StringIO()
        sio.write('digraph {\n')
        sio.write('splines=curved\n')
        sio.write('rankdir=BT\n')

        if graph_attrs:
            sio.write('graph [{0}];\n'.format(
                ' '.join(f'{k}={self._repr_in_dot(v)}' for k, v in graph_attrs.items())))
        if node_attrs:
            sio.write('node [{0}];\n'.format(
                ' '.join(f'{k}={self._repr_in_dot(v)}' for k, v in node_attrs.items())))

        chunk_style = '[shape=box]'
        operand_style = '[shape=circle]'

        visited = set()
        for node in self.iter_nodes():
            op = node.op
            op_name = type(op).__name__
            if op.stage is not None:
                op_name = f'{op_name}:{op.stage.name}'
            if op.key in visited:
                continue
            for input_chunk in (op.inputs or []):
                if input_chunk.key not in visited:
                    sio.write(f'"Chunk:{input_chunk.key[:trunc_key]}" {chunk_style}\n')
                    visited.add(input_chunk.key)
                if op.key not in visited:
                    sio.write(f'"{op_name}:{op.key[:trunc_key]}" {operand_style}\n')
                    visited.add(op.key)
                sio.write(f'"Chunk:{input_chunk.key[:trunc_key]}" -> "{op_name}:{op.key[:5]}"\n')

            for output_chunk in (op.outputs or []):
                if output_chunk.key not in visited:
                    tmp_chunk_style = chunk_style
                    if result_chunk_keys and output_chunk.key in result_chunk_keys:
                        tmp_chunk_style = '[shape=box,style=filled,fillcolor=cadetblue1]'
                    sio.write(f'"Chunk:{output_chunk.key[:trunc_key]}" {tmp_chunk_style}\n')
                    visited.add(output_chunk.key)
                if op.key not in visited:
                    sio.write(f'"{op_name}:{op.key[:trunc_key]}" {operand_style}\n')
                    visited.add(op.key)
                sio.write(f'"{op_name}:{op.key[:trunc_key]}" -> "Chunk:{output_chunk.key[:5]}"\n')

        sio.write('}')
        return sio.getvalue()

    def _repr_svg_(self):  # pragma: no cover
        from graphviz import Source
        return Source(self.to_dot())._repr_svg_()

    @classmethod
    def deserialize(cls, s_graph):
        cdef:
            DirectedGraph graph = cls()
            list nodes = []

        from .core import ChunkData, TileableData

        node_type = TileableData if s_graph.level == SerializableGraph.Level.ENTITY else ChunkData
        nodes = [node for node in s_graph.nodes if isinstance(node, node_type)]
        nodes_set = set(nodes)
        for node in nodes:
            graph._add_node(node)
            for inode in node.inputs or []:
                if inode in nodes_set:
                    if inode not in graph:
                        graph._add_node(inode)
                    graph._add_edge(inode, node)

        return graph

    def to_pb(self, pb_obj=None, data_serial_type=None, pickle_protocol=None):
        return self.serialize().to_pb(
            obj=pb_obj, data_serial_type=data_serial_type, pickle_protocol=pickle_protocol)

    @classmethod
    def from_pb(cls, pb_obj):
        try:
            return cls.deserialize(SerializableGraph.from_pb(pb_obj))
        except KeyError as e:
            logger.error(f'Failed to deserialize graph, graph_def: {pb_obj}')
            raise

    def to_json(self, json_obj=None, data_serial_type=None, pickle_protocol=None):
        return self.serialize().to_json(
            obj=json_obj, data_serial_type=data_serial_type, pickle_protocol=pickle_protocol)

    @classmethod
    def from_json(cls, json_obj):
        return cls.deserialize(SerializableGraph.from_json(json_obj))

    def compose(self, list keys=None):
        from .optimizes.chunk_graph.fuse import Fusion

        return Fusion(self).compose(keys=keys)

    def decompose(self, nodes=None):
        from .optimizes.chunk_graph.fuse import Fusion

        Fusion(self).decompose(nodes=nodes)

    def view(self, filename='default', graph_attrs=None, node_attrs=None, result_chunk_keys=None):  # pragma: no cover
        from graphviz import Source

        g = Source(self.to_dot(graph_attrs, node_attrs, result_chunk_keys=result_chunk_keys))
        g.view(filename=filename, cleanup=True)

    def to_dag(self):
        dag = DAG()
        dag._nodes = self._nodes.copy()
        dag._predecessors = self._predecessors.copy()
        dag._successors = self._successors.copy()
        return dag


class GraphContainsCycleError(Exception):
    pass


cdef class DAG(DirectedGraph):
    def to_dag(self):
        return self

    def topological_iter(self, succ_checker=None, reverse=False):
        cdef:
            dict preds, succs
            set visited = set()
            list stack

        if len(self) == 0:
            return

        if reverse:
            preds, succs = self._successors, self._predecessors
        else:
            preds, succs = self._predecessors, self._successors

        # copy predecessors and successors
        succs = dict((k, set(v)) for k, v in succs.items())
        preds = dict((k, set(v)) for k, v in preds.items())

        def _default_succ_checker(_, predecessors):
            return len(predecessors) == 0

        succ_checker = succ_checker or _default_succ_checker

        stack = list((p for p, l in preds.items() if len(l) == 0))
        if not stack:
            raise GraphContainsCycleError
        while stack:
            node = stack.pop()
            yield node
            visited.add(node)
            for succ in succs.get(node, {}):
                if succ in visited:
                    raise GraphContainsCycleError
                succ_preds = preds[succ]
                succ_preds.remove(node)
                if succ_checker(succ, succ_preds):
                    stack.append(succ)
        if len(visited) != len(self):
            raise GraphContainsCycleError


class SerializableGraph(Serializable):
    class Level(Enum):
        CHUNK = 0
        ENTITY = 1

    _nodes = ListField('node', ValueType.reference(None))
    _level = Int8Field('level')

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from .serialize.protos.graph_pb2 import GraphDef

            return GraphDef

        return super().cls(provider)

    @property
    def level(self):
        return SerializableGraph.Level(self._level)

    @level.setter
    def level(self, level):
        self._level = level.value

    @property
    def nodes(self):
        return self._nodes
