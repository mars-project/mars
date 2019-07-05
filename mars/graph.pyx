#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

from .compat import six, Enum
from .serialize import Serializable
from .serialize.core cimport ValueType, ProviderType, \
    OneOfField, ListField, Int8Field

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
        return node in self

    def add_node(self, node, node_attr=None, **node_attrs):
        if node_attr is None:
            node_attr = node_attrs
        else:
            try:
                node_attr.update(node_attrs)
            except AttributeError:
                raise TypeError('The node_attr argument must be a dictionary')

        if node not in self._nodes:
            self._nodes[node] = node_attr
            self._successors[node] = dict()
            self._predecessors[node] = dict()
        else:
            self._nodes[node].update(node_attr)

    def remove_node(self, node):
        if node not in self._nodes:
            raise KeyError('Node %s does not exist in the directed graph' % node)

        del self._nodes[node]

        for succ in self._successors[node]:
            del self._predecessors[succ][node]
        del self._successors[node]

        for pred in self._predecessors[node]:
            del self._successors[pred][node]
        del self._predecessors[node]

    def add_edge(self, u, v, edge_attr=None, **edge_attrs):
        cdef:
            dict u_succ, v_pred
        for n in (u, v):
            if n not in self:
                raise KeyError('Node %s does not exist in the directed graph' % n)

        if edge_attr is None:
            edge_attr = edge_attrs
        else:
            try:
                edge_attr.update(edge_attrs)
            except AttributeError:
                raise TypeError('The edge_attr argument must be a dictionary')

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
            raise KeyError('Edge %s->%s does not exist in the directed graph' % (u, v))

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
            raise KeyError('Node %s does not exist in the directed graph' % n)

    def iter_successor_items(self):
        return self._successors.items()

    cpdef list successors(self, n):
        return list(self._successors[n])

    def iter_predecessors(self, n):
        try:
            return iter(self._predecessors[n])
        except KeyError:
            raise KeyError('Node %s does not exist in the directed graph' % n)

    def iter_predecessor_items(self):
        return self._predecessors.items()

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

    def traverse(self, visit_predicate=None):
        cdef:
            set visited = set()

        def _default_visit_predicate(n, visited):
            preds = self.predecessors(n)
            return not preds or all(pred in visited for pred in preds)

        q = deque(self.iter_indep())
        visit_predicate = visit_predicate or _default_visit_predicate

        while q:
            node = q.popleft()
            if node in visited:
                continue
            preds = self.predecessors(node)
            if visit_predicate(node, visited):
                yield node
                visited.add(node)
                q.extend(n for n in self[node] if n not in visited)
            else:
                q.extend(n for n in preds if n not in visited)
                q.append(node)

    def dfs(self, start=None, visit_predicate=None, successors=None, reverse=False):
        cdef:
            set visited = set()
            list stack

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
        visit_predicate = visit_predicate or _default_visit_predicate

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            preds = self.predecessors(node)
            if visit_predicate(node, visited):
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

        if start:
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

    def build_undirected(self):
        graph = DirectedGraph()
        for n in self:
            if n not in graph:
                graph.add_node(n)
            for succ in self.iter_successors(n):
                if succ not in graph:
                    graph.add_node(succ)
                graph.add_edge(n, succ)
                graph.add_edge(succ, n)
        return graph

    def build_reversed(self):
        graph = DirectedGraph()
        for n in self:
            if n not in graph:
                graph.add_node(n)
            for succ in self.iter_successors(n):
                if succ not in graph:
                    graph.add_node(succ)
                graph.add_edge(succ, n)
        return graph

    def serialize(self):
        cdef:
            set visited = set()
            list nodes = []

        from .tensor.core import CHUNK_TYPE, TENSOR_TYPE, Chunk, Tensor

        level = None

        def add_obj(c):
            # we only record the op key in serialized chunk or tensor,
            # so the op should be added as a serializable node
            if c.op not in visited:
                nodes.append(SerializableGraphNode(_node=c.op))
                visited.add(c.op)
            if c not in visited:
                nodes.append(SerializableGraphNode(_node=c))
                visited.add(c)

        for node in self.iter_nodes():
            if isinstance(node, CHUNK_TYPE):
                node = node.data if isinstance(node, Chunk) else node
                add_obj(node)
                if node.composed:
                    for c in node.composed:
                        nodes.append(SerializableGraphNode(_node=c.op))
            elif isinstance(node, TENSOR_TYPE):
                node = node.data if isinstance(node, Tensor) else node
                if level is None:
                    level = SerializableGraph.Level.TENSOR
                for c in (node.chunks or ()):
                    add_obj(c)
                add_obj(node)
            else:
                raise TypeError('Unknown node type to serialize: {0}'.format(type(node)))

        s_graph = SerializableGraph(_nodes=nodes)
        s_graph.level = level if level is not None else SerializableGraph.Level.CHUNK
        return s_graph

    @classmethod
    def _repr_in_dot(cls, val):
        if isinstance(val, bool):
            return 'true' if val else 'false'
        if isinstance(val, six.string_types):
            return '"{0}"'.format(val)
        return val

    def to_dot(self, graph_attrs=None, node_attrs=None):
        sio = six.StringIO()
        sio.write('digraph {\n')
        sio.write('splines=curved\n')
        sio.write('rankdir=BT\n')

        if graph_attrs:
            sio.write('graph [{0}];\n'.format(
                ' '.join('{0}={1}'.format(k, self._repr_in_dot(v))
                         for k, v in six.iteritems(graph_attrs))))
        if node_attrs:
            sio.write('node [{0}];\n'.format(
                ' '.join('{0}={1}'.format(k, self._repr_in_dot(v))
                         for k, v in six.iteritems(node_attrs))))

        chunk_style = '[shape=box]'
        operand_style = '[shape=circle]'

        visited = set()
        for node in self.iter_nodes():
            op = node.op
            if op.key in visited:
                continue
            for input_chunk in (op.inputs or []):
                if input_chunk.key not in visited:
                    sio.write('"Chunk:%s" %s\n' % (input_chunk.key[:5], chunk_style))
                    visited.add(input_chunk.key)
                if op.key not in visited:
                    sio.write('"%s:%s" %s\n' % (type(op).__name__, op.key[:5], operand_style))
                    visited.add(op.key)
                sio.write('"Chunk:%s" -> "%s:%s"\n' % (input_chunk.key[:5], type(op).__name__, op.key[:5]))

            for output_chunk in (op.outputs or []):
                if output_chunk.key not in visited:
                    sio.write('"Chunk:%s" %s\n' % (output_chunk.key[:5], chunk_style))
                    visited.add(output_chunk.key)
                if op.key not in visited:
                    sio.write('"%s:%s" %s\n' % (type(op).__name__, op.key[:5], operand_style))
                    visited.add(op.key)
                sio.write('"%s:%s" -> "Chunk:%s"\n' % (type(op).__name__, op.key[:5], output_chunk.key[:5]))

        sio.write('}')
        return sio.getvalue()

    def _repr_svg_(self):
        from graphviz import Source

        return Source(self.to_dot())._repr_svg_()

    @classmethod
    def deserialize(cls, s_graph):
        from .tensor.core import ChunkData, TensorData

        graph = cls()
        node_type = TensorData if s_graph.level == SerializableGraph.Level.TENSOR else ChunkData
        for node in s_graph.nodes:
            if isinstance(node, node_type):
                graph.add_node(node)
                if node.inputs:
                    for inode in node.inputs:
                        graph.add_node(inode)
                        graph.add_edge(inode, node)

        return graph

    def to_pb(self, pb_obj=None):
        return self.serialize().to_pb(obj=pb_obj)

    @classmethod
    def from_pb(cls, pb_obj):
        try:
            return cls.deserialize(SerializableGraph.from_pb(pb_obj))
        except KeyError as e:
            logger.error('Failed to deserialize graph, graph_def: {0}'.format(pb_obj))
            raise

    def to_json(self, json_obj=None):
        return self.serialize().to_json(obj=json_obj)

    @classmethod
    def from_json(cls, json_obj):
        return cls.deserialize(SerializableGraph.from_json(json_obj))

    def compose(self, list keys=None):
        from .fuse import Fusion

        return Fusion(self).compose(keys=keys)

    def decompose(self, nodes=None):
        from .fuse import Fusion

        Fusion(self).decompose(nodes=nodes)

    def view(self, filename='default', graph_attrs=None, node_attrs=None):
        from graphviz import Source

        g = Source(self.to_dot(graph_attrs, node_attrs))
        g.view(filename=filename, cleanup=True)


class GraphContainsCycleError(Exception):
    pass


cdef class DAG(DirectedGraph):
    def topological_iter(self, succ_checker=None, reverse=False):
        cdef:
            dict preds, succs
            set visited = set()
            list stack

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
        if self._nodes and not stack:
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


class SerializableGraphNode(Serializable):
    _node = OneOfField('node', op='mars.operands.Operand',
                       chunk='mars.tensor.core.ChunkData', tensor='mars.tensor.core.TensorData')

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from .serialize.protos.graph_pb2 import GraphDef

            return GraphDef.NodeDef

        return super(SerializableGraphNode, cls).cls(provider)

    @property
    def node(self):
        return getattr(self, '_node', None)


class SerializableGraph(Serializable):
    class Level(Enum):
        CHUNK = 0
        TENSOR = 1

    _nodes = ListField('node', ValueType.reference(SerializableGraphNode))
    _level = Int8Field('level')

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from .serialize.protos.graph_pb2 import GraphDef

            return GraphDef

        return super(SerializableGraph, cls).cls(provider)

    @property
    def level(self):
        return SerializableGraph.Level(self._level)

    @level.setter
    def level(self, level):
        self._level = level.value

    @property
    def nodes(self):
        for n in self._nodes:
            yield n.node
