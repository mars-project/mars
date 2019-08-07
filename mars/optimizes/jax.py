from ..tensor.fuse import TensorJaxFuseChunk


class JaxOptimizer(object):
    def __init__(self, graph):
        self._graph = graph

    def optimize(self, keys=None):
        self.compose(keys=keys)

    def _compose_graph(self, composes):
        graph = self._graph
        composed_nodes = []

        for c in composes:
            head_node = c[0]
            tail_node = c[-1]

            op = TensorJaxFuseChunk(dtype=tail_node.dtype)
            composed_chunk = op(c).data
            graph.add_node(composed_chunk)
            for node in graph.iter_successors(tail_node):
                graph.add_edge(composed_chunk, node)
            for node in graph.iter_predecessors(head_node):
                graph.add_edge(node, composed_chunk)
            for node in c:
                graph.remove_node(node)
            composed_nodes.append(composed_chunk)

        return composed_nodes

    def compose(self, keys=None):
        composes = []
        visited = set()
        keys = set(keys or [])

        graph = self._graph
        for v in graph.bfs():
            if v.op.gpu or v.op.sparse:
                # break out
                return []
            if not hasattr(v.op, 'execute_jax') or v.key in keys:
                continue
            if v in visited:
                continue
            if graph.count_successors(v) != 1:
                continue
            selected = [v]
            # add successors
            cur_node = graph.successors(v)[0]
            while graph.count_predecessors(cur_node) == 1 \
                    and hasattr(cur_node.op, 'execute_jax') and cur_node.key not in keys:
                selected.append(cur_node)
                if graph.count_successors(cur_node) != 1:
                    break
                else:
                    cur_node = graph.successors(cur_node)[0]
            if len(selected) > 1:
                visited.update(selected)
                composes.append(list(selected))
        return self._compose_graph(composes)
