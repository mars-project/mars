class Optimizer(object):
    def __init__(self, graph):
        self.explored = set()
        self.graph = graph
        self.keys = []

    # in compose traverse, whether a node should be skipped
    def _can_skip(self, node):
        raise NotImplementedError

    # in compose traverse, whether a node should be break
    def _can_break(self, node):
        raise NotImplementedError

    # whether a node is supported for jax or numexpr optimization
    def _support(self, node):
        raise NotImplementedError

    def compose(self, keys):
        composes = []
        self.keys = set(keys or [])
        graph = self.graph
        for v in graph.bfs():
            if v.op.gpu or v.op.sparse:
                # break out
                return []
            if self._can_skip(v):
                continue
            selected = [v]
            # add successors
            cur_node = graph.successors(v)[0]
            while graph.count_predecessors(cur_node) == 1 \
                    and self._support(cur_node) and cur_node.key not in self.keys:
                selected.append(cur_node)
                if self._can_break(cur_node):
                    break
                else:
                    cur_node = graph.successors(cur_node)[0]
            if len(selected) > 1:
                self.explored.update(selected)
                composes.append(list(selected))

        # compose to fused node
        composed_nodes = []

        for c in composes:
            head_node = c[0]
            tail_node = c[-1]

            op = self._get_fused_chunk(tail_node)
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
