from .utils import Optimizer


class JaxOptimizer(Optimizer):
    def __init__(self, graph):
        super(JaxOptimizer, self).__init__(graph, 'numexpr')

    def optimize(self, keys=None):
        self.compose(keys)

    def _can_break(self, node):
        if not self._jax_compat(node.op):
            return True

        if self.graph.count_successors(node) != 1:
            return True
        return False

    def _support(self, node):
        if self._jax_compat(node.op):
            return True
        return False

    def _can_skip(self, node):
        if not self._jax_compat(node.op) or node.key in self.keys:
            return True
        if node in self.explored:
            return True
        if self.graph.count_successors(node) != 1:
            return True
        return False

    @staticmethod
    def _get_fused_chunk(tail_node):
        from ..tensor.fuse import TensorJaxFuseChunk
        return TensorJaxFuseChunk(dtype=tail_node.dtype)
