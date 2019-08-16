from .optimizer import Optimizer


class JaxOptimizer(Optimizer):
    def __init__(self, graph):
        super(JaxOptimizer, self).__init__(graph)

    def optimize(self, keys=None):
        self.compose(keys)

    def _can_break(self, node):
        if not self._support(node):
            return True
        if self.graph.count_successors(node) != 1:
            return True
        return False

    def _support(self, node):
        op = node.op
        if hasattr(op, 'jax_function') and op.jax_function() is not NotImplementedError:
            return True
        return False

    def _can_skip(self, node):
        if super(JaxOptimizer, self)._can_skip(node):
            return True
        return False

    @staticmethod
    def _get_fused_chunk(tail_node):
        from ..tensor.fuse import TensorJaxFuseChunk
        return TensorJaxFuseChunk(dtype=tail_node.dtype)
