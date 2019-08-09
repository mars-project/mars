# from .ne import SUPPORT_OP, REDUCTION_OP
from ..tensor import arithmetic
from ..tensor import reduction

NE_REDUCTION_OP = (reduction.TensorSum, reduction.TensorProd,
                   reduction.TensorMax, reduction.TensorMin)
NE_SUPPORT_OP = (
    arithmetic.TensorSubtract,
    arithmetic.TensorMultiply,
    arithmetic.TensorDivide,
    arithmetic.TensorPower,
    arithmetic.TensorMod,
    arithmetic.TensorNegative,
    arithmetic.TensorAbs,
    arithmetic.TensorConj,
    arithmetic.TensorExp,
    arithmetic.TensorLog,
    arithmetic.TensorLog10,
    arithmetic.TensorExpm1,
    arithmetic.TensorLog1p,
    arithmetic.TensorSqrt,

    arithmetic.TensorEqual,
    arithmetic.TensorNotEqual,
    arithmetic.TensorLessThan,
    arithmetic.TensorLessEqual,
    arithmetic.TensorGreaterThan,
    arithmetic.TensorGreaterEqual,

    arithmetic.TensorSin,
    arithmetic.TensorCos,
    arithmetic.TensorTan,
    arithmetic.TensorArcsin,
    arithmetic.TensorArccos,
    arithmetic.TensorArctan,
    arithmetic.TensorSinh,
    arithmetic.TensorCosh,
    arithmetic.TensorTanh,
    arithmetic.TensorArcsinh,
    arithmetic.TensorArccosh,
    arithmetic.TensorArctanh,

    arithmetic.TensorLshift,
    arithmetic.TensorRshift,

    arithmetic.TensorTreeAdd,
    arithmetic.TensorTreeMultiply,

    reduction.TensorSum,
    reduction.TensorProd,
    reduction.TensorMax,
    reduction.TensorMin
)


def check_reduction_axis(node):
    return len(node.op.axis) == 1 or len(node.op.axis) == node.ndim


class Composer:
    def __init__(self, graph, engine):
        self.engine = engine
        self.explored = set()
        self.graph = graph
        self.keys = []

    def _jax_compat(self, op):
        if hasattr(op, 'execute_jax'):
            try:
                op.execute_jax()
            except NotImplementedError:
                return False
            return True
        return False

    def _can_skip(self, node):
        op = node.op

        if self.engine == 'numexpr':
            if not isinstance(op, NE_SUPPORT_OP) or node.key in self.keys:
                return True
            if node in self.explored or isinstance(op, NE_REDUCTION_OP):
                return True
            if self.graph.count_successors(node) != 1:
                return True

        if self.engine == 'jax':
            if not self._jax_compat(node.op) or node.key in self.keys:
                return True
            if node in self.explored:
                return True
            if self.graph.count_successors(node) != 1:
                return True

        return False

    def _can_break(self, node):
        if self.engine == 'numexpr':
            if self.graph.count_successors(node) != 1 or isinstance(node.op, NE_REDUCTION_OP):
                return True
        if self.engine == 'jax':
            if not self._jax_compat(node.op):
                return True

            if self.graph.count_successors(node) != 1:
                return True
        return False

    def _support(self, node):
        op_type = type(node.op)

        if self.engine == 'numexpr':
            if op_type in NE_REDUCTION_OP:
                return check_reduction_axis(node)
            return op_type in NE_SUPPORT_OP
        if self.engine == 'jax':
            if self._jax_compat(node.op):
                return True
            return False

    def _get_fused_chunk(self, tail_node):
        if self.engine == 'numexpr':
            from ..tensor.fuse import TensorNeFuseChunk
            return TensorNeFuseChunk(dtype=tail_node.dtype)

        if self.engine == 'jax':
            from ..tensor.fuse import TensorJaxFuseChunk
            return TensorJaxFuseChunk(dtype=tail_node.dtype)

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
