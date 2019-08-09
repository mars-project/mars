import importlib.util

from ...serialize import DataTypeField
from ..operands import TensorFuse
from ..array_utils import as_same_device
from .core import TensorFuseChunkMixin, estimate_fuse_size

spec = importlib.util.find_spec('jax')
if spec is None:
    JAX_INSTALLED = False
else:
    JAX_INSTALLED = True


class TensorJaxFuseChunk(TensorFuse, TensorFuseChunkMixin):
    _op_type_ = None  # no opcode, cannot be serialized
    _dtype = DataTypeField('dtype')

    # use for jax-fused operand
    def __init__(self, dtype=None, **kw):
        super(TensorJaxFuseChunk, self).__init__(_dtype=dtype, **kw)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        inputs = as_same_device([ctx[c.key] for c in op.inputs], device=op.device)
        # execute the fuse operands in jax

        if JAX_INSTALLED:
            if len(inputs) == 1:
                inputs = inputs[0]
            for operand in op.operands:
                jax_function = operand.execute_jax()
                # binary operator
                from ..arithmetic.core import TensorBinOp
                import numpy as np
                if isinstance(operand, TensorBinOp):
                    # if there is scalar
                    other = None
                    if np.isscalar(operand.lhs):
                        other = operand.lhs
                    if np.isscalar(operand.rhs):
                        other = operand.rhs
                    if other is None:
                        inputs = jax_function(*inputs)
                    else:
                        inputs = jax_function(inputs, other)
                else:
                    inputs = jax_function(inputs)

            ctx[chunk.key] = inputs

    @classmethod
    def estimate_size(cls, ctx, op):
        estimate_fuse_size(ctx, op)
