import importlib.util

from ...serialize import DataTypeField
from ..operands import TensorFuse
from .core import TensorFuseChunkMixin, estimate_fuse_size

spec = importlib.util.find_spec('jax')
if spec is None:
    JAX_INSTALLED = False
else:
    JAX_INSTALLED = False


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
        # execute the fuse operands in jax
        if JAX_INSTALLED:
            for op in op.operands:
                op.execute_jax(ctx, op)
        else:
            for op in op.operands:
                op.execute(ctx, op)

    @classmethod
    def estimate_size(cls, ctx, op):
        estimate_fuse_size(ctx, op)
