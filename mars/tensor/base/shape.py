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

import numpy as np

from ... import opcodes
from ...core import ExecutableTuple
from ...serialize import KeyField, Int32Field
from ...utils import calc_nsplits
from ..datasource import tensor as astensor
from ..operands import TensorOperand, TensorOperandMixin, TensorOrder


class TensorGetShape(TensorOperand, TensorOperandMixin):
    _op_type_ = opcodes.GET_SHAPE

    _a = KeyField('a')
    _ndim = Int32Field('ndim')

    def __init__(self, prepare_inputs=None, a=None, ndim=None, dtype=None, **kw):
        super().__init__(_dtype=dtype, _a=a, _ndim=ndim,
                         _prepare_inputs=prepare_inputs, **kw)

    @property
    def a(self):
        return self._a

    @property
    def ndim(self):
        return self._ndim

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self._a is not None:
            self._a = self._inputs[0]

    @property
    def output_limit(self):
        return self._ndim

    def __call__(self, a):
        if not np.isnan(a.size):
            return ExecutableTuple([astensor(s) for s in a.shape])

        self._a = a
        kws = []
        for i in range(self.output_limit):
            kws.append({
                'shape': (),
                'dtype': np.dtype(np.intc),
                'order': TensorOrder.C_ORDER,
                'i': i
            })
        return ExecutableTuple(self.new_tensors([a], kws=kws))

    @classmethod
    def tile(cls, op):
        a = op.a
        outs = op.outputs

        chunk_op = TensorGetShape(prepare_inputs=[False] * len(a.chunks),
                                  ndim=op.ndim)
        chunk_kws = []
        for out in outs:
            params = out.params
            params['index'] = ()
            chunk_kws.append(params)
        chunks = chunk_op.new_chunks(a.chunks, kws=chunk_kws)

        kws = []
        for c, out in zip(chunks, outs):
            params = out.params
            params['chunks'] = [c]
            params['nsplits'] = ()
            kws.append(params)
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, kws=kws,
                                  output_limit=op.output_limit)

    @classmethod
    def execute(cls, ctx, op):
        chunk_idx_tochunk_shapes = \
            {c.index: cm.chunk_shape for c, cm
             in zip(op.inputs, ctx.get_chunk_metas([c.key for c in op.inputs]))}
        nsplits = calc_nsplits(chunk_idx_tochunk_shapes)
        shape = tuple(sum(ns) for ns in nsplits)
        for o, s in zip(op.outputs, shape):
            ctx[o.key] = s


def shape(a):
    """
    Return the shape of a tensor.

    Parameters
    ----------
    a : array_like
        Input tensor.

    Returns
    -------
    shape : ExecutableTuple of tensors
        The elements of the shape tuple give the lengths of the
        corresponding array dimensions.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.shape(mt.eye(3)).execute()
    (3, 3)
    >>> mt.shape([[1, 2]]).execute()
    (1, 2)
    >>> mt.shape([0]).execute()
    (1,)
    >>> mt.shape(0).execute()
    ()

    >>> a = mt.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
    >>> mt.shape(a).execute()
    (2,)

    """
    a = astensor(a)
    op = TensorGetShape(ndim=a.ndim)
    return op(a)
