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


from ... import opcodes as OperandDef
from ...serialize import KeyField, ListField
from ..operands import TensorHasInput, TensorOperandMixin
from ..array_utils import get_array_module
from ..core import TensorOrder


class TensorSlice(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.SLICE

    _input = KeyField('input')
    _slices = ListField('slices')

    def __init__(self, slices=None, dtype=None, sparse=False, **kw):
        super().__init__(_slices=slices, _dtype=dtype, _sparse=sparse, **kw)

    @property
    def slices(self):
        return self._slices

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def _get_order(self, kw, i):
        order = kw.pop('order', None)
        if order is None:
            inp = self.input
            if inp is None or inp.order == TensorOrder.C_ORDER:
                return TensorOrder.C_ORDER

            for shape, slc in zip(inp.shape, self._slices):
                if slc is None:
                    continue
                s = slc.indices(shape)
                if s[0] == 0 and s[1] == shape and s[2] == 1:
                    continue
                else:
                    return TensorOrder.C_ORDER

            return inp.order

        return order[i] if isinstance(order, (list, tuple)) else order

    @classmethod
    def execute(cls, ctx, op):
        inp = ctx[op.inputs[0].key]
        if op.input.ndim == 0 and not hasattr(inp, 'shape'):
            # scalar, but organize it into an array
            inp = get_array_module(inp).array(inp)
        x = inp[tuple(op.slices)]
        out = op.outputs[0]
        ctx[out.key] = x.astype(x.dtype, order=out.order.value, copy=False)
