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

import cloudpickle

from .... import opcodes as OperandDef
from ....serialize import KeyField, StringField, BytesField, Int8Field
from ...operands import TensorOperand, TensorOperandMixin
from ...array_utils import as_same_device, cp, device


class TensorCdist(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.CDIST

    _xa = KeyField('XA')
    _xb = KeyField('XB')
    _metric = StringField('metric')
    _metric_func = BytesField('metric_func', on_serialize=cloudpickle.dumps,
                              on_deserialize=cloudpickle.loads)
    _p = Int8Field('p')
    _w = KeyField('w')
    _v = KeyField('V')
    _vi = KeyField('VI')
    _out = KeyField('out')

    def __init__(self, metric=None, metric_func=None, p=None, w=None,
                 v=None, vi=None, out=None, dtype=None, **kw):
        super().__init__(_metric=metric, _metric_func=metric_func, _p=p,
                         _w=w, _v=v, _vi=vi, _out=out, _dtype=dtype, **kw)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        self._xa = next(inputs_iter)
        self._xb = next(inputs_iter)
        if self._w is not None:
            self._w = next(inputs_iter)
        if self._v is not None:
            self._v = next(inputs_iter)
        if self._vi is not None:
            self._vi = next(inputs_iter)
        if self._out is not None:
            self._out = next(inputs_iter)

    @property
    def xa(self):
        return self._xa

    @property
    def xb(self):
        return self._xb

    @property
    def metric(self):
        return self._metric

    @property
    def metric_func(self):
        return self._metric_func

    @property
    def p(self):
        return self._p

    @property
    def w(self):
        return self._w

    @property
    def v(self):
        return self._v

    @property
    def vi(self):
        return self._vi

    @property
    def out(self):
        return self._out

    @classmethod
    def execute(cls, ctx, op):
        from scipy.spatial.distance import cdist

        inputs, device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)

        if xp is cp:
            raise NotImplementedError('`cdist` does not support running on GPU yet')

        with device(device_id):
            inputs_iter = iter(inputs)
            xa = next(inputs_iter)
            xb = next(inputs_iter)
            kw = dict()
            if op.p is not None:
                kw['p'] = op.p
            if op.w is not None:
                kw['w'] = next(inputs_iter)
            if op.v is not None:
                kw['V'] = next(inputs_iter)
            if op.vi is not None:
                kw['VI'] = next(inputs_iter)
            if op.out is not None:
                kw['out'] = next(inputs_iter)

        metric = op.metric if op.metric is not None else op.metric_func
        ctx[op.outputs[0].key] = cdist(xa, xb, metric=metric, **kw)
