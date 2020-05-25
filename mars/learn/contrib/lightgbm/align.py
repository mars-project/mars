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

from .... import opcodes
from ....context import get_context, RunningMode
from ....core import ExecutableTuple
from ....serialize import AnyField
from ...operands import LearnOperand, LearnOperandMixin
from ...utils import get_output_types


class LGBMAlign(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.LGBM_ALIGN

    _data = AnyField('data')
    _label = AnyField('label')
    _sample_weight = AnyField('sample_weight')

    def __init__(self, data=None, label=None, sample_weight=None,
                 output_types=None, **kw):
        super().__init__(_data=data, _label=label, _sample_weight=sample_weight,
                         _output_types=output_types, **kw)

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def sample_weight(self):
        return self._sample_weight

    @property
    def output_limit(self):
        return 2 if self._sample_weight is None else 3

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        it = iter(inputs)
        self._data = next(it)
        self._label = next(it)
        if self._sample_weight is not None:
            self._sample_weight = next(it)

    def __call__(self, data, label, sample_weight=None):
        kws = [data.params, label.params]
        inputs = [data, label]
        if hasattr(sample_weight, 'params'):
            kws.append(sample_weight.params)
            inputs.append(sample_weight)
        tileables = self.new_tileables(inputs, kws=kws)
        return ExecutableTuple(tileables)

    @classmethod
    def tile(cls, op: "LGBMAlign"):
        data = op.data
        label = op.label
        sample_weight = op.sample_weight

        ctx = get_context()
        if ctx.running_mode != RunningMode.distributed:
            data = data.rechunk(tuple((s,) for s in data.shape))._inplace_tile()
            label = label.rechunk(tuple((s,) for s in label.shape))._inplace_tile()
            if sample_weight is not None:
                sample_weight = sample_weight.rechunk(tuple((s,) for s in sample_weight.shape))._inplace_tile()
        else:
            if len(data.nsplits[1]) != 1:
                data = data.rechunk({1: data.shape[1]})._inplace_tile()
            label = label.rechunk((data.nsplits[0],))._inplace_tile()
            if sample_weight is not None:
                sample_weight = sample_weight.rechunk((data.nsplits[0],))._inplace_tile()

        outputs = [data, label]
        if sample_weight is not None:
            outputs.append(sample_weight)

        kws = [data.params, label.params]
        kws[0]['chunks'] = data.chunks
        kws[0]['nsplits'] = data.nsplits
        kws[1]['chunks'] = label.chunks
        kws[1]['nsplits'] = label.nsplits

        inputs = [data, label]
        if hasattr(sample_weight, 'params'):
            kws.append(sample_weight.params)
            inputs.append(sample_weight)
            kws[-1]['chunks'] = sample_weight.chunks
            kws[-1]['nsplits'] = sample_weight.nsplits

        new_op = op.copy().reset_key()
        tileables = new_op.new_tileables(op.inputs, kws=kws)

        return tileables


def align_inputs(data, label, sample_weight=None):
    op = LGBMAlign(data=data, label=label, sample_weight=sample_weight,
                   output_types=get_output_types(data, label, sample_weight))
    return op(data, label, sample_weight)
