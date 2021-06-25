# Copyright 1999-2021 Alibaba Group Holding Ltd.
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
from ....core import ExecutableTuple, get_output_types, recursive_tile
from ....serialization.serializables import AnyField
from ....utils import has_unknown_shape
from ...operands import LearnOperand, LearnOperandMixin


class LGBMAlign(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.LGBM_ALIGN

    _data = AnyField('data')
    _label = AnyField('label')
    _sample_weight = AnyField('sample_weight')
    _init_score = AnyField('init_score')

    def __init__(self, data=None, label=None, sample_weight=None, init_score=None,
                 output_types=None, **kw):
        super().__init__(_data=data, _label=label, _sample_weight=sample_weight,
                         _init_score=init_score, _output_types=output_types, **kw)

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
    def init_score(self):
        return self._init_score

    @property
    def output_limit(self):
        return 2 if self._sample_weight is None else 3

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        it = iter(inputs)
        self._data = next(it)
        for attr in ('_label', '_sample_weight', '_init_score'):
            if getattr(self, attr) is not None:
                setattr(self, attr, next(it))

    def __call__(self):
        kws, inputs = [], []
        for arg in [self.data, self.label, self.sample_weight, self.init_score]:
            if hasattr(arg, 'params'):
                kws.append(arg.params)
                inputs.append(arg)
        tileables = self.new_tileables(inputs, kws=kws)
        return ExecutableTuple(tileables)

    @classmethod
    def tile(cls, op: "LGBMAlign"):
        inputs = [d for d in [op.data, op.label, op.sample_weight, op.init_score] if d is not None]
        data = op.data

        # check inputs to make sure no unknown chunk shape exists
        if has_unknown_shape(*inputs):
            yield

        if len(data.nsplits[1]) != 1:
            data = yield from recursive_tile(data.rechunk({1: data.shape[1]}))
        outputs = [data]
        for inp in inputs[1:]:
            if inp is not None:
                outputs.append(
                    (yield from recursive_tile(inp.rechunk((data.nsplits[0],)))))

        kws = []
        for o in outputs:
            kw = o.params.copy()
            kw.update(dict(chunks=o.chunks, nsplits=o.nsplits))
            kws.append(kw)

        new_op = op.copy().reset_key()
        tileables = new_op.new_tileables(inputs, kws=kws)

        return tileables


def align_data_set(dataset):
    out_types = get_output_types(dataset.data, dataset.label, dataset.sample_weight, dataset.init_score)
    op = LGBMAlign(data=dataset.data, label=dataset.label, sample_weight=dataset.sample_weight,
                   init_score=dataset.init_score, output_types=out_types)
    return op()
