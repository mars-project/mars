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

from ... import opcodes as OperandDef
from ...serialize import KeyField, StringField
from ...operands import OperandStage
from ...tensor.core import TensorOrder, CHUNK_TYPE as TENSOR_CHUNK_TYPE
from ...tensor.array_utils import as_same_device, device, issparse
from ...config import options
from ...utils import ceildiv
from ..operands import LearnOperand, LearnOperandMixin, OutputType
from .core import get_output_types


class CheckBase(LearnOperand, LearnOperandMixin):
    _input = KeyField('input')
    _value = KeyField('value')
    _err_msg = StringField('err_msg')

    def __init__(self, input=None, value=None, err_msg=None, stage=None,
                 gpu=None, output_types=None, **kw):
        super().__init__(_input=input, _value=value, _err_msg=err_msg,
                         _stage=stage, _output_types=output_types,
                         _gpu=gpu, **kw)

    @property
    def input(self):
        return self._input

    @property
    def value(self):
        return self._value

    @property
    def err_msg(self):
        return self._err_msg

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self._input is not None:
            self._input = self._inputs[0]
        if self._value is not None:
            self._value = self._inputs[-1]

    def __call__(self, x, value=None):
        # output input if value not specified
        self._value = value = value if value is not None else x
        self._output_types = get_output_types(value)
        self._stage = OperandStage.agg
        return self.new_tileable([x, value],
                                 kws=[value.params])

    @classmethod
    def tile(cls, op):
        combine_size = options.combine_size
        x, value = op.input, op.value
        check_chunks = []
        for i, chunk in enumerate(x.chunks):
            chunk_op = cls(err_msg=op.err_msg, stage=OperandStage.map,
                           output_types=[OutputType.tensor])
            check_chunk = chunk_op.new_chunk([chunk], shape=(),
                                             index=(i,),
                                             dtype=np.dtype(bool),
                                             order=TensorOrder.C_ORDER)
            check_chunks.append(check_chunk)

        while len(check_chunks) > 1:
            prev_check_chunks = check_chunks
            check_chunks = []
            chunk_size = ceildiv(len(prev_check_chunks), combine_size)
            for i in range(chunk_size):
                chunks = prev_check_chunks[i * combine_size: (i + 1) * combine_size]
                chunk_op = cls(err_msg=op.err_msg, stage=OperandStage.combine,
                               output_types=[OutputType.tensor])
                check_chunk = chunk_op.new_chunk(chunks, shape=(),
                                                 index=(i,),
                                                 dtype=np.dtype(bool),
                                                 order=TensorOrder.C_ORDER)
                check_chunks.append(check_chunk)

        check_chunk = check_chunks[0]
        out_chunks = []
        for val_chunk in value.chunks:
            chunk_op = cls(value=val_chunk, err_msg=op.err_msg, stage=OperandStage.agg,
                           output_types=op.output_types)
            out_chunk = chunk_op.new_chunk([check_chunk, val_chunk], kws=[val_chunk.params])
            out_chunks.append(out_chunk)

        new_op = op.copy()
        kw = op.outputs[0].params
        kw['chunks'] = out_chunks
        kw['nsplits'] = value.nsplits
        return new_op.new_tileables(op.inputs, kws=[kw])


class CheckNonNegative(CheckBase):
    _op_type_ = OperandDef.CHECK_NON_NEGATIVE

    _whom = StringField('whom')

    def __init__(self, input=None, value=None, whom=None, err_msg=None,
                 stage=None, gpu=None, output_types=None, **kw):
        super().__init__(input=input, value=value, _whom=whom,
                         err_msg=err_msg, stage=stage,
                         output_types=output_types,
                         gpu=gpu, **kw)
        if self._err_msg is None and self._whom is not None:
            self._err_msg = "Negative values in data passed to %s" % self._whom

    @property
    def whom(self):
        return self._whom

    @classmethod
    def _execute_tensor(cls, ctx, op):
        (x,), device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            if issparse(x) and x.nnz == 0:
                x_min = 0
            else:
                x_min = xp.min(x)

            if x_min < 0:
                raise ValueError(op.err_msg)

            ctx[op.outputs[0].key] = np.array(True)

    @classmethod
    def _execute_df(cls, ctx, op):
        x = ctx[op.inputs[0].key]
        x_min = x.min().min()
        if x_min < 0:
            raise ValueError(op.err_msg)

        ctx[op.outputs[0].key] = np.array(True)

    @classmethod
    def _execute_map(cls, ctx, op):
        if isinstance(op.inputs[0], TENSOR_CHUNK_TYPE):
            return cls._execute_tensor(ctx, op)
        else:
            return cls._execute_df(ctx, op)

    @classmethod
    def _execute_combine(cls, ctx, op):
        # just pass value cuz all inputs executed successfully
        ctx[op.outputs[0].key] = np.array(True)

    @classmethod
    def _execute_agg(cls, ctx, op):
        ctx[op.outputs[0].key] = ctx[op.value.key]

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            return cls._execute_map(ctx, op)
        elif op.stage == OperandStage.combine:
            return cls._execute_combine(ctx, op)
        else:
            assert op.stage == OperandStage.agg
            return cls._execute_agg(ctx, op)


def check_non_negative_then_return_value(to_check, value, whom):
    op = CheckNonNegative(input=to_check, value=value, whom=whom)
    return op(to_check, value)
