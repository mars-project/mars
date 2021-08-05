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

import itertools
import socket

import numpy as np

from ... import opcodes
from ...core.operand import OperandStage
from ...serialization.serializables import FieldTypes, Int32Field, ListField, StringField
from ...tensor.merge import TensorConcatenate
from ...utils import get_next_port
from ..operands import LearnOperand, LearnOperandMixin, OutputType


class CollectPorts(LearnOperand, LearnOperandMixin):
    _op_code_ = opcodes.COLLECT_PORTS

    _socket_type = Int32Field('socket_type')
    _index = Int32Field('index')
    _workers = ListField('workers', FieldTypes.string)
    _tileable_key = StringField('tileable_key')

    def __init__(self, workers=None, socket_type=None, tileable_key=None, index=None, **kw):
        super().__init__(
            _socket_type=socket_type, _workers=workers, _tileable_key=tileable_key,
            _index=index, _pure_depends=[True], **kw)

    @property
    def socket_type(self):
        return self._socket_type

    @property
    def workers(self):
        return self._workers

    @property
    def tileable_key(self):
        return self._tileable_key

    def __call__(self, dep=None):
        self._output_types = [OutputType.tensor]
        if dep:
            deps = [dep]
        else:
            deps = None
        return self.new_tileable(
            deps, shape=(len(self.workers,),), dtype=np.dtype(int))

    @classmethod
    def tile(cls, op: "CollectPorts"):
        chunks = []
        if op.inputs:
            chunk_iter = itertools.cycle(op.inputs[0].chunks)
        else:
            chunk_iter = itertools.repeat(None)
        for idx, (worker, inp) in enumerate(zip(op.workers, chunk_iter)):
            new_op = op.copy().reset_key()
            new_op._workers = [worker]
            new_op.expect_worker = worker
            new_op.stage = OperandStage.map
            new_op._tileable_key = op.outputs[0].key
            new_op._index = idx
            new_op._pure_depends = [True]
            inps = [inp] if inp else None
            chunks.append(new_op.new_chunk(
                inps, index=(idx,), shape=(1,), dtype=np.dtype(int)))

        concat_op = TensorConcatenate(axis=0, dtype=chunks[0].dtype)
        concat_chunk = concat_op.new_chunk(
            chunks, shape=(len(op.workers),), index=(0,))

        new_op = op.copy().reset_key()
        params = op.outputs[0].params
        params.update(dict(chunks=[concat_chunk], nsplits=((len(op.workers),),)))
        return new_op.new_tileables(op.inputs, **params)

    @classmethod
    def execute(cls, ctx, op):
        assert ctx.band[0] == op.expect_worker
        socket_type = op.socket_type or socket.SOCK_STREAM
        port_num = get_next_port(socket_type, occupy=False)
        ctx[op.outputs[0].key] = np.array([port_num], dtype=int)


def collect_ports(workers, input_tileable=None):
    op = CollectPorts(workers=workers)
    return op(input_tileable)
