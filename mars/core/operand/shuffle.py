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

from ... import opcodes
from ...serialization.serializables import BoolField, StringField
from .base import Operand, VirtualOperand, OperandStage


class ShuffleProxy(VirtualOperand):
    _op_type_ = opcodes.SHUFFLE_PROXY

    _assign_reducers = BoolField('assign_reducers')

    def __init__(self, assign_reducers=True, **kw):
        super().__init__(_assign_reducers=assign_reducers, **kw)

    @property
    def assign_reducers(self) -> bool:
        return self._assign_reducers


class MapReduceOperand(Operand):
    shuffle_key = StringField('shuffle_key', default=None)

    def get_dependent_data_keys(self):
        from .fetch import FetchShuffle

        if self.stage == OperandStage.reduce:
            inputs = self.inputs or ()
            deps = []
            for inp in inputs:
                if isinstance(inp.op, ShuffleProxy):
                    deps.extend([(chunk.key, self.shuffle_key) for chunk in inp.inputs or ()])
                elif isinstance(inp.op, FetchShuffle):
                    deps.extend([(k, self.shuffle_key) for k in inp.op.to_fetch_keys])
                else:
                    deps.append(inp.key)
            return deps
        return super().get_dependent_data_keys()

