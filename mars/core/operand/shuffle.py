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
from ...serialization.serializables import Int32Field, StringField
from .base import Operand, VirtualOperand, OperandStage


class ShuffleProxy(VirtualOperand):
    _op_type_ = opcodes.SHUFFLE_PROXY


class MapReduceOperand(Operand):
    shuffle_key = StringField('shuffle_key', default=None)
    map_input_id = Int32Field('map_input_id', default=0)

    def _new_chunks(self, inputs, kws=None, **kw):
        if getattr(self, 'shuffle_key', None) is None:
            index = None
            if kws:
                index = kws[0].get('index')
            index = index or kw.get('index')

            self.shuffle_key = ','.join(str(i) for i in index)

        return super()._new_chunks(inputs, kws, **kw)

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

    def _get_mapper_key_idx_pairs(self, sort_index: bool):
        input_chunk = self.inputs[0]
        if isinstance(input_chunk.op, ShuffleProxy):
            keys = [inp.key for inp in input_chunk.inputs]
            idxes = [inp.index for inp in input_chunk.inputs]
        else:
            keys = input_chunk.op.to_fetch_keys
            idxes = input_chunk.op.to_fetch_idxes
        key_idx_pairs = zip(keys, idxes)
        if sort_index:
            key_idx_pairs = sorted(key_idx_pairs)
        return key_idx_pairs

    def iter_mapper_data_with_index(self, ctx, sort_index=False, pop=False, skip_none=False):
        for key, idx in self._get_mapper_key_idx_pairs(sort_index):
            try:
                if pop:
                    yield idx, ctx.pop((key, self.shuffle_key))
                else:
                    yield idx, ctx[key, self.shuffle_key]
            except KeyError:
                if not skip_none:
                    raise
                if not pop:
                    ctx[key, self.shuffle_key] = None

    def iter_mapper_data(self, ctx, sort_index=False, pop=False, skip_none=False):
        for _idx, data in self.iter_mapper_data_with_index(
                ctx, sort_index=sort_index, pop=pop, skip_none=skip_none):
            yield data
