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

import enum

from .fetch import PushShuffle
from ... import opcodes
from ...serialization.serializables import (
    Int32Field,
    StringField,
    TupleField,
    FieldTypes,
)
from .base import Operand, VirtualOperand, OperandStage


class ShuffleProxy(VirtualOperand):
    _op_type_ = opcodes.SHUFFLE_PROXY


class ShuffleType(enum.Enum):
    # pull-based shuffle will wait all mappers finish before scheduling reducers.
    PULL = 0
    # push-based shuffle will push mapper data to reducers ASAP, so the shuffle execution
    # will get better performance by IO, pipeline and locality improvement.
    PUSH = 1


class MapReduceOperand(Operand):
    """
    An operand for shuffle execution which partitions data by the value in each record’s partition key, and
    send the partitioned data from all mappers to all reducers.
    """

    reducer_index = TupleField("reducer_index", FieldTypes.uint64)
    mapper_id = Int32Field("mapper_id", default=0)
    reducer_phase = StringField("reducer_phase", default=None)

    def _new_chunks(self, inputs, kws=None, **kw):
        if getattr(self, "reducer_index", None) is None:
            index = None
            if kws:
                index = kws[0].get("index")
            self.reducer_index = index or kw.get("index")

        return super()._new_chunks(inputs, kws, **kw)

    def get_dependent_data_keys(self):
        from .fetch import FetchShuffle

        if self.stage == OperandStage.reduce:
            inputs = self.inputs or ()
            deps = []
            for inp in inputs:
                if isinstance(inp.op, ShuffleProxy):
                    deps.extend(
                        [(chunk.key, self.reducer_index) for chunk in inp.inputs or ()]
                    )
                elif isinstance(inp.op, FetchShuffle):
                    deps.extend([(k, self.reducer_index) for k in inp.op.source_keys])
                else:
                    # push-based shuffle doesn't store data keys.
                    assert not isinstance(inp.op, PushShuffle)
                    deps.append(inp.key)
            return deps
        return super().get_dependent_data_keys()

    def _iter_mapper_key_idx_pairs(self, input_id=0, mapper_id=None):
        # TODO(chaokunyang) add mapper key, mapper index, reducer index document.
        # key is mapper chunk key, index is mapper chunk index.
        input_chunk = self.inputs[input_id]
        if isinstance(input_chunk.op, PushShuffle):
            # For push shuffle, since data are directly pushed to reducers, chunk key and index are not needed any more.
            # Before reducer executing, all shuffle block of same reducers will be pushed to same reducers, identified
            # by their index, so we use mapper index as key_idx_pairs.
            # For operands such as PSRS(PSRSSortRegularSample -> PSRSShuffle -> PSRSAlign)
            return ((i, i) for i in range(input_chunk.op.num_mappers))
        if isinstance(input_chunk.op, ShuffleProxy):
            keys = [inp.key for inp in input_chunk.inputs]
            idxes = [inp.index for inp in input_chunk.inputs]
            mappers = (
                [inp.op.mapper_id for inp in input_chunk.inputs]
                if mapper_id is not None
                else None
            )
        else:
            keys = input_chunk.op.source_keys
            idxes = input_chunk.op.source_idxes
            mappers = input_chunk.op.source_mappers

        if mapper_id is None:
            key_idx_pairs = zip(keys, idxes)
        else:
            key_idx_pairs = (
                (key, idx)
                for key, idx, mapper in zip(keys, idxes, mappers)
                if mapper == mapper_id
            )
        return key_idx_pairs

    def iter_mapper_data_with_index(
        self, ctx, input_id=0, mapper_id=None, pop=False, skip_none=False
    ):
        for key, idx in self._iter_mapper_key_idx_pairs(input_id, mapper_id):
            try:
                if pop:
                    yield idx, ctx.pop((key, self.reducer_index))
                else:
                    yield idx, ctx[key, self.reducer_index]
            except KeyError:
                if not skip_none:  # pragma: no cover
                    raise
                if not pop:
                    ctx[key, self.reducer_index] = None

    def iter_mapper_data(
        self, ctx, input_id=0, mapper_id=0, pop=False, skip_none=False
    ):
        for _idx, data in self.iter_mapper_data_with_index(
            ctx, input_id, mapper_id, pop=pop, skip_none=skip_none
        ):
            yield data

    def execute(self, ctx, op):
        """The mapper stage must ensure all mapper blocks are inserted into ctx in the order of reducers and no blocks
        for some reducers are missing. This is needed by push-based shuffle, which shuffle block are identified by the
        index instead of data keys.
        For operands implementation simplicity, we can sort the `ctx` by key which are (chunk key, reducer index) tuple
        and relax the insert order requirements.
        """
