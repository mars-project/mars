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
import pandas as pd
from pandas.core.internals.blocks import Block
from pandas.core.internals.managers import BlockManager

from ... import opcodes as OperandDef
from ...core import OutputType
from ...config import options
from ...serialize import StringField
from ...context import get_context, RunningMode
from ...tiles import TilesError
from ...utils import calc_nsplits, check_chunks_unknown_shape
from ..operands import DataFrameOperand, DataFrameOperandMixin

try:
    import vineyard
except ImportError:
    vineyard = None


class DataFrameFromVineyard(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_FROM_VINEYARD

    # vineyard ipc socket
    _vineyard_socket = StringField('vineyard_socket')

    # ObjectID in vineyard
    _object_id = StringField('object_id')

    def __init__(self, vineyard_socket=None, object_id=None, **kw):
        super().__init__(_vineyard_socket=vineyard_socket, _object_id=object_id,
                         _output_types=[OutputType.dataframe], **kw)

    @property
    def vineyard_socket(self):
        return self._vineyard_socket

    @property
    def object_id(self):
        return self._object_id

    def check_inputs(self, inputs):
        # no inputs
        if inputs and len(inputs) > 0:
            raise ValueError("DataFrame data source has no inputs")

    def _new_chunks(self, inputs, kws=None, **kw):
        shape = kw.get('shape', None)
        self.extra_params['shape'] = shape  # set shape to make the operand key different
        return super()._new_chunks(inputs, kws=kws, **kw)

    def _new_tileables(self, inputs, kws=None, **kw):
        shape = kw.get('shape', None)
        self.extra_params['shape'] = shape  # set shape to make the operand key different
        return super()._new_tileables(inputs, kws=kws, **kw)

    def __call__(self, shape, chunk_size=None):
        return self.new_dataframe(None, shape, dtypes=[], raw_chunk_size=chunk_size,
                                  output_types=[OutputType.dataframe])

    @classmethod
    def tile(cls, op):
        if vineyard is None:
            raise RuntimeError('vineyard is not available')
        client = vineyard.connect(op.vineyard_socket)

        ctx = get_context()
        if ctx.running_mode == RunningMode.distributed:
            metas = ctx.get_worker_metas()
            workers = {meta['vineyard']['instance_id']: addr for addr, meta in metas.items()}
        else:
            workers = {client.instance_id: '127.0.0.1'}

        df = client.get(op.object_id)

        out_chunks, idx = [], 0
        for instance_id in client.instances:
            chunk_op = op.copy().reset_key()
            # some instances may not have chunks
            try:
                chunk_blob_id = vineyard.ObjectID.stringify(df.local_chunks(instance_id))
            except RuntimeError:
                chunk_blob_id = None
            if chunk_blob_id:
                chunk_op._object_id = chunk_blob_id
                chunk_op._expect_worker = workers[instance_id]
                out_chunks.append(chunk_op.new_chunk(None, shape=(np.nan,), dtypes=[], index=(idx,)))
                idx += 1

        new_op = op.copy()
        return new_op.new_dataframes(None, shape=(np.nan,), chunks=out_chunks,
                                     nsplits=((np.nan,) * len(out_chunks),))

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError('vineyard is not available')
        client = vineyard.connect(op.vineyard_socket)

        # object ids vector as np.ndarray
        ctx[op.outputs[0].key] = pd.DataFrame(np.array(client.get(op.object_id), copy=False))


class DataFrameFromVineyardChunk(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_FROM_VINEYARD_CHUNK

    # vineyard ipc socket
    _vineyard_socket = StringField('vineyard_socket')

    # ObjectID in vineyard
    _object_id = StringField('object_id')

    def __init__(self, vineyard_socket=None, **kw):
        super().__init__(_vineyard_socket=vineyard_socket, _object_id=None,
                         _output_types=[OutputType.dataframe], **kw)

    @property
    def vineyard_socket(self):
        return self._vineyard_socket

    @property
    def object_id(self):
        return self._object_id

    def __call__(self, a):
        if not isinstance(a.op, DataFrameFromVineyard):
            raise ValueError('Not a vineyard dataframe')
        self._object_id = a.op.object_id
        return self.new_dataframe([a], output_types=[OutputType.dataframe])

    @classmethod
    def tile(cls, op):
        # ensure the input op has been evaluated
        check_chunks_unknown_shape(op.inputs, TilesError)

        if vineyard is None:
            raise RuntimeError('vineyard is not available')
        client = vineyard.connect(op.vineyard_socket)

        ctx = get_context()
        if ctx.running_mode == RunningMode.distributed:
            metas = ctx.get_worker_metas()
            workers = {meta['vineyard']['instance_id']: addr for addr, meta in metas.items()}
        else:
            workers = {client.instance_id: '127.0.0.1'}

        df = client.get(op.object_id)
        available_instances = []
        for instance_id in client.instances:
            try:
                df.local_chunks(instance_id)
                available_instances.append(instance_id)
            except RuntimeError:
                pass

        all_chunk_ids = ctx.get_chunk_results([c.key for c in op.inputs[0].chunks])
        all_chunk_ids = pd.concat(all_chunk_ids, axis='columns')

        chunk_map = {}
        for instance_id, (_, chunk_ids), in_chunk in \
                zip(available_instances, all_chunk_ids.items(), op.inputs[0].chunks):
            for chunk_id in chunk_ids:
                meta = client.get_meta(chunk_id)  # FIXME use batch get meta
                chunk_index = (int(meta['chunk_index_row']), int(meta['chunk_index_column']))
                shape = (-1, int(meta['column_size']))
                chunk_map[chunk_index] = (instance_id, chunk_id, shape, in_chunk)

        nsplits = calc_nsplits({chunk_index: shape
                                for chunk_index, (_, _, shape, _) in chunk_map.items()})

        out_chunks = []
        for chunk_index, (instance_id, chunk_id, shape, in_chunk) in chunk_map.items():
            chunk_op = op.copy().reset_key()
            chunk_op._object_id = vineyard.ObjectID.stringify(chunk_id)
            chunk_op._expect_worker = workers[instance_id]
            out_chunks.append(chunk_op.new_chunk([in_chunk], shape=shape, index=chunk_index))

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, shape=(-1, -1), dtypes=[],
                                     chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError('vineyard is not available')
        client = vineyard.connect(op.vineyard_socket)

        # chunk has no tensor chunk
        df_chunk = client.get(op.object_id)
        if not df_chunk.columns:
            ctx[op.outputs[0].key] = pd.DataFrame()
        else:
            # ensure zero-copy
            blocks = []
            index_size = 0
            for idx, name in enumerate(df_chunk.columns):
                value = df_chunk[name].numpy()
                blocks.append(Block(np.expand_dims(value, 0), slice(idx, idx + 1, 1)))
                index_size = len(value)
            ctx[op.outputs[0].key] = pd.DataFrame(BlockManager(blocks, [df_chunk.columns, np.arange(index_size)]))


def from_vineyard(df, vineyard_socket=None):
    if vineyard_socket is None:
        vineyard_socket = options.vineyard.socket

    if vineyard is not None and isinstance(df, vineyard.GlobalObject):
        object_id = vineyard.ObjectID.stringify(df.id)
    else:
        object_id = df

    gather_op = DataFrameFromVineyard(vineyard_socket=vineyard_socket, object_id=object_id)
    op = DataFrameFromVineyardChunk(vineyard_socket=vineyard_socket)
    return op(gather_op(shape=(np.nan,), chunk_size=(np.nan,)))
