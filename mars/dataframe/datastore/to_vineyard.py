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

from collections import defaultdict
import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...config import options
from ...context import get_context, RunningMode
from ...serialize import TupleField, ListField, StringField
from ...tiles import TilesError
from ...utils import check_chunks_unknown_shape
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType

try:
    import vineyard
except ImportError:
    vineyard = None


class DataFrameToVineyardChunk(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_STORE_VINEYARD_CHUNK

    # vineyard ipc socket
    _vineyard_socket = StringField('vineyard_socket')

    def __init__(self, vineyard_socket=None, dtypes=None, **kw):
        super().__init__(_vineyard_socket=vineyard_socket, _dtypes=dtypes, _object_type=ObjectType.dataframe, **kw)

    def __call__(self, df):
        return self.new_dataframe([df], shape=(0, 0), dtypes=df.dtypes,
                                  index_value=df.index_value, columns_value=df.columns_value)

    @property
    def vineyard_socket(self):
        return self._vineyard_socket

    @classmethod
    def _get_out_chunk(cls, op, in_chunk):
        chunk_op = op.copy().reset_key()
        out_chunk_shape = (np.nan,) * in_chunk.ndim
        return chunk_op.new_chunk([in_chunk], shape=out_chunk_shape,
                                  index=in_chunk.index)

    @classmethod
    def tile(cls, op):
        # Not truely necessary, but putting a barrier here will make things simple
        check_chunks_unknown_shape(op.inputs, TilesError)

        in_df = op.inputs[0]

        out_chunks = []
        for chunk in in_df.chunks:
            chunk_op = op.copy().reset_key()
            chunk = chunk_op.new_chunk([chunk], shape=chunk.shape, dtypes=chunk.dtypes,
                                      index_value=chunk.index_value,
                                      columns_value=chunk.columns_value,
                                      index=chunk.index)
            out_chunks.append(chunk)

        new_op = op.copy().reset_key()
        return new_op.new_dataframes(op.inputs, shape=op.inputs[0].shape,
                                     dtypes=in_df.dtypes,
                                     index_value=in_df.index_value,
                                     columns_value=in_df.columns_value,
                                     chunks=out_chunks, nsplits=((np.nan,),))

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError('vineyard is not available')
        client = vineyard.connect(op.vineyard_socket)

        df = vineyard.DataFrameBuilder(client)
        for name, value in ctx[op.inputs[0].key].iteritems():
            np_value = value.to_numpy(copy=False)
            if not np_value.flags['C_CONTIGUOUS']:
                np_value = np.ascontiguousarray(np_value)
            df.add(str(name), vineyard.TensorBuilder.from_numpy(client, np_value))
        df.chunk_index = op.inputs[0].index
        df = df.build(client)
        df.persist(client)

        # store the result object id to execution context
        ctx[op.outputs[0].key] = pd.DataFrame([(client.instance_id, df.id)])


class DataFrameToVineyardChunkMap(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_STORE_VINEYARD_CHUNK_MAP

    _local_chunks = ListField("local_chunks")

    # vineyard ipc socket
    _vineyard_socket = StringField('vineyard_socket')

    def __init__(self, vineyard_socket=None, **kw):
        super().__init__(_vineyard_socket=vineyard_socket, _object_type=ObjectType.dataframe, **kw)

    @property
    def local_chunks(self):
        return self._local_chunks

    @property
    def vineyard_socket(self):
        return self._vineyard_socket

    def __call__(self, df):
        return self.new_dataframe([df], shape=df.shape, dtypes=df.dtypes,
                                  index_value=df.index_value, columns_value=df.columns_value)

    @classmethod
    def _process_out_chunks(cls, op, out_chunks):
        merge_op = DataFrameToVinyardStoreGlobalMeta(
                vineyard_socket=op.vineyard_socket,
                chunk_shape=op.inputs[0].chunk_shape,
                shape=op.inputs[0].shape,
                dtypes=op.inputs[0].dtypes)
        return merge_op.new_chunks(out_chunks, shape=out_chunks[0].shape,
                                   index=(0,) * out_chunks[0].ndim)

    @classmethod
    def tile(cls, op):
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

        all_chunk_ids = ctx.get_chunk_results([c.key for c in op.inputs[0].chunks])
        all_chunk_ids = pd.concat(all_chunk_ids, axis='index')
        chunk_map = defaultdict(list)
        for instance_id, chunk_id in all_chunk_ids.itertuples(index=False):
            chunk_map[instance_id].append(chunk_id)

        out_chunks = []
        for idx, (instance_id, local_chunks) in enumerate(chunk_map.items()):
            chunk_op = op.copy().reset_key()
            chunk_op._local_chunks = local_chunks
            chunk_op._expect_worker = workers[instance_id]
            out_chunks.append(chunk_op.new_chunk(op.inputs[0].chunks, shape=(1,), index=(idx,)))
        out_chunks = cls._process_out_chunks(op, out_chunks)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, shape=(len(out_chunks),),
                                     dtypes=op.inputs[0].dtypes,
                                     index_value=op.inputs[0].index_value,
                                     columns_value=op.inputs[0].columns_value,
                                     chunks=out_chunks,
                                     nsplits=((1,) * len(out_chunks),))

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError('vineyard is not available')
        client = vineyard.connect(op.vineyard_socket)
        chunk_blob = vineyard.UInt64VectorBuilder(client, op.local_chunks).build(client)
        ctx[op.outputs[0].key] = (client.instance_id, chunk_blob.id)


class DataFrameToVinyardStoreGlobalMeta(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_STORE_VINEYARD_GLOBAL_META

    _shape = TupleField('shape')
    _chunk_shape = TupleField('chunk_shape')

    # vineyard ipc socket
    _vineyard_socket = StringField('vineyard_socket')

    def __init__(self, vineyard_socket=None, chunk_shape=None, shape=None, **kw):
        super().__init__(_vineyard_socket=vineyard_socket,
                         _chunk_shape=chunk_shape, _shape=shape,
                         _object_type=ObjectType.dataframe, **kw)

    @property
    def shape(self):
        return self._shape

    @property
    def chunk_shape(self):
        return self._chunk_shape

    @property
    def vineyard_socket(self):
        return self._vineyard_socket

    @classmethod
    def _process_out_chunks(cls, op, out_chunks):
        if len(out_chunks) == 1:
            return out_chunks
        else:
            raise NotImplementedError('not implemented')

    @classmethod
    def tile(cls, op):
        return [super().tile(op)[0]]

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError('vineyard is not available')
        client = vineyard.connect(op.vineyard_socket)

        df = vineyard.GlobalDataFrameBuilder(client)
        for in_chunk in op.inputs:
            instance_id, chunk_blob_id = ctx[in_chunk.key]
            df.add_chunks(instance_id, chunk_blob_id)
        df.chunk_shape = op.chunk_shape
        df = df.build(client)
        df.persist(client)

        # store the result object id to execution context
        ctx[op.outputs[0].key] = vineyard.ObjectID.stringify(df.id)


def to_vineyard(df, vineyard_socket=None):
    if vineyard_socket is None:
        vineyard_socket = options.vineyard.socket

    chunk_op = DataFrameToVineyardChunk(vineyard_socket=vineyard_socket)
    chunk_map_op = DataFrameToVineyardChunkMap(vineyard_socket=vineyard_socket)
    return chunk_map_op(chunk_op(df))
