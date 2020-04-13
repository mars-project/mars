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

from ... import opcodes as OperandDef
from ...config import options
from ...context import get_context, RunningMode
from ...serialize import TupleField, ListField, KeyField, StringField
from ...tiles import TilesError
from ...utils import check_chunks_unknown_shape
from ..datasource import tensor as astensor
from .core import TensorDataStore

try:
    import vineyard
except ImportError:
    vineyard = None


# Note [Tensor to vineyard]
#
#   step 1: store local chunks
#   step 2: run iterative tiling, to gather, and build local blob of chunks id, with `expect_worker`
#   step 3: build chunk_map

class TensorVineyardDataStoreChunk(TensorDataStore):
    _op_type_ = OperandDef.TENSOR_STORE_VINEYARD_CHUNK

    _input = KeyField('input')

    # vineyard ipc socket
    _vineyard_socket = StringField('vineyard_socket')

    def __init__(self, vineyard_socket=None, dtype=None, sparse=None, **kw):
        super().__init__(_vineyard_socket=vineyard_socket, _dtype=dtype, _sparse=sparse, **kw)

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

        out = super().tile(op)[0]
        new_op = op.copy().reset_key()
        return new_op.new_tensors(out.inputs, shape=op.input.shape, dtype=out.dtype,
                                  chunks=out.chunks, nsplits=((np.nan,),))

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError('vineyard is not available')
        client = vineyard.connect(op.vineyard_socket)

        np_value = ctx[op.input.key]
        if not np_value.flags['C_CONTIGUOUS']:
            np_value = np.ascontiguousarray(np_value)
        tensor = vineyard.TensorBuilder.from_numpy(client, np_value)
        tensor.shape = op.input.shape
        tensor.chunk_index = op.input.index
        tensor = tensor.build(client)
        tensor.persist(client)

        # store the result object id to execution context
        ctx[op.outputs[0].key] = np.array([(client.instance_id, tensor.id)])


class TensorVineyardDataStoreChunkMap(TensorDataStore):
    _op_type_ = OperandDef.TENSOR_STORE_VINEYARD_CHUNK_MAP

    _input = KeyField('input')
    _local_chunks = ListField("local_chunks")

    # vineyard ipc socket
    _vineyard_socket = StringField('vineyard_socket')

    def __init__(self, vineyard_socket=None, dtype=None, sparse=None, **kw):
        super().__init__(_vineyard_socket=vineyard_socket, _dtype=dtype, _sparse=sparse, **kw)

    @property
    def local_chunks(self):
        return self._local_chunks

    @property
    def vineyard_socket(self):
        return self._vineyard_socket

    @classmethod
    def _process_out_chunks(cls, op, out_chunks):
        merge_op = TensorVineyardDataStoreGlobalMeta(
                vineyard_socket=op.vineyard_socket,
                chunk_shape=op.inputs[0].chunk_shape, shape=op.inputs[0].shape,
                sparse=op.sparse, dtype=op.inputs[0].dtype)
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

        all_chunk_ids = np.concatenate(ctx.get_chunk_results([c.key for c in op.input.chunks]))
        chunk_map = defaultdict(list)
        for instance_id, chunk_id in all_chunk_ids:
            chunk_map[instance_id].append(chunk_id)

        out_chunks = []
        for idx, (instance_id, local_chunks) in enumerate(chunk_map.items()):
            chunk_op = op.copy().reset_key()
            chunk_op._local_chunks = local_chunks
            chunk_op._expect_worker = workers[instance_id]
            out_chunks.append(chunk_op.new_chunk(op.input.chunks, shape=(1,), index=(idx,)))
        out_chunks = cls._process_out_chunks(op, out_chunks)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=(len(out_chunks),), chunks=out_chunks,
                                  nsplits=((1,) * len(out_chunks),))

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError('vineyard is not available')
        client = vineyard.connect(op.vineyard_socket)
        chunk_blob = vineyard.UInt64VectorBuilder(client, op.local_chunks).build(client)
        ctx[op.outputs[0].key] = (client.instance_id, chunk_blob.id)


class TensorVineyardDataStoreGlobalMeta(TensorDataStore):
    _op_type_ = OperandDef.TENSOR_STORE_VINEYARD_GLOBAL_META

    _input = KeyField('input')
    _shape = TupleField('shape')
    _chunk_shape = TupleField('chunk_shape')

    # vineyard ipc socket
    _vineyard_socket = StringField('vineyard_socket')

    def __init__(self, vineyard_socket=None,
                 chunk_shape=None, shape=None, dtype=None, sparse=None, **kw):
        super().__init__(_vineyard_socket=vineyard_socket,
                         _chunk_shape=chunk_shape, _shape=shape, _dtype=dtype, _sparse=sparse, **kw)

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

        tensor = vineyard.GlobalTensorBuilder(client)
        for in_chunk in op.inputs:
            instance_id, chunk_blob_id = ctx[in_chunk.key]
            tensor.add_chunks(instance_id, chunk_blob_id)
        tensor.shape = op.shape
        tensor.chunk_shape = op.chunk_shape
        tensor = tensor.build(client)
        tensor.persist(client)

        # store the result object id to execution context
        ctx[op.outputs[0].key] = vineyard.ObjectID.stringify(tensor.id)


def tovineyard(x, vineyard_socket=None):
    if vineyard_socket is None:
        vineyard_socket = options.vineyard.socket

    x = astensor(x)
    chunk_op = TensorVineyardDataStoreChunk(vineyard_socket=vineyard_socket,
                                            dtype=x.dtype, sparse=x.issparse())
    chunk_map_op = TensorVineyardDataStoreChunkMap(vineyard_socket=vineyard_socket,
                                                   dtype=x.dtype, sparse=x.issparse())
    return chunk_map_op(chunk_op(x))
