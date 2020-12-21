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

import json
import numpy as np

from ... import opcodes as OperandDef
from ...config import options
from ...serialize import TupleField, KeyField, StringField
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
#   step 3: build global tensor object.

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
        out_chunk_shape = (1,)
        return chunk_op.new_chunk([in_chunk], shape=out_chunk_shape,
                                  index=in_chunk.index)

    @classmethod
    def _process_out_chunks(cls, op, out_chunks):
        merge_op = TensorVineyardDataStoreGlobalMeta(
                vineyard_socket=op.vineyard_socket,
                chunk_shape=op.inputs[0].chunk_shape, shape=op.inputs[0].shape,
                sparse=op.sparse, dtype=op.inputs[0].dtype)
        return merge_op.new_chunks(out_chunks, shape=(1,),
                                   index=(0,) * out_chunks[0].ndim)

    @classmethod
    def tile(cls, op):
        # Not truely necessary, but putting a barrier here will make things simple
        check_chunks_unknown_shape(op.inputs, TilesError)

        out = super().tile(op)[0]
        new_op = op.copy().reset_key()
        return new_op.new_tensors(out.inputs, shape=op.input.shape, dtype=out.dtype,
                                  chunks=out.chunks, nsplits=((np.prod(op.input.chunk_shape),),))

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError('vineyard is not available')
        client = vineyard.connect(op.vineyard_socket)

        # setup builder context
        from vineyard.data.tensor import numpy_ndarray_builder

        np_value = ctx[op.input.key]
        if not np_value.flags['C_CONTIGUOUS']:
            np_value = np.ascontiguousarray(np_value)
        tensor_id = client.put(np_value, numpy_ndarray_builder, partition_index=op.input.index)
        client.persist(tensor_id)

        # store the result object id to execution context
        ctx[op.outputs[0].key] = (client.instance_id, repr(tensor_id))


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

        meta = vineyard.ObjectMeta()
        instances = set()
        chunks = set()
        for idx, in_chunk in enumerate(op.inputs):
            instance_id, chunk_id = ctx[in_chunk.key]
            instances.add(instance_id)
            chunks.add(chunk_id)
            meta.add_member('object_%d' % idx, vineyard.ObjectID(chunk_id))
        meta['typename'] = 'vineyard::ObjectSet'
        meta['num_of_instances'] = len(instances)
        meta['num_of_objects'] = len(chunks)
        object_set_id = client.create_metadata(meta)

        meta = vineyard.ObjectMeta()
        meta['typename'] = 'vineyard::GlobalTensor<%s>' % op.dtype.name
        meta['shape_'] = json.dumps(op.shape)
        meta['chunk_shape_'] = json.dumps(op.chunk_shape)
        meta.add_member('chunks_', object_set_id)
        global_tensor_id = client.create_metadata(meta)
        client.persist(global_tensor_id)

        # # store the result object id to execution context
        ctx[op.outputs[0].key] = repr(global_tensor_id)


def tovineyard(x, vineyard_socket=None):
    if vineyard_socket is None:
        vineyard_socket = options.vineyard.socket

    x = astensor(x)
    op = TensorVineyardDataStoreChunk(vineyard_socket=vineyard_socket,
                                      dtype=x.dtype, sparse=x.issparse())
    return op(x)
