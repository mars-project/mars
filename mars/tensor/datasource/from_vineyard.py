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
from ...serialize import StringField
from ...context import get_context, RunningMode
from ...utils import calc_nsplits
from .core import TensorNoInput

try:
    import vineyard
    from vineyard.data.utils import normalize_dtype
except ImportError:
    vineyard = None


# Note [Tensor from vineyard]
#
#   tiling using the metadata, to generate real chunks with `expect_worker`


class TensorFromVineyard(TensorNoInput):
    _op_type_ = OperandDef.TENSOR_FROM_VINEYARD

    # vineyard ipc socket
    _vineyard_socket = StringField('vineyard_socket')

    # ObjectID in vineyard
    _object_id = StringField('object_id')

    def __init__(self, vineyard_socket=None, object_id=None,
                 dtype=None, gpu=None, sparse=None, **kw):
        super().__init__(_vineyard_socket=vineyard_socket, _object_id=object_id,
                         _dtype=dtype, _gpu=gpu, _sparse=sparse, **kw)

    @property
    def vineyard_socket(self):
        return self._vineyard_socket

    @property
    def object_id(self):
        return self._object_id

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

        tensor_meta = client.get_meta(vineyard.ObjectID(op.object_id))

        chunk_map = {}
        dtype = None
        for idx in range(int(tensor_meta['partitions_-size'])):
            chunk_meta = tensor_meta['partitions_-%d' % idx]
            if dtype is None:
                dtype = normalize_dtype(chunk_meta['value_type_'],
                                        chunk_meta.get('value_type_meta_', None))
            chunk_location = int(chunk_meta['instance_id'])
            shape = tuple(json.loads(chunk_meta['shape_']))
            chunk_index = tuple(json.loads(chunk_meta['partition_index_']))
            chunk_map[chunk_index] = (chunk_location, chunk_meta['id'], shape)

        nsplits = calc_nsplits({chunk_index: shape
                                for chunk_index, (_, _, shape) in chunk_map.items()})

        out_chunks = []
        for chunk_index, (instance_id, chunk_id, shape) in chunk_map.items():
            chunk_op = op.copy().reset_key()
            chunk_op._object_id = chunk_id
            chunk_op._expect_worker = workers[instance_id]
            out_chunks.append(chunk_op.new_chunk([], shape=shape, index=chunk_index))

        new_op = op.copy()
        return new_op.new_tileables(op.inputs, dtype=dtype, chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError('vineyard is not available')
        client = vineyard.connect(op.vineyard_socket)

        # setup resolver context
        from vineyard.data.tensor import tensor_resolver

        # chunk has a tensor chunk
        ctx[op.outputs[0].key] = client.get(vineyard.ObjectID(op.object_id), tensor_resolver)


def from_vineyard(tensor, vineyard_socket=None):
    if vineyard_socket is None:
        vineyard_socket = options.vineyard.socket

    if vineyard is not None and isinstance(tensor, vineyard.Object):
        if 'vineyard::GlobalTensor' not in tensor.typename:
            raise TypeError('The input tensor %r is not a vineyard\' GlobalTensor' % tensor)
        object_id = repr(tensor.id)
    else:
        object_id = tensor
    op = TensorFromVineyard(vineyard_socket=vineyard_socket, object_id=object_id,
                            dtype=np.dtype('byte'), gpu=False)
    return op(shape=(np.nan,), chunk_size=(np.nan,))
