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
from ...config import options
from ...serialize import KeyField, StringField
from ...context import get_context, RunningMode
from ...tiles import TilesError
from ...utils import calc_nsplits, check_chunks_unknown_shape
from .core import TensorNoInput, TensorHasInput

try:
    import vineyard
except ImportError:
    vineyard = None


# Note [Tensor from vineyard]
#
#   step 1: generate a chunk every worker to accumulate all chunks id
#   step 2: run iterative tiling, to generate real chunks with `expect_worker`


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

        tensor = client.get(op.object_id)

        out_chunks, idx = [], 0
        for instance_id in client.instances:
            chunk_op = op.copy().reset_key()
            # some instances may not have chunks
            try:
                chunk_blob_id = vineyard.ObjectID.stringify(tensor.local_chunks(instance_id))
            except RuntimeError:
                chunk_blob_id = None
            if chunk_blob_id:
                chunk_op._object_id = chunk_blob_id
                chunk_op._expect_worker = workers[instance_id]
                out_chunks.append(chunk_op.new_chunk(None, shape=(np.nan,), index=(idx,)))
                idx += 1

        new_op = op.copy()
        return new_op.new_tileables(None, shape=(np.nan,), chunks=out_chunks,
                                    nsplits=((np.nan,) * len(out_chunks),))

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError('vineyard is not available')
        client = vineyard.connect(op.vineyard_socket)

        # object ids vector as np.ndarray
        ctx[op.outputs[0].key] = np.array(client.get(op.object_id), copy=False)


class TensorFromVineyardChunk(TensorHasInput):
    _op_type_ = OperandDef.TENSOR_FROM_VINEYARD_CHUNK

    # vineyard ipc socket
    _vineyard_socket = StringField('vineyard_socket')

    # ObjectID in vineyard
    _input = KeyField('_input')
    _object_id = StringField('object_id')

    def __init__(self, vineyard_socket=None, dtype=None, gpu=None, sparse=None, **kw):
        super().__init__(_vineyard_socket=vineyard_socket, _object_id=None,
                         _dtype=dtype, _gpu=gpu, _sparse=sparse, **kw)

    @property
    def vineyard_socket(self):
        return self._vineyard_socket

    @property
    def object_id(self):
        return self._object_id

    def __call__(self, a, order=None):
        if not isinstance(a.op, TensorFromVineyard):
            raise ValueError('Not a vineyard tensor')
        self._object_id = a.op.object_id
        return super().__call__(a, order=order)

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

        tensor = client.get(op.object_id)
        available_instances = []
        for instance_id in client.instances:
            try:
                tensor.local_chunks(instance_id)
                available_instances.append(instance_id)
            except RuntimeError:
                pass

        all_chunk_ids = ctx.get_chunk_results([c.key for c in op.input.chunks])

        chunk_map = {}
        for instance_id, chunk_ids, in_chunk in \
                zip(available_instances, all_chunk_ids, op.input.chunks):
            for chunk_id in chunk_ids:
                meta = client.get_meta(chunk_id)  # FIXME use batch get meta
                chunk_index = tuple(int(x) for x in meta['chunk_index'].split(' '))
                shape = tuple(int(x) for x in meta['shape'].split(' '))
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
        return new_op.new_tileables(op.inputs, chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError('vineyard is not available')
        client = vineyard.connect(op.vineyard_socket)

        # chunk has a tensor chunk
        ctx[op.outputs[0].key] = client.get(op.object_id).numpy()


def from_vineyard(tensor, vineyard_socket=None):
    if vineyard_socket is None:
        vineyard_socket = options.vineyard.socket

    if vineyard is not None and isinstance(tensor, vineyard.GlobalObject):
        object_id = vineyard.ObjectID.stringify(tensor.id)
    else:
        object_id = tensor
    gather_op = TensorFromVineyard(vineyard_socket=vineyard_socket, object_id=object_id,
                                   dtype=np.dtype('byte'), gpu=False)
    op = TensorFromVineyardChunk(vineyard_socket=vineyard_socket)
    return op(gather_op(shape=(np.nan,), chunk_size=(np.nan,)))
