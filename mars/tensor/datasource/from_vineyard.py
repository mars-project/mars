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
from ...serialization.serializables import Int32Field, StringField
from ...storage.base import StorageLevel
from ...utils import calc_nsplits, has_unknown_shape
from ...core.context import get_context
from ..operands import TensorOperand, TensorOperandMixin
from .core import TensorNoInput

try:
    import vineyard
    from vineyard.data.utils import normalize_dtype
except ImportError:
    vineyard = None


def resolve_vineyard_socket(ctx, op):
    if op.vineyard_socket is None:  # pragma: no cover
        storage_backend = ctx.get_storage_info(level=StorageLevel.MEMORY)
        if storage_backend.get("name", None) == "vineyard":
            return storage_backend["socket"]
        else:
            return op.vineyard_socket
    else:
        return op.vineyard_socket


class TensorFromVineyard(TensorNoInput):
    _op_type_ = OperandDef.TENSOR_FROM_VINEYARD_META

    # vineyard ipc socket
    vineyard_socket = StringField("vineyard_socket")

    # ObjectID in vineyard
    object_id = StringField("object_id")

    # a dummy attr to make sure ops have different keys
    operator_index = Int32Field("operator_index")

    def __init__(self, vineyard_socket=None, object_id=None, **kw):
        super().__init__(vineyard_socket=vineyard_socket, object_id=object_id, **kw)

    @classmethod
    def tile(cls, op):
        ctx = get_context()
        workers = ctx.get_worker_addresses()

        out_chunks = []
        for index, worker in enumerate(workers):
            chunk_op = op.copy().reset_key()
            chunk_op.expect_worker = worker
            chunk_op.operator_index = index
            out_chunk = chunk_op.new_chunk(
                [], dtype=np.dtype(object), shape=(1,), index=(index,)
            )
            out_chunks.append(out_chunk)

        new_op = op.copy().reset_key()
        return new_op.new_tensors(
            op.inputs,
            shape=(np.nan,),
            dtype=np.dtype(object),
            chunks=out_chunks,
            nsplits=((np.nan,) * len(workers),),
        )

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError("vineyard is not available")

        socket = resolve_vineyard_socket(ctx, op)
        client = vineyard.connect(socket)

        meta = client.get_meta(vineyard.ObjectID(op.object_id))
        chunks = []
        for idx in range(meta["partitions_-size"]):
            chunk_meta = meta["partitions_-%d" % idx]
            if not chunk_meta.islocal:
                continue
            dtype = normalize_dtype(
                chunk_meta["value_type_"], chunk_meta.get("value_type_meta_", None)
            )
            shape = tuple(json.loads(chunk_meta["shape_"]))
            chunk_index = tuple(json.loads(chunk_meta["partition_index_"]))
            # chunk: (chunk_id, worker_address, dtype, shape, index)
            chunks.append(
                (repr(chunk_meta.id), ctx.worker_address, dtype, shape, chunk_index)
            )

        holder = np.empty((1,), dtype=object)
        holder[0] = chunks
        ctx[op.outputs[0].key] = np.asarray(holder)


class TensorFromVineyardChunk(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.TENSOR_FROM_VINEYARD_CHUNK

    # vineyard ipc socket
    vineyard_socket = StringField("vineyard_socket")

    # ObjectID of chunk in vineyard
    object_id = StringField("object_id")

    def __init__(self, vineyard_socket=None, object_id=None, **kw):
        super().__init__(vineyard_socket=vineyard_socket, object_id=object_id, **kw)

    def __call__(self, meta):
        return self.new_tensor([meta], shape=(np.nan,))

    @classmethod
    def tile(cls, op):
        if has_unknown_shape(*op.inputs):
            yield

        ctx = get_context()

        in_chunk_keys = [chunk.key for chunk in op.inputs[0].chunks]
        out_chunks = []
        chunk_map = dict()
        dtype = None
        for chunk, infos in zip(
            op.inputs[0].chunks, ctx.get_chunks_result(in_chunk_keys)
        ):
            for info in infos[0]:  # n.b. 1-element ndarray
                chunk_op = op.copy().reset_key()
                chunk_op.object_id = info[0]
                chunk_op.expect_worker = info[1]
                dtype = info[2]
                shape = info[3]
                chunk_index = info[4]
                chunk_map[chunk_index] = info[3]
                out_chunk = chunk_op.new_chunk(
                    [chunk], shape=shape, dtype=dtype, index=chunk_index
                )
                out_chunks.append(out_chunk)

        nsplits = calc_nsplits(chunk_map)
        shape = [np.sum(nsplit) for nsplit in nsplits]
        new_op = op.copy().reset_key()
        return new_op.new_tensors(
            op.inputs, shape=shape, dtype=dtype, chunks=out_chunks, nsplits=nsplits
        )

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError("vineyard is not available")

        socket = resolve_vineyard_socket(ctx, op)
        client = vineyard.connect(socket)

        client = vineyard.connect(socket)
        ctx[op.outputs[0].key] = client.get(vineyard.ObjectID(op.object_id))


def fromvineyard(tensor, vineyard_socket=None):
    if vineyard is not None and isinstance(tensor, vineyard.Object):  # pragma: no cover
        if "vineyard::GlobalTensor" not in tensor.typename:
            raise TypeError(
                "The input tensor %r is not a vineyard' GlobalTensor" % tensor
            )
        object_id = tensor.id
    else:
        object_id = tensor
    if vineyard is not None and isinstance(object_id, vineyard.ObjectID):
        object_id = repr(object_id)
    metaop = TensorFromVineyard(
        vineyard_socket=vineyard_socket,
        object_id=object_id,
        dtype=np.dtype("byte"),
        gpu=None,
    )
    meta = metaop(shape=(np.nan,), chunk_size=(np.nan,))
    op = TensorFromVineyardChunk(
        vineyard_socket=vineyard_socket, object_id=object_id, gpu=None
    )
    return op(meta)
