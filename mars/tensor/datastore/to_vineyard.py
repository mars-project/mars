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

from typing import Tuple
import numpy as np

from ... import opcodes as OperandDef
from ...core.operand.base import SchedulingHint
from ...serialization.serializables import FieldTypes, KeyField, StringField, TupleField
from ...storage.base import StorageLevel
from ..datasource import tensor as astensor
from .core import TensorDataStore

try:
    import vineyard
    from vineyard.data.tensor import make_global_tensor
    from vineyard.data.utils import to_json
except ImportError:
    vineyard = None


def resolve_vineyard_socket(ctx, op) -> Tuple[str, bool]:
    storage_backend = ctx.get_storage_info(level=StorageLevel.MEMORY)
    if storage_backend.get("name", None) == "vineyard":  # pragma: no cover
        if (
            op.vineyard_socket is not None
            and op.vineyard_socket != storage_backend["socket"]
        ):
            return op.vineyard_socket, True
        else:
            return storage_backend["socket"], False
    else:
        return op.vineyard_socket, True


class TensorVineyardDataStoreChunk(TensorDataStore):
    _op_type_ = OperandDef.TENSOR_STORE_VINEYARD_CHUNK

    _input = KeyField("input")

    # vineyard ipc socket
    vineyard_socket = StringField("vineyard_socket")

    # a dummy attr to make sure ops have different keys
    operator_index = TupleField("operator_index", FieldTypes.int32)

    def __init__(self, vineyard_socket=None, **kw):
        super().__init__(vineyard_socket=vineyard_socket, **kw)

    @classmethod
    def _process_out_chunks(cls, op, out_chunks):
        merge_op = TensorVineyardDataStoreMeta(
            vineyard_socket=op.vineyard_socket, sparse=op.sparse, dtype=np.dtype("O")
        )
        return merge_op.new_chunks(
            out_chunks, shape=(1,), dtype=np.dtype("O"), index=(0,) * out_chunks[0].ndim
        )

    @classmethod
    def tile(cls, op):
        out_chunks = []
        scheduling_hint = SchedulingHint(fuseable=False)
        for idx, chunk in enumerate(op.inputs[0].chunks):
            chunk_op = op.copy().reset_key()
            chunk_op.scheduling_hint = scheduling_hint
            chunk_op.operator_index = chunk.index
            out_chunk = chunk_op.new_chunk(
                [chunk], dtype=np.dtype("O"), shape=(1,), index=(idx,)
            )
            out_chunks.append(out_chunk)
        out_chunks = cls._process_out_chunks(op, out_chunks)

        new_op = op.copy().reset_key()
        return new_op.new_tensors(
            op.inputs,
            shape=(len(out_chunks),),
            dtype=np.dtype("O"),
            chunks=out_chunks,
            nsplits=((1,),),
        )

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError("vineyard is not available")
        socket, needs_put = resolve_vineyard_socket(ctx, op)
        client = vineyard.connect(socket)

        # some op might be fused and executed twice on different workers
        if not needs_put:
            # might be fused
            try:  # pragma: no cover
                meta = ctx.get_chunks_meta([op.inputs[0].key])[0]
                tensor_id = vineyard.ObjectID(meta["object_ref"])
                if not client.exists(tensor_id):
                    needs_put = True
            except KeyError:
                needs_put = True
        if needs_put:
            tensor_id = client.put(
                np.asarray(ctx[op.inputs[0].key]), partition_index=op.inputs[0].index
            )
        else:  # pragma: no cover
            meta = client.get_meta(tensor_id)
            new_meta = vineyard.ObjectMeta()
            for k, v in meta.items():
                if k not in ["id", "signature", "instance_id"]:
                    if isinstance(v, vineyard.ObjectMeta):
                        new_meta.add_member(k, v)
                    else:
                        new_meta[k] = v
            new_meta["partition_index_"] = to_json(op.inputs[0].index)
            tensor_id = client.create_metadata(new_meta).id

        client.persist(tensor_id)
        holder = np.empty((1,), dtype=object)
        holder[0] = tensor_id
        ctx[op.outputs[0].key] = holder


class TensorVineyardDataStoreMeta(TensorDataStore):
    _op_type_ = OperandDef.TENSOR_STORE_VINEYARD_META

    _input = KeyField("input")

    # vineyard ipc socket
    vineyard_socket = StringField("vineyard_socket")

    def __init__(self, vineyard_socket=None, dtype=None, sparse=None, **kw):
        super().__init__(
            vineyard_socket=vineyard_socket, dtype=dtype, sparse=sparse, **kw
        )

    @classmethod
    def tile(cls, op):
        chunk_op = op.copy().reset_key()
        out_chunk = chunk_op.new_chunk(
            op.inputs[0].chunks, dtype=np.dtype("O"), shape=(1,), index=(0,)
        )
        new_op = op.copy().reset_key()
        return new_op.new_tensors(
            op.inputs,
            shape=(1,),
            dtype=np.dtype("O"),
            chunks=[out_chunk],
            nsplits=((1,),),
        )

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError("vineyard is not available")

        socket, _ = resolve_vineyard_socket(ctx, op)
        client = vineyard.connect(socket)

        # # store the result object id to execution context
        chunks = [ctx[chunk.key][0] for chunk in op.inputs]
        holder = np.empty((1,), dtype=object)
        holder[0] = make_global_tensor(client, chunks).id
        ctx[op.outputs[0].key] = holder


def tovineyard(x, vineyard_socket=None):
    x = astensor(x)
    op = TensorVineyardDataStoreChunk(
        vineyard_socket=vineyard_socket, dtype=x.dtype, sparse=x.issparse()
    )
    return op(x)
