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

from ... import opcodes as OperandDef
from ...core import OutputType
from ...serialization.serializables import FieldTypes, StringField, TupleField
from ...tensor.datastore.to_vineyard import resolve_vineyard_socket
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index

try:
    import vineyard
    from vineyard.data.dataframe import make_global_dataframe
    from vineyard.data.utils import to_json
except ImportError:
    vineyard = None


class DataFrameToVineyardChunk(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_STORE_VINEYARD_CHUNK

    # vineyard ipc socket
    vineyard_socket = StringField("vineyard_socket")

    # a dummy attr to make sure ops have different keys
    operator_index = TupleField("operator_index", FieldTypes.int32)

    def __init__(self, vineyard_socket=None, dtypes=None, **kw):
        super().__init__(
            vineyard_socket=vineyard_socket,
            _dtypes=dtypes,
            _output_types=[OutputType.dataframe],
            **kw
        )

    def __call__(self, df):
        return self.new_dataframe(
            [df],
            shape=(0, 0),
            dtypes=df.dtypes,
            index_value=df.index_value,
            columns_value=df.columns_value,
        )

    @classmethod
    def _process_out_chunks(cls, op, out_chunks):
        dtypes = pd.Series([np.dtype("O")], index=pd.Index([0]))
        merge_op = DataFrameToVinyardStoreMeta(
            vineyard_socket=op.vineyard_socket,
            chunk_shape=op.inputs[0].chunk_shape,
            shape=(1, 1),
            dtypes=dtypes,
        )
        return merge_op.new_chunks(
            out_chunks, shape=(1, 1), dtypes=dtypes, index=(0, 0)
        )

    @classmethod
    def tile(cls, op):
        out_chunks = []
        dtypes = pd.Series([np.dtype("O")], index=pd.Index([0]))
        for idx, chunk in enumerate(op.inputs[0].chunks):
            chunk_op = op.copy().reset_key()
            chunk_op.operator_index = chunk.index
            out_chunk = chunk_op.new_chunk(
                [chunk],
                shape=(1, 1),
                dtypes=dtypes,
                index_value=chunk.index_value,
                columns_value=chunk.columns_value,
                index=(idx, 0),
            )
            out_chunks.append(out_chunk)
        out_chunks = cls._process_out_chunks(op, out_chunks)

        in_df = op.inputs[0]
        new_op = op.copy().reset_key()
        return new_op.new_dataframes(
            op.inputs,
            shape=(len(out_chunks), 1),
            dtypes=dtypes,
            index_value=in_df.index_value,
            columns_value=in_df.columns_value,
            chunks=out_chunks,
            nsplits=((np.prod(op.inputs[0].chunk_shape),),),
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
                df_id = vineyard.ObjectID(meta["object_ref"])
                if not client.exists(df_id):
                    needs_put = True
            except KeyError:
                needs_put = True
        if needs_put:
            df_id = client.put(
                ctx[op.inputs[0].key], partition_index=op.inputs[0].index
            )
        else:  # pragma: no cover
            meta = client.get_meta(df_id)
            new_meta = vineyard.ObjectMeta()
            for k, v in meta.items():
                if k not in ["id", "signature", "instance_id"]:
                    if isinstance(v, vineyard.ObjectMeta):
                        new_meta.add_member(k, v)
                    else:
                        new_meta[k] = v
            new_meta["partition_index_"] = to_json(op.inputs[0].index)
            df_id = client.create_metadata(new_meta).id

        client.persist(df_id)
        ctx[op.outputs[0].key] = pd.DataFrame({0: [df_id]})


class DataFrameToVinyardStoreMeta(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_STORE_VINEYARD_META

    # vineyard ipc socket
    vineyard_socket = StringField("vineyard_socket")

    def __init__(self, vineyard_socket=None, dtypes=None, **kw):
        super().__init__(
            vineyard_socket=vineyard_socket,
            dtypes=dtypes,
            _output_types=[OutputType.dataframe],
            **kw
        )

    @classmethod
    def tile(cls, op):
        dtypes = pd.Series([np.dtype("O")], index=pd.Index([0]))
        chunk_op = op.copy().reset_key()
        out_chunk = chunk_op.new_chunk(
            op.inputs[0].chunks,
            shape=(1, 1),
            dtypes=dtypes,
            index_value=parse_index(pd.Index([-1])),
            columns_value=parse_index(pd.Index([0])),
            index=(0, 0),
        )
        new_op = op.copy().reset_key()
        return new_op.new_dataframes(
            op.inputs,
            shape=(1, 1),
            dtypes=dtypes,
            index_value=parse_index(pd.Index([0])),
            columns_value=parse_index(pd.Index([0])),
            chunks=[out_chunk],
            nsplits=((1,), (1,)),
        )

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError("vineyard is not available")

        socket, _ = resolve_vineyard_socket(ctx, op)
        client = vineyard.connect(socket)

        # # store the result object id to execution context
        chunks = [ctx[chunk.key][0][0] for chunk in op.inputs]
        ctx[op.outputs[0].key] = pd.DataFrame(
            {0: [make_global_dataframe(client, chunks).id]}
        )


def to_vineyard(df, vineyard_socket=None):
    op = DataFrameToVineyardChunk(vineyard_socket=vineyard_socket)
    return op(df)
