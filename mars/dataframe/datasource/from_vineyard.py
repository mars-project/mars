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
from ...core.context import get_context
from ...serialization.serializables import Int32Field, StringField
from ...tensor.datasource.from_vineyard import resolve_vineyard_socket
from ...utils import calc_nsplits, has_unknown_shape
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index


try:
    import vineyard
    from vineyard.data.utils import normalize_dtype, from_json
except ImportError:
    vineyard = None


class DataFrameFromVineyard(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_FROM_VINEYARD_CHUNK

    # generated columns for metadata
    generated_columns = ["id", "worker_address", "dtypes", "shape", "index", "columns"]

    # vineyard ipc socket
    vineyard_socket = StringField("vineyard_socket")

    # ObjectID in vineyard
    object_id = StringField("object_id")

    # a dummy attr to make sure ops have different keys
    operator_index = Int32Field("operator_index")

    def __init__(self, vineyard_socket=None, object_id=None, **kw):
        super().__init__(
            vineyard_socket=vineyard_socket,
            object_id=object_id,
            _output_types=[OutputType.dataframe],
            **kw
        )

    def check_inputs(self, inputs):
        # no inputs
        if inputs and len(inputs) > 0:
            raise ValueError("DataFrame data source has no inputs")

    def _new_chunks(self, inputs, kws=None, **kw):
        shape = kw.get("shape", None)
        self.extra_params[
            "shape"
        ] = shape  # set shape to make the operand key different
        return super()._new_chunks(inputs, kws=kws, **kw)

    def _new_tileables(self, inputs, kws=None, **kw):
        shape = kw.get("shape", None)
        self.extra_params[
            "shape"
        ] = shape  # set shape to make the operand key different
        return super()._new_tileables(inputs, kws=kws, **kw)

    def __call__(self, shape, dtypes=None, index_value=None, columns_value=None):
        return self.new_dataframe(
            None,
            shape,
            dtypes=dtypes,
            index_value=index_value,
            columns_value=columns_value,
        )

    @classmethod
    def tile(cls, op):
        ctx = get_context()
        workers = ctx.get_worker_addresses()

        out_chunks = []
        dtypes = pd.Series(
            [np.dtype("O")] * len(cls.generated_columns), index=cls.generated_columns
        )
        for index, worker in enumerate(workers):
            chunk_op = op.copy().reset_key()
            chunk_op.expect_worker = worker
            chunk_op.operator_index = index
            out_chunk = chunk_op.new_chunk(
                [],
                dtypes=dtypes,
                shape=(1, len(cls.generated_columns)),
                index=(index, 0),
                index_value=parse_index(pd.RangeIndex(0, 1)),
                columns_value=parse_index(pd.Index(cls.generated_columns)),
            )
            out_chunks.append(out_chunk)

        new_op = op.copy().reset_key()
        return new_op.new_dataframes(
            op.inputs,
            shape=(np.nan, np.nan),
            dtypes=dtypes,
            chunks=out_chunks,
            nsplits=((np.nan,), (np.nan,)),
            # use the same value as `read_csv`
            index_value=parse_index(pd.RangeIndex(0, 1)),
            columns_value=parse_index(pd.Index(cls.generated_columns)),
        )

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError("vineyard is not available")

        socket = resolve_vineyard_socket(ctx, op)
        client = vineyard.connect(socket)

        meta = client.get_meta(vineyard.ObjectID(op.object_id))
        chunks, dtypes = [], None
        for idx in range(meta["partitions_-size"]):
            chunk_meta = meta["partitions_-%d" % idx]
            columns = pd.Index(from_json(chunk_meta["columns_"]))
            shape = (np.nan, len(columns))
            if not chunk_meta.islocal:
                continue
            if dtypes is None:
                dtypes = []
                for idx in range(len(columns)):
                    column_meta = chunk_meta["__values_-value-%d" % idx]
                    dtype = normalize_dtype(
                        column_meta["value_type_"],
                        column_meta.get("value_type_meta_", None),
                    )
                    dtypes.append(dtype)
                dtypes = pd.Series(dtypes, index=columns)
            chunk_index = (
                chunk_meta["partition_index_row_"],
                chunk_meta["partition_index_column_"],
            )
            # chunk: (chunk_id, worker_address, dtype, shape, index, columns)
            chunks.append(
                (
                    repr(chunk_meta.id),
                    ctx.worker_address,
                    dtypes,
                    shape,
                    chunk_index,
                    columns,
                )
            )

        ctx[op.outputs[0].key] = pd.DataFrame(chunks, columns=cls.generated_columns)


class DataFrameFromVineyardChunk(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.TENSOR_FROM_VINEYARD_CHUNK

    # vineyard ipc socket
    vineyard_socket = StringField("vineyard_socket")

    # ObjectID of chunk in vineyard
    object_id = StringField("object_id")

    def __init__(self, vineyard_socket=None, object_id=None, **kw):
        super().__init__(vineyard_socket=vineyard_socket, object_id=object_id, **kw)

    def __call__(self, meta):
        return self.new_dataframe([meta])

    @classmethod
    def tile(cls, op):
        if has_unknown_shape(*op.inputs):
            yield

        ctx = get_context()

        in_chunk_keys = [chunk.key for chunk in op.inputs[0].chunks]
        out_chunks = []
        chunk_map = dict()
        dtypes, columns = None, None
        for chunk, infos in zip(
            op.inputs[0].chunks, ctx.get_chunks_result(in_chunk_keys)
        ):
            for _, info in infos.iterrows():
                chunk_op = op.copy().reset_key()
                chunk_op.object_id = info["id"]
                chunk_op.expect_worker = info["worker_address"]
                dtypes = info["dtypes"]
                columns = info["columns"]
                shape = info["shape"]
                chunk_index = info["index"]
                chunk_map[chunk_index] = info["shape"]
                out_chunk = chunk_op.new_chunk(
                    [chunk],
                    shape=shape,
                    index=chunk_index,
                    dtypes=dtypes,
                    index_value=parse_index(pd.RangeIndex(0, -1)),
                    columns_value=parse_index(pd.Index(columns)),
                )
                out_chunks.append(out_chunk)

        nsplits = calc_nsplits(chunk_map)
        shape = [np.sum(nsplit) for nsplit in nsplits]
        new_op = op.copy().reset_key()
        return new_op.new_dataframes(
            op.inputs,
            shape=shape,
            dtypes=dtypes,
            chunks=out_chunks,
            nsplits=nsplits,
            index_value=parse_index(pd.RangeIndex(0, -1)),
            columns_value=parse_index(pd.Index(columns)),
        )

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError("vineyard is not available")

        socket = resolve_vineyard_socket(ctx, op)
        client = vineyard.connect(socket)

        client = vineyard.connect(socket)
        ctx[op.outputs[0].key] = client.get(vineyard.ObjectID(op.object_id))


def from_vineyard(df, vineyard_socket=None):
    if vineyard is not None and isinstance(df, vineyard.Object):  # pragma: no cover
        if "vineyard::GlobalDataFrame" not in df.typename:
            raise TypeError(
                "The input dataframe %r is not a vineyard' GlobalDataFrame" % df
            )
        object_id = df.id
    else:
        object_id = df
    if vineyard is not None and isinstance(object_id, vineyard.ObjectID):
        object_id = repr(object_id)
    metaop = DataFrameFromVineyard(
        vineyard_socket=vineyard_socket,
        object_id=object_id,
        dtype=np.dtype("byte"),
        gpu=None,
    )
    meta = metaop(
        shape=(np.nan,),
        dtypes=pd.Series([]),
        index_value=parse_index(pd.Index([])),
        columns_value=parse_index(pd.Index([])),
    )
    op = DataFrameFromVineyardChunk(
        vineyard_socket=vineyard_socket, object_id=object_id, gpu=None
    )
    return op(meta)
