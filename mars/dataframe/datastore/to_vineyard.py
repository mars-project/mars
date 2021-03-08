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
from ...context import RunningMode, get_context
from ...core import OutputType
from ...serialize import TupleField, StringField
from ...tiles import TilesError
from ...utils import check_chunks_unknown_shape
from ..operands import DataFrameOperand, DataFrameOperandMixin

try:
    import vineyard
except ImportError:
    vineyard = None


class DataFrameToVineyardChunk(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_STORE_VINEYARD_CHUNK

    # vineyard ipc socket
    _vineyard_socket = StringField('vineyard_socket')

    # vineyard object id
    _vineyard_object_id = StringField('vineyard_object_id')

    def __init__(self, vineyard_socket=None, dtypes=None, **kw):
        super().__init__(_vineyard_socket=vineyard_socket, _dtypes=dtypes,
                         _output_types=[OutputType.dataframe], **kw)

    def __call__(self, df):
        return self.new_dataframe([df], shape=(0, 0), dtypes=df.dtypes,
                                  index_value=df.index_value, columns_value=df.columns_value)

    @property
    def vineyard_socket(self):
        return self._vineyard_socket

    @property
    def vineyard_object_id(self):
        return self._vineyard_object_id

    @classmethod
    def _get_out_chunk(cls, op, in_chunk):
        chunk_op = op.copy().reset_key()
        out_chunk_shape = (np.nan,) * in_chunk.ndim
        return chunk_op.new_chunk([in_chunk], shape=out_chunk_shape,
                                  index=in_chunk.index)

    @classmethod
    def _process_out_chunks(cls, op, out_chunks):
        merge_op = DataFrameToVinyardStoreGlobalMeta(
                vineyard_socket=op.vineyard_socket,
                chunk_shape=op.inputs[0].chunk_shape,
                shape=op.inputs[0].shape,
                dtypes=op.inputs[0].dtypes)
        return merge_op.new_chunks(out_chunks, shape=(1,),
                                   index=(0,) * out_chunks[0].ndim)

    @classmethod
    def tile(cls, op):
        # Not truely necessary, but putting a barrier here will make things simple
        check_chunks_unknown_shape(op.inputs, TilesError)

        in_df = op.inputs[0]

        ctx = get_context()

        out_chunks = []
        for chunk in in_df.chunks:
            chunk_op = op.copy().reset_key()
            if options.vineyard.enabled and ctx.running_mode != RunningMode.local:
                object_id = ctx.get_vineyard_object_id(chunk.key)
                if object_id is not None:
                    chunk_op._vineyard_object_id = repr(object_id)
                else:
                    chunk_op._vineyard_object_id = ''
            else:
                chunk_op._vineyard_object_id = ''
            chunk = chunk_op.new_chunk([chunk], shape=chunk.shape, dtypes=chunk.dtypes,
                                       index_value=chunk.index_value,
                                       columns_value=chunk.columns_value,
                                       index=chunk.index)
            out_chunks.append(chunk)

        out_chunks = cls._process_out_chunks(op, out_chunks)

        new_op = op.copy().reset_key()
        return new_op.new_dataframes(op.inputs, shape=op.inputs[0].shape,
                                     dtypes=in_df.dtypes,
                                     index_value=in_df.index_value,
                                     columns_value=in_df.columns_value,
                                     chunks=out_chunks, nsplits=((np.prod(op.inputs[0].chunk_shape),),))

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError('vineyard is not available')
        client = vineyard.connect(op.vineyard_socket)

        # setup builder context
        from vineyard.core import default_builder_context, default_resolver_context
        from vineyard.data.dataframe import register_dataframe_types
        from vineyard.data.tensor import register_tensor_types
        register_dataframe_types(builder_ctx=default_builder_context,
                                 resolver_ctx=default_resolver_context)
        register_tensor_types(builder_ctx=default_builder_context,
                              resolver_ctx=default_resolver_context)

        if options.vineyard.enabled and op.vineyard_object_id:
            # the chunk already exists in vineyard
            df_id = vineyard.ObjectID(op.vineyard_object_id)
        else:
            df_id = client.put(ctx[op.inputs[0].key], partition_index=op.inputs[0].index)

        client.persist(df_id)

        # store the result object id to execution context
        ctx[op.outputs[0].key] = (client.instance_id, repr(df_id))


class DataFrameToVinyardStoreGlobalMeta(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_STORE_VINEYARD_GLOBAL_META

    _shape = TupleField('shape')
    _chunk_shape = TupleField('chunk_shape')

    # vineyard ipc socket
    _vineyard_socket = StringField('vineyard_socket')

    def __init__(self, vineyard_socket=None, chunk_shape=None, shape=None, **kw):
        super().__init__(_vineyard_socket=vineyard_socket,
                         _chunk_shape=chunk_shape, _shape=shape,
                         _output_types=[OutputType.dataframe], **kw)

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
        meta.set_global(True)
        meta['typename'] = 'vineyard::GlobalDataFrame'
        meta['partition_shape_row_'] = op.shape[0]
        meta['partition_shape_column_'] = op.shape[1]

        for idx, in_chunk in enumerate(op.inputs):
            _, chunk_id = ctx[in_chunk.key]
            meta.add_member('partitions_-%d' % idx, vineyard.ObjectID(chunk_id))
        meta['partitions_-size'] = len(op.inputs)

        global_dataframe_id = client.create_metadata(meta)
        client.persist(global_dataframe_id)

        # # store the result object id to execution context
        ctx[op.outputs[0].key] = repr(global_dataframe_id)


def to_vineyard(df, vineyard_socket=None):
    if vineyard_socket is None:
        vineyard_socket = options.vineyard.socket

    op = DataFrameToVineyardChunk(vineyard_socket=vineyard_socket)
    return op(df)
