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
import pandas as pd

from ... import opcodes as OperandDef
from ...core import OutputType
from ...config import options
from ...serialize import StringField
from ...context import get_context, RunningMode
from ...utils import calc_nsplits
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index

try:
    import vineyard
    from vineyard.data.utils import normalize_dtype
except ImportError:
    vineyard = None


class DataFrameFromVineyard(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_FROM_VINEYARD

    # vineyard ipc socket
    _vineyard_socket = StringField('vineyard_socket')

    # ObjectID in vineyard
    _object_id = StringField('object_id')

    def __init__(self, vineyard_socket=None, object_id=None, **kw):
        super().__init__(_vineyard_socket=vineyard_socket, _object_id=object_id,
                         _output_types=[OutputType.dataframe], **kw)

    @property
    def vineyard_socket(self):
        return self._vineyard_socket

    @property
    def object_id(self):
        return self._object_id

    def check_inputs(self, inputs):
        # no inputs
        if inputs and len(inputs) > 0:
            raise ValueError("DataFrame data source has no inputs")

    def _new_chunks(self, inputs, kws=None, **kw):
        shape = kw.get('shape', None)
        self.extra_params['shape'] = shape  # set shape to make the operand key different
        return super()._new_chunks(inputs, kws=kws, **kw)

    def _new_tileables(self, inputs, kws=None, **kw):
        shape = kw.get('shape', None)
        self.extra_params['shape'] = shape  # set shape to make the operand key different
        return super()._new_tileables(inputs, kws=kws, **kw)

    def __call__(self, shape, chunk_size=None, dtypes=None, index_value=None, columns_value=None):
        return self.new_dataframe(None, shape, dtypes=dtypes,
                                  index_value=index_value,
                                  columns_value=columns_value)

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

        df_meta = client.get_meta(vineyard.ObjectID(op.object_id))

        chunk_map = {}
        df_columns, df_dtypes = [], []
        for idx in range(int(df_meta['partitions_-size'])):
            chunk_meta = df_meta['partitions_-%d' % idx]
            chunk_location = int(chunk_meta['instance_id'])
            columns = json.loads(chunk_meta['columns_'])
            shape = (np.nan, len(columns))
            if not columns:
                # note that in vineyard dataframe are splitted along the index axis.
                df_columns = columns
            if not df_dtypes:
                for column_idx in range(len(columns)):
                    column_meta = chunk_meta['__values_-value-%d' % column_idx]
                    dtype = normalize_dtype(column_meta['value_type_'],
                                            column_meta.get('value_type_meta_', None))
                    df_dtypes.append(dtype)
            chunk_index = (int(chunk_meta['partition_index_row_']), int(chunk_meta['partition_index_column_']))
            chunk_map[chunk_index] = (chunk_location, chunk_meta['id'], shape, columns)

        nsplits = calc_nsplits({chunk_index: shape
                                for chunk_index, (_, _, shape, _) in chunk_map.items()})

        out_chunks = []
        for chunk_index, (instance_id, chunk_id, shape, columns) in chunk_map.items():
            chunk_op = op.copy().reset_key()
            chunk_op._object_id = chunk_id
            chunk_op._expect_worker = workers[instance_id]
            out_chunks.append(chunk_op.new_chunk([], shape=shape, index=chunk_index,
                              # use the same value as `read_csv`
                              index_value=parse_index(pd.RangeIndex(0, -1)),
                              columns_value=parse_index(pd.Index(columns))))

        new_op = op.copy()
        # n.b.: the `shape` will be filled by `_update_tileable_and_chunk_shape`.
        return new_op.new_dataframes(op.inputs, shape=(np.nan, np.nan), dtypes=df_dtypes,
                                     chunks=out_chunks, nsplits=nsplits,
                                     # use the same value as `read_csv`
                                     index_value=parse_index(pd.RangeIndex(0, -1)),
                                     columns_value=parse_index(pd.Index(df_columns)))

    @classmethod
    def execute(cls, ctx, op):
        if vineyard is None:
            raise RuntimeError('vineyard is not available')
        client = vineyard.connect(op.vineyard_socket)

        # setup resolver context
        from vineyard.core import default_builder_context, default_resolver_context
        from vineyard.data.dataframe import register_dataframe_types
        from vineyard.data.tensor import register_tensor_types
        register_dataframe_types(builder_ctx=default_builder_context,
                                 resolver_ctx=default_resolver_context)
        register_tensor_types(builder_ctx=default_builder_context,
                              resolver_ctx=default_resolver_context)

        # chunk has a dataframe chunk
        ctx[op.outputs[0].key] = client.get(vineyard.ObjectID(op.object_id))


def from_vineyard(df, vineyard_socket=None):
    if vineyard_socket is None:
        vineyard_socket = options.vineyard.socket

    if vineyard is not None and isinstance(df, vineyard.Object):
        if 'vineyard::GlobalDataFrame' not in df.typename:
            raise TypeError('The input dataframe %r is not a vineyard\' GlobalDataFrame' % df)
        object_id = repr(df.id)
    else:
        object_id = df

    op = DataFrameFromVineyard(vineyard_socket=vineyard_socket, object_id=object_id)
    return op(shape=(np.nan,), chunk_size=(np.nan,), dtypes=pd.Series([]),
              index_value=parse_index(pd.Index([])),
              columns_value=parse_index(pd.Index([])))
