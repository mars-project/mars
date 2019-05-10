# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

from .... import opcodes as OperandDef
from ....serialize import ValueType, DictField, TupleField, StringField, Int64Field, KeyField
from ..datasource import tensor as astensor
from .core import TensorDataStore
from .utils import check_tiledb_array_with_tensor, get_tiledb_schema_from_tensor


class TensorTileDBDataStore(TensorDataStore):
    _op_type_ = OperandDef.TENSOR_STORE_TILEDB

    _input = KeyField('input')
    _tiledb_config = DictField('tiledb_config')
    # URI of array to write
    _tiledb_uri = StringField('tiledb_uri')
    # encryption key to decrypt if provided
    _tiledb_key = StringField('tiledb_key')
    # open array at a given timestamp if provided
    _tiledb_timestamp = Int64Field('tiledb_timestamp')
    _axis_offsets = TupleField('axis_offsets', ValueType.int64)

    def __init__(self, tiledb_config=None, tiledb_uri=None, tiledb_key=None,
                 tiledb_timestamp=None, dtype=None, sparse=None, **kw):
        super(TensorTileDBDataStore, self).__init__(
            _tiledb_config=tiledb_config, _tiledb_uri=tiledb_uri, _tiledb_key=tiledb_key,
            _tiledb_timestamp=tiledb_timestamp, _dtype=dtype, _sparse=sparse, **kw)

    @property
    def tiledb_config(self):
        return self._tiledb_config

    @property
    def tiledb_uri(self):
        return self._tiledb_uri

    @property
    def tiledb_key(self):
        return self._tiledb_key

    @property
    def tiledb_timestamp(self):
        return self._tiledb_timestamp

    @property
    def axis_offsets(self):
        return self._axis_offsets

    @classmethod
    def _get_out_chunk(cls, op, in_chunk):
        chunk_op = op.copy().reset_key()
        nsplits = op.input.nsplits
        axis_offsets = []
        for axis, idx in enumerate(in_chunk.index):
            axis_offsets.append(sum(nsplits[axis][:idx]))
        chunk_op._axis_offsets = tuple(axis_offsets)
        out_chunk_shape = (0,) * in_chunk.ndim
        return chunk_op.new_chunk([in_chunk], shape=out_chunk_shape,
                                  index=in_chunk.index)

    @classmethod
    def tile(cls, op):
        import tiledb

        tensor = super(TensorTileDBDataStore, cls).tile(op)[0]

        ctx = tiledb.Ctx(op.tiledb_config)
        tiledb_array_type = tiledb.SparseArray if tensor.issparse() else tiledb.DenseArray
        try:
            tiledb_array_type(uri=op.tiledb_uri, key=op.tiledb_key,
                              timestamp=op.tiledb_timestamp, ctx=ctx)
        except tiledb.TileDBError:
            # not exist, try to create TileDB Array by given uri
            tiledb_array_schema = get_tiledb_schema_from_tensor(op.input, ctx, op.input.nsplits)
            tiledb_array_type.create(op.tiledb_uri, tiledb_array_schema, key=op.tiledb_key)

        return [tensor]


def totiledb(uri, x, ctx=None, key=None, timestamp=None):
    import tiledb

    x = astensor(x)
    raw_ctx = ctx
    if raw_ctx is None:
        ctx = tiledb.Ctx()

    tiledb_array_type = tiledb.SparseArray if x.issparse() else tiledb.DenseArray
    try:
        tiledb_array = tiledb_array_type(uri=uri, key=key, timestamp=timestamp, ctx=ctx)
        # if already created, we will check the shape and dtype
        check_tiledb_array_with_tensor(x, tiledb_array)
    except tiledb.TileDBError:
        # not exist, as we don't know the tile,
        # we will create the tiledb array in the tile of tensor
        pass

    tiledb_config = None if raw_ctx is None else raw_ctx.config().dict()
    op = TensorTileDBDataStore(tiledb_config=tiledb_config, tiledb_uri=uri,
                               tiledb_key=key, tiledb_timestamp=timestamp,
                               dtype=x.dtype, sparse=x.issparse())
    return op(x)
