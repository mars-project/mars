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
from ....serialize import DictField, StringField, Int64Field
from .core import TensorNoInput


class TensorTileDBDataSource(TensorNoInput):
    _op_type_ = OperandDef.TENSOR_FROM_TILEDB

    _tiledb_config = DictField('tiledb_config')
    # URI of array to open
    _tiledb_uri = StringField('tiledb_uri')
    # encryption key to decrypt if provided
    _tiledb_key = StringField('tiledb_key')
    # open array at a given timestamp if provided
    _tiledb_timestamp = Int64Field('tiledb_timestamp')

    def __init__(self, tiledb_config=None, tiledb_uri=None, tiledb_key=None,
                 tiledb_timstamp=None, dtype=None, gpu=None, sparse=None, **kw):
        super(TensorTileDBDataSource, self).__init__(
            _tiledb_config=tiledb_config, _tiledb_uri=tiledb_uri, _tiledb_key=tiledb_key,
            _tiledb_timestamp=tiledb_timstamp, _dtype=dtype, _gpu=gpu, _sparse=sparse, **kw)

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


def fromtiledb(uri, ctx=None, key=None, timestamp=None, gpu=None):
    import tiledb

    raw_ctx = ctx
    if raw_ctx is None:
        ctx = tiledb.Ctx()

    # get metadata from tiledb
    try:
        tiledb_arr = tiledb.DenseArray(ctx, uri, key=key, timestamp=timestamp)
        sparse = False
    except ValueError:
        # if the array is not dense, ValueError will be raised by tiledb
        tiledb_arr = tiledb.SparseArray(ctx, uri, key=key, timestamp=timestamp)
        sparse = True

    if tiledb_arr.nattr > 1:
        raise NotImplementedError('Does not supported TileDB array schema '
                                  'with more than 1 attr')

    dtype = tiledb_arr.attr(0).dtype
    tiledb_config = None if raw_ctx is None else ctx.config().dict()
    op = TensorTileDBDataSource(tiledb_config=tiledb_config, tiledb_uri=uri,
                                tiledb_key=key, tiledb_timstamp=timestamp,
                                gpu=gpu, sparse=sparse, dtype=dtype)
    chunk_size = tuple(int(tiledb_arr.domain.dim(i).tile)
                       for i in range(tiledb_arr.domain.ndim))
    return op(tiledb_arr.shape, chunk_size=chunk_size)
