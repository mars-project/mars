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

import numpy as np

from ...operands import ShuffleProxy
from ...core import TileableOperandMixin, FuseChunkData, FuseChunk
from ...operands import Operand, ShuffleMap, ShuffleReduce, Fuse
from ..core import DataFrameChunkData, DataFrameChunk, DataFrameData, DataFrame


class DataFrameOperandMixin(TileableOperandMixin):
    __slots__ = ()
    _op_module_ = 'dataframe'

    def _create_chunk(self, output_idx, index, **kw):
        data = DataFrameChunkData(_index=index, _shape=kw.pop('shape', None), _op=self,
                                  _dtypes=kw.pop('dtypes', None),
                                  _index_value=kw.pop('index_value', None),
                                  _columns_value=kw.pop('columns_value', None), **kw)
        return DataFrameChunk(data)

    def _create_tileable(self, output_idx, **kw):
        if kw.get('nsplits', None) is not None:
            kw['_nsplits'] = kw['nsplits']
        data = DataFrameData(_shape=kw.pop('shape', None), _op=self,
                             _chunks=kw.pop('chunks', None),
                             _dtypes=kw.pop('dtypes', None),
                             _index_value=kw.pop('index_value', None),
                             _columns_value=kw.pop('columns_value', None), **kw)
        return DataFrame(data)

    def new_dataframes(self, inputs, shape=None, dtypes=None, index_value=None, columns_value=None,
                       chunks=None, nsplits=None, output_limit=None, kws=None, **kw):
        return self.new_tileables(inputs, shape=shape, dtypes=dtypes, index_value=index_value,
                                  columns_value=columns_value, chunks=chunks, nsplits=nsplits,
                                  output_limit=output_limit, kws=kws, **kw)

    def new_dataframe(self, inputs, shape=None, dtypes=None, index_value=None, columns_value=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new tensor with more than 1 outputs')

        return self.new_dataframes(inputs, shape=shape, dtypes=dtypes,
                                   index_value=index_value, columns_value=columns_value, **kw)[0]

    @staticmethod
    def _merge_shape(*shapes):
        ret = [np.nan, np.nan]
        for shape in shapes:
            for i, s in enumerate(shape):
                if np.isnan(ret[i]) and not np.isnan(s):
                    ret[i] = s
        return tuple(ret)


class DataFrameOperand(Operand):
    pass


class DataFrameShuffleProxy(ShuffleProxy, DataFrameOperandMixin):
    def __init__(self, **kwargs):
        super(DataFrameShuffleProxy, self).__init__(**kwargs)


class DataFrameShuffleMap(ShuffleMap):
    pass


class DataFrameShuffleReduce(ShuffleReduce):
    pass


class DataFrameFuseMixin(DataFrameOperandMixin):
    __slots__ = ()

    def _create_chunk(self, output_idx, index, **kw):
        data = FuseChunkData(_index=index, _shape=kw.pop('shape', None), _op=self, **kw)

        return FuseChunk(data)


class DataFrameFuseChunk(Fuse, DataFrameFuseMixin):
    def __init__(self, sparse=False, **kwargs):
        super(DataFrameFuseChunk, self).__init__(_sparse=sparse, **kwargs)
