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

from ...operands import ShuffleProxy
from ...core import TileableOperandMixin
from ..core import DataFrameChunkData, DataFrameChunk, DataFrameData, DataFrame


class DataFrameOperandMixin(TileableOperandMixin):
    __slots__ = ()
    _op_module_ = 'dataframe'

    def _create_chunk(self, output_idx, index, shape, **kw):
        data = DataFrameChunkData(_index=index, _shape=shape, _op=self,
                                  _dtypes=kw.pop('dtypes', None),
                                  _index_value=kw.pop('index_value', None),
                                  _columns_value=kw.pop('columns_value', None), **kw)
        return DataFrameChunk(data)

    def _create_tileable(self, output_idx, shape, nsplits, chunks, **kw):
        if nsplits is not None:
            kw['_nsplits'] = nsplits
        data = DataFrameData(_shape=shape, _op=self, _chunks=chunks,
                             _dtypes=kw.pop('dtypes', None),
                             _index_value=kw.pop('index_value', None),
                             _columns_value=kw.pop('columns_value', None), **kw)
        return DataFrame(data)

    def new_dataframes(self, inputs, shape, dtypes=None, index_value=None, columns_value=None,
                       chunks=None, nsplits=None, output_limit=None, kws=None, **kw):
        return self.new_tileables(inputs, shape, dtypes=dtypes, index_value=index_value,
                                  columns_value=columns_value, chunks=chunks, nsplits=nsplits,
                                  output_limit=output_limit, kws=kws, **kw)

    def new_dataframe(self, inputs, shape, dtypes=None, index_value=None, columns_value=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new tensor with more than 1 outputs')

        return self.new_dataframes(inputs, shape, dtypes=dtypes,
                                   index_value=index_value, columns_value=columns_value, **kw)[0]


class DataFrameShuffleProxy(ShuffleProxy, DataFrameOperandMixin):
    def __init__(self, **kwargs):
        super(DataFrameShuffleProxy, self).__init__(**kwargs)

    def calc_shape(self, *inputs_shape):
        return ()
