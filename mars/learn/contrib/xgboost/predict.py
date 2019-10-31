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

import pickle
import base64

import numpy as np
import pandas as pd

from .... import opcodes as OperandDef
from ....operands import Operand
from ....core import TileableOperandMixin
from ....serialize import KeyField, StringField, Int8Field
from ....dataframe.operands import ObjectType
from ....dataframe.core import DATAFRAME_TYPE, SERIES_CHUNK_TYPE, DATAFRAME_CHUNK_TYPE, \
    DataFrameData, DataFrame, SeriesData, Series, DataFrameChunkData, DataFrameChunk, \
    SeriesChunkData, SeriesChunk
from ....dataframe.utils import parse_index
from ....tensor.core import TENSOR_TYPE, CHUNK_TYPE as TENSOR_CHUNK_TYPE, TensorData, Tensor, \
    TensorChunkData, TensorChunk, TensorOrder
from ....compat import six
from ....utils import register_tokenizer, to_str
from .dmatrix import ToDMatrix, check_data, \
    _on_serialize_object_type, _on_deserialize_object_type

try:
    from xgboost import Booster

    register_tokenizer(Booster, pickle.dumps)
except ImportError:
    pass


def _on_serialize_model(m):
    return to_str(base64.b64encode(pickle.dumps(m)))


def _on_deserialize_model(ser):
    return pickle.loads(base64.b64decode(ser))


class XGBPredict(Operand, TileableOperandMixin):
    _op_module_ = 'learn'
    _op_type_ = OperandDef.XGBOOST_PREDICT

    _data = KeyField('data')
    _model = StringField('model', on_serialize=_on_serialize_model, on_deserialize=_on_deserialize_model)
    _object_type = Int8Field('object_type', on_serialize=_on_serialize_object_type,
                             on_deserialize=_on_deserialize_object_type)

    def __init__(self, data=None, model=None, object_type=None, gpu=None, **kw):
        super(XGBPredict, self).__init__(_data=data, _model=model, _gpu=gpu,
                                         _object_type=object_type, **kw)

    @property
    def data(self):
        return self._data

    @property
    def model(self):
        return self._model

    @property
    def object_type(self):
        return self._object_type

    def _set_inputs(self, inputs):
        super(XGBPredict, self)._set_inputs(inputs)
        self._data = self._inputs[0]

    def _create_chunk(self, output_idx, index, **kw):
        shape = kw.pop('shape', self._data.shape)
        if isinstance(self._data, TENSOR_CHUNK_TYPE):
            dtype = kw.pop('dtype', self._data.dtype)
            order = kw.pop('order', self._data.order)
            data = TensorChunkData(shape=shape, dtype=dtype, op=self,
                                   order=order, index=index, **kw)
            return TensorChunk(data)
        else:
            assert isinstance(self._data, DATAFRAME_CHUNK_TYPE)
            if self._model.attr('num_class'):
                data = DataFrameChunkData(shape=shape, op=self,
                                          index=index, **kw)
                return DataFrameChunk(data)
            else:
                data = SeriesChunkData(shape=shape, op=self,
                                       index=index, **kw)
                return SeriesChunk(data)

    def _create_tileable(self, output_idx, **kw):
        shape = kw.pop('shape', self._data.shape)
        chunks = kw.pop('chunks', None)
        nsplits = kw.pop('nsplits', None)
        if isinstance(self._data, TENSOR_TYPE):
            dtype = kw.pop('dtype', self._data.dtype)
            order = kw.pop('order', self._data.order)
            data = TensorData(shape=shape, dtype=dtype, op=self,
                              order=order, chunks=chunks, nsplits=nsplits, **kw)
            return Tensor(data)
        else:
            assert isinstance(self._data, DATAFRAME_TYPE)
            if self._model.attr('num_class'):
                data = DataFrameData(shape=shape, op=self,
                                     chunks=chunks, nsplits=nsplits, **kw)
                return DataFrame(data)
            else:
                data = SeriesData(shape=shape, op=self,
                                  chunks=chunks, nsplits=nsplits, **kw)
                return Series(data)

    def __call__(self):
        num_class = self._model.attr('num_class')
        if num_class is not None:
            num_class = int(num_class)
        if num_class is not None:
            shape = (len(self._data), int(self._model.attr('num_class')))
        else:
            shape = (len(self._data),)
        if isinstance(self._data, TENSOR_TYPE):
            return self.new_tileable([self._data], shape=shape, dtype=np.dtype(np.float32),
                                     order=TensorOrder.C_ORDER)
        elif num_class is not None:
            # dataframe
            dtypes = pd.DataFrame(np.random.rand(0, num_class), dtype=np.float32).dtypes
            return self.new_tileable([self._data], shape=shape, dtypes=dtypes,
                                     columns_value=parse_index(dtypes.index),
                                     index_value=self._data.index_value)
        else:
            # series
            return self.new_tileable([self._data], shape=shape, index_value=self._data.index_value,
                                     name='predictions', dtype=np.dtype(np.float32))

    @classmethod
    def tile(cls, op):
        out = op.outputs[0]
        out_chunks = []
        data = op.data
        if data.chunk_shape[1] > 1:
            data = data.rechunk({1: op.data.shape[1]}).single_tiles()
        for in_chunk in data.chunks:
            chunk_op = op.copy().reset_key()
            chunk_index = (in_chunk.index[0],)
            if op.model.attr('num_class'):
                chunk_shape = (len(in_chunk), 2)
                chunk_index += (0,)
            else:
                chunk_shape = (len(in_chunk),)
            if isinstance(op._data, TENSOR_TYPE):
                out_chunk = chunk_op.new_chunk([in_chunk], shape=chunk_shape,
                                               order=out.order, index=chunk_index)
            elif op.model.attr('num_class'):
                # dataframe chunk
                out_chunk = chunk_op.new_chunk([in_chunk], shape=chunk_shape,
                                               dtypes=data.dtypes,
                                               columns_value=data.columns,
                                               index_value=in_chunk.index_value,
                                               index=chunk_index)
            else:
                # series chunk
                out_chunk = chunk_op.new_chunk([in_chunk], shape=chunk_shape,
                                               dtype=out.dtype,
                                               index_value=in_chunk.index_value,
                                               name=out.name, index=chunk_index)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        params = out.params
        params['chunks'] = out_chunks
        nsplits = (data.nsplits[0],)
        if out.ndim > 1:
            nsplits += ((out.shape[1],),)
        params['nsplits'] = nsplits
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def execute(cls, ctx, op):
        from xgboost import DMatrix

        raw_data = data = ctx[op.data.key]
        if isinstance(data, tuple):
            data = ToDMatrix.get_xgb_dmatrix(data)
        else:
            data = DMatrix(data)
        result = op.model.predict(data)

        if isinstance(op.outputs[0], DATAFRAME_CHUNK_TYPE):
            result = pd.DataFrame(result, index=raw_data.index)
        elif isinstance(op.outputs[0], SERIES_CHUNK_TYPE):
            result = pd.Series(result, index=raw_data.index, name='predictions')

        ctx[op.outputs[0].key] = result


def predict(model, data, session=None, run_kwargs=None, run=True):
    from xgboost import Booster

    data = check_data(data)
    if not isinstance(model, Booster):
        raise TypeError('model has to be a xgboost.Booster, got {0} instead'.format(type(model)))
    kw = dict()
    if isinstance(data, DATAFRAME_TYPE):
        kw['object_type'] = ObjectType.dataframe if model.attr('num_class') else ObjectType.series

    op = XGBPredict(data=data, model=model, gpu=data.op.gpu, **kw)
    result = op()
    if run:
        result.execute(session=session, fetch=False, **(run_kwargs or dict()))
    return result
