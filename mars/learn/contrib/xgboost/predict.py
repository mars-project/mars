# Copyright 1999-2021 Alibaba Group Holding Ltd.
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

import numpy as np
import pandas as pd

from .... import opcodes as OperandDef
from ....core import recursive_tile
from ....serialization.serializables import KeyField, BytesField, DictField
from ....dataframe.core import SERIES_CHUNK_TYPE, DATAFRAME_CHUNK_TYPE
from ....dataframe.utils import parse_index
from ....tensor.core import TENSOR_TYPE, TensorOrder
from ....utils import has_unknown_shape, ensure_own_data
from ...operands import LearnOperand, LearnOperandMixin, OutputType
from .dmatrix import ToDMatrix, check_data


class XGBPredict(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.XGBOOST_PREDICT

    _data = KeyField('data')
    _model = BytesField('model', on_serialize=pickle.dumps, on_deserialize=pickle.loads)
    _kwargs = DictField('kwargs')

    def __init__(self, data=None, model=None, kwargs=None,
                 output_types=None, gpu=None, **kw):
        super().__init__(_data=data, _model=model, _kwargs=kwargs,
                         _gpu=gpu, _output_types=output_types, **kw)

    @property
    def data(self):
        return self._data

    @property
    def model(self):
        return self._model

    @property
    def kwargs(self):
        return self._kwargs if self._kwargs is not None else dict()

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._data = self._inputs[0]

    def __call__(self):
        num_class = self._model.attr('num_class')
        if num_class is not None:
            num_class = int(num_class)
        if num_class is not None:
            shape = (self._data.shape[0], num_class)
        else:
            shape = (self._data.shape[0],)
        inputs = [self._data]
        if self.output_types[0] == OutputType.tensor:
            # tensor
            return self.new_tileable(inputs, shape=shape, dtype=np.dtype(np.float32),
                                     order=TensorOrder.C_ORDER)
        elif self.output_types[0] == OutputType.dataframe:
            # dataframe
            dtypes = pd.DataFrame(np.random.rand(0, num_class), dtype=np.float32).dtypes
            return self.new_tileable(inputs, shape=shape, dtypes=dtypes,
                                     columns_value=parse_index(dtypes.index),
                                     index_value=self._data.index_value)
        else:
            # series
            return self.new_tileable(inputs, shape=shape, index_value=self._data.index_value,
                                     name='predictions', dtype=np.dtype(np.float32))

    @classmethod
    def tile(cls, op):
        out = op.outputs[0]
        out_chunks = []
        data = op.data
        if data.chunk_shape[1] > 1:
            if has_unknown_shape(op.data):
                yield
            data = yield from recursive_tile(data.rechunk({1: op.data.shape[1]}))
        for in_chunk in data.chunks:
            chunk_op = op.copy().reset_key()
            chunk_index = (in_chunk.index[0],)
            if op.model.attr('num_class'):
                chunk_shape = (in_chunk.shape[0], int(op.model.attr('num_class')))
                chunk_index += (0,)
            else:
                chunk_shape = (in_chunk.shape[0],)
            if op.output_types[0] == OutputType.tensor:
                out_chunk = chunk_op.new_chunk([in_chunk], shape=chunk_shape,
                                               dtype=out.dtype,
                                               order=out.order, index=chunk_index)
            elif op.output_types[0] == OutputType.dataframe:
                # dataframe chunk
                out_chunk = chunk_op.new_chunk([in_chunk], shape=chunk_shape,
                                               dtypes=data.dtypes,
                                               columns_value=data.columns_value,
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
            data = ToDMatrix.get_xgb_dmatrix(ensure_own_data(data))
        else:
            data = data.spmatrix if hasattr(data, 'spmatrix') else data
            data = DMatrix(data)

        # do not pass arguments that are None
        kwargs = dict((k, v) for k, v in op.kwargs.items()
                      if v is not None)
        result = op.model.predict(data, **kwargs)

        if isinstance(op.outputs[0], DATAFRAME_CHUNK_TYPE):
            result = pd.DataFrame(result, index=raw_data.index)
        elif isinstance(op.outputs[0], SERIES_CHUNK_TYPE):
            result = pd.Series(result, index=raw_data.index, name='predictions')

        ctx[op.outputs[0].key] = result


def predict(model, data, output_margin=False, ntree_limit=None,
            validate_features=True, base_margin=None,
            session=None, run_kwargs=None, run=True):
    from xgboost import Booster

    data = check_data(data)
    if not isinstance(model, Booster):
        raise TypeError(f'model has to be a xgboost.Booster, got {type(model)} instead')

    num_class = model.attr('num_class')
    if isinstance(data, TENSOR_TYPE):
        output_types = [OutputType.tensor]
    elif num_class is not None:
        output_types = [OutputType.dataframe]
    else:
        output_types = [OutputType.series]

    kwargs = {
        'output_margin': output_margin,
        'ntree_limit': ntree_limit,
        'validate_features': validate_features,
        'base_margin': base_margin
    }
    op = XGBPredict(data=data, model=model, kwargs=kwargs,
                    gpu=data.op.gpu, output_types=output_types)
    result = op()
    if run:
        result.execute(session=session, **(run_kwargs or dict()))
    return result
