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

from .... import opcodes
from ....core import recursive_tile
from ....dataframe.utils import parse_index
from ....serialization.serializables import BoolField, BytesField, DictField, KeyField
from ....tensor.core import TENSOR_TYPE, TensorOrder
from ...operands import LearnOperand, LearnOperandMixin, OutputType


class LGBMPredict(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.LGBM_PREDICT

    _data = KeyField('data')
    _model = BytesField('model', on_serialize=pickle.dumps, on_deserialize=pickle.loads)
    _proba = BoolField('proba')
    _kwds = DictField('kwds')

    def __init__(self, data=None, model=None, proba=None, kwds=None,
                 output_types=None, **kw):
        super().__init__(_data=data, _model=model, _proba=proba, _kwds=kwds,
                         _output_types=output_types, **kw)

    @property
    def data(self):
        return self._data

    @property
    def model(self):
        return self._model

    @property
    def proba(self) -> bool:
        return self._proba

    @property
    def kwds(self) -> dict:
        return self._kwds

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        it = iter(inputs)
        self._data = next(it)

    def __call__(self):
        num_class = int(getattr(self.model, 'n_classes_', 2))
        if self.proba:
            shape = (self.data.shape[0], num_class)
        else:
            shape = (self.data.shape[0],)

        if self._proba:
            dtype = np.dtype(np.float_)
        elif hasattr(self.model, 'classes_'):
            dtype = np.array(self.model.classes_).dtype
        else:
            dtype = getattr(self.model, 'out_dtype_', np.dtype('float'))

        if self.output_types[0] == OutputType.tensor:
            # tensor
            return self.new_tileable([self.data], shape=shape, dtype=dtype,
                                     order=TensorOrder.C_ORDER)
        elif self.output_types[0] == OutputType.dataframe:
            # dataframe
            dtypes = pd.Series([dtype] * num_class)
            columns_value = parse_index(pd.Index(self.model.classes_), store_data=True)
            return self.new_tileable([self.data], shape=shape, dtypes=dtypes,
                                     columns_value=columns_value,
                                     index_value=self.data.index_value)
        else:
            return self.new_tileable([self.data], shape=shape, index_value=self.data.index_value,
                                     name='predictions', dtype=dtype)

    @classmethod
    def tile(cls, op: "LGBMPredict"):
        out = op.outputs[0]
        out_chunks = []
        data = op.data
        if data.chunk_shape[1] > 1:
            data = yield from recursive_tile(data.rechunk({1: op.data.shape[1]}))

        for in_chunk in data.chunks:
            chunk_op = op.copy().reset_key()
            chunk_index = (in_chunk.index[0],)

            if len(out.shape) > 1:
                chunk_shape = (in_chunk.shape[0], out.shape[1])
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
                                               dtypes=out.dtypes,
                                               columns_value=out.columns_value,
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
    def execute(cls, ctx, op: "LGBMPredict"):
        in_data = ctx[op.data.key]
        in_data = in_data.spmatrix if hasattr(in_data, 'spmatrix') else in_data
        out = op.outputs[0]

        if op.data.shape[0] == 0:
            result = np.array([])
        elif op.proba:
            result = op.model.predict_proba(in_data, **op.kwds)
        else:
            result = op.model.predict(in_data, **op.kwds)

        if op.output_types[0] == OutputType.dataframe:
            result = pd.DataFrame(result, index=in_data.index, columns=out.columns_value.to_pandas())
        elif op.output_types[0] == OutputType.series:
            result = pd.Series(result, index=in_data.index, name='predictions')

        ctx[out.key] = result


def predict_base(model, data, session=None, run_kwargs=None, run=True, **kwargs):
    from lightgbm import LGBMModel

    if not isinstance(model, LGBMModel):
        raise TypeError(f'model has to be a lightgbm.LGBMModel, got {type(model)} instead')
    model = model.to_local() if hasattr(model, 'to_local') else model

    proba = kwargs.pop('proba', hasattr(model, 'classes_'))

    if isinstance(data, TENSOR_TYPE):
        output_types = [OutputType.tensor]
    elif proba:
        output_types = [OutputType.dataframe]
    else:
        output_types = [OutputType.series]

    op = LGBMPredict(data=data, model=model, gpu=data.op.gpu, output_types=output_types,
                     proba=proba, kwds=kwargs)
    result = op()
    if run:
        result.execute(session=session, **(run_kwargs or dict()))
    return result


def predict(model, data, session=None, run_kwargs=None, run=True, **kw):
    if hasattr(model, 'classes_'):
        return predict_base(model, data, session=session, run_kwargs=run_kwargs,
                            proba=False, run=run, **kw)
    else:
        return predict_base(model, data, session=session, run_kwargs=run_kwargs,
                            run=run, **kw)


def predict_proba(model, data, session=None, run_kwargs=None, run=True, **kw):
    return predict_base(model, data, session=session, run_kwargs=run_kwargs,
                        run=run, proba=True, **kw)
