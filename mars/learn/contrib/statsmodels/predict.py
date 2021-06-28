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

import pickle  # nosec  # pylint: disable=import_pickle

import numpy as np

from .... import opcodes
from ....core import OutputType, recursive_tile
from ....dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
from ....serialization.serializables import BytesField, DictField, TupleField
from ...operands import LearnOperand, LearnOperandMixin


class StatsModelsPredict(LearnOperand, LearnOperandMixin):
    _op_code_ = opcodes.STATSMODELS_PREDICT

    _model_results = BytesField('model_results', on_serialize=pickle.dumps,
                                on_deserialize=pickle.loads)
    _predict_args = TupleField('predict_args')
    _predict_kwargs = DictField('predict_kwargs')

    def __init__(self, model_results=None, predict_args=None, predict_kwargs=None, **kw):
        super().__init__(_model_results=model_results, _predict_args=predict_args,
                         _predict_kwargs=predict_kwargs, **kw)

    @property
    def model_results(self):
        return self._model_results

    @property
    def predict_args(self) -> tuple:
        return self._predict_args

    @property
    def predict_kwargs(self) -> dict:
        return self._predict_kwargs

    def __call__(self, exog):
        if isinstance(exog, (DATAFRAME_TYPE, SERIES_TYPE)):
            self._output_types = [OutputType.series]
            kwargs = dict(
                shape=exog.shape[:1],
                index_value=exog.index_value,
                dtype=np.dtype('float'),
            )
        else:
            self._output_types = [OutputType.tensor]
            kwargs = dict(
                shape=exog.shape[:1],
                dtype=np.dtype('float'),
            )
        return self.new_tileable([exog], **kwargs)

    @classmethod
    def tile(cls, op: 'StatsModelsPredict'):
        exog = op.inputs[0]
        out = op.outputs[0]

        if exog.ndim > 1 and exog.chunk_shape[1] > 1:
            exog = yield from recursive_tile(exog.rechunk({1: exog.shape[1]}))

        chunks = []
        for in_chunk in exog.chunks:
            if isinstance(exog, (DATAFRAME_TYPE, SERIES_TYPE)):
                kwargs = dict(
                    index=in_chunk.index[:1],
                    shape=in_chunk.shape[:1],
                    index_value=in_chunk.index_value,
                    dtype=out.dtype,
                )
            else:
                kwargs = dict(
                    index=in_chunk.index[:1],
                    shape=in_chunk.shape[:1],
                    dtype=out.dtype,
                )

            new_op = op.copy().reset_key()
            chunks.append(new_op.new_chunk([in_chunk], **kwargs))

        new_op = op.copy().reset_key()
        return new_op.new_tileables([exog], chunks=chunks, nsplits=(exog.nsplits[0],),
                                    **out.params)

    @classmethod
    def execute(cls, ctx, op: 'StatsModelsPredict'):
        in_data = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = op.model_results.model.predict(
            in_data, *op.predict_args, **op.predict_kwargs)
