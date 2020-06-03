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

import itertools

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...config import options
from ...serialize import IndexField, DataTypeField, KeyField
from ...tensor.utils import get_chunk_slices
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import parse_index, decide_series_chunk_size


class IndexDataSource(DataFrameOperand, DataFrameOperandMixin):
    """
    Represent data from pandas Index
    """

    _op_type_ = OperandDef.INDEX_DATA_SOURCE

    _input = KeyField('input')
    _data = IndexField('data')
    _dtype = DataTypeField('dtype')

    def __init__(self, input=None, data=None, dtype=None, gpu=None,  # pylint: disable=redefined-builtin
                 sparse=None, **kw):
        super().__init__(_input=input, _data=data, _dtype=dtype, _gpu=gpu,
                         _sparse=sparse, _object_type=ObjectType.index, **kw)

    @property
    def input(self):
        return self._input

    @property
    def data(self):
        return self._data

    @property
    def dtype(self):
        return self._dtype

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if inputs is not None and len(inputs) > 0:
            self._input = self._inputs[0]

    def __call__(self, shape=None, chunk_size=None, inp=None, name=None):
        if inp is None:
            # create from pandas Index
            name = name if name is not None else self._data.name
            return self.new_index(None, shape=shape, dtype=self._dtype,
                                  index_value=parse_index(self._data),
                                  name=name, raw_chunk_size=chunk_size)
        elif hasattr(inp, 'index_value'):
            # get index from Mars DataFrame, Series or Index
            name = name if name is not None else inp.index_value.name
            if inp.index_value.has_value():
                self._data = data = inp.index_value.to_pandas()
                return self.new_index(None, shape=(inp.shape[0],), dtype=data.dtype,
                                      index_value=parse_index(data),
                                      name=name, raw_chunk_size=chunk_size)
            else:
                if self._dtype is None:
                    self._dtype = inp.index_value.to_pandas().dtype
                return self.new_index([inp], shape=(inp.shape[0],),
                                      dtype=self._dtype, index_value=inp.index_value,
                                      name=name)
        else:
            if inp.ndim != 1:
                raise ValueError('Index data must be 1-dimensional')
            # get index from tensor
            dtype = inp.dtype if self._dtype is None else self._dtype
            pd_index = pd.Index([], dtype=dtype)
            if self._dtype is None:
                self._dtype = pd_index.dtype
            return self.new_index([inp], shape=inp.shape, dtype=self._dtype,
                                  index_value=parse_index(pd_index, inp),
                                  name=name)

    @classmethod
    def _tile_from_pandas(cls, op):
        index = op.outputs[0]
        raw_index = op.data

        memory_usage = raw_index.memory_usage(deep=True)
        chunk_size = index.extra_params.raw_chunk_size or options.chunk_size
        chunk_size = decide_series_chunk_size(index.shape, chunk_size, memory_usage)
        chunk_size_idxes = (range(len(size)) for size in chunk_size)

        out_chunks = []
        for chunk_index, chunk_shape in zip(itertools.product(*chunk_size_idxes),
                                            itertools.product(*chunk_size)):
            chunk_op = op.copy().reset_key()
            slc = get_chunk_slices(chunk_size, chunk_index)
            chunk_op._data = chunk_data = raw_index[slc]
            out_chunk = chunk_op.new_chunk(None, shape=chunk_shape, dtype=index.dtype,
                                           index=chunk_index, name=index.name,
                                           index_value=parse_index(chunk_data))
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_indexes(None, index.shape, dtype=index.dtype,
                                  index_value=index.index_value,
                                  name=index.name, chunks=out_chunks, nsplits=chunk_size)

    @classmethod
    def _tile_from_dataframe(cls, op):
        inp = op.input
        out = op.outputs[0]

        out_chunks = []
        if inp.ndim == 1:
            # series, index
            for c in inp.chunks:
                chunk_op = op.copy().reset_key()
                out_chunk = chunk_op.new_chunk([c], shape=c.shape,
                                               dtype=out.dtype, index=c.index,
                                               index_value=c.index_value,
                                               name=out.name)
                out_chunks.append(out_chunk)
            nsplits = inp.nsplits
        else:
            # DataFrame
            nsplit = inp.nsplits[1]
            axis_1_index = np.argmin(nsplit).item()
            for i in range(inp.chunk_shape[0]):
                chunk_index = (i, axis_1_index)
                c = inp.cix[chunk_index]
                chunk_op = op.copy().reset_key()
                out_chunk = chunk_op.new_chunk([c], shape=(c.shape[0],),
                                               dtype=out.dtype, index=(i,),
                                               index_value=c.index_value,
                                               name=out.name)
                out_chunks.append(out_chunk)
            nsplits = (inp.nsplits[0],)

        new_op = op.copy()
        params = out.params
        params['chunks'] = out_chunks
        params['nsplits'] = nsplits
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def _tile_from_tensor(cls, op):
        inp = op.input
        out = op.outputs[0]
        out_chunks = []
        for c in inp.chunks:
            chunk_op = op.copy().reset_key()
            index_value = parse_index(out.index_value.to_pandas(), c)
            out_chunk = chunk_op.new_chunk([c], shape=c.shape,
                                           dtype=out.dtype, index=c.index,
                                           index_value=index_value,
                                           name=out.name)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        params = out.params
        params['chunks'] = out_chunks
        params['nsplits'] = inp.nsplits
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def tile(cls, op):
        if op.input is None:
            # from pandas
            return cls._tile_from_pandas(op)
        elif hasattr(op.input, 'index_value'):
            # from DataFrame or Series
            return cls._tile_from_dataframe(op)
        else:
            # from tensor
            return cls._tile_from_tensor(op)

    @classmethod
    def execute(cls, ctx, op):
        if op.input is None:
            # from pandas
            ctx[op.outputs[0].key] = op.data
        else:
            out = op.outputs[0]
            inp = ctx[op.input.key]
            dtype = out.dtype if out.dtype != np.object else None
            if hasattr(inp, 'index'):
                # DataFrame, Series
                ctx[out.key] = pd.Index(inp.index, dtype=dtype, name=out.name)
            else:
                ctx[out.key] = pd.Index(inp, dtype=dtype, name=out.name)


def from_pandas(data, chunk_size=None, gpu=False, sparse=False):
    op = IndexDataSource(data=data, gpu=gpu, sparse=sparse, dtype=data.dtype)
    return op(shape=data.shape, chunk_size=chunk_size)


def from_tileable(tileable, dtype=None, name=None):
    op = IndexDataSource(gpu=tileable.op.gpu, sparse=tileable.issparse(),
                         dtype=dtype)
    return op(inp=tileable, name=name)
