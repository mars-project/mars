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


import itertools

import numpy as np

from ...config import options
from ...serialization.serializables import FieldTypes, StringField, TupleField
from ..utils import normalize_shape, decide_chunk_sizes
from ..core import TensorOrder
from ..operands import TensorOperand, TensorOperandMixin


class TensorDataSource(TensorOperand, TensorOperandMixin):
    """
    Tensor data source base class, provide universal tile logic,
    subclass can overwrite tile method.
    """

    __slots__ = ()

    def to_chunk_op(self, *args):
        chunk_shape = args[0]
        chunk_op = self.copy().reset_key()
        chunk_op.extra_params = {'size': chunk_shape}  # to make op key different
        return chunk_op

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]

        chunk_size = tensor.extra_params.raw_chunk_size or options.chunk_size
        chunk_size = decide_chunk_sizes(tensor.shape, chunk_size, tensor.dtype.itemsize)
        chunk_size_idxes = (range(len(size)) for size in chunk_size)

        out_chunks = []
        for chunk_shape, chunk_idx in zip(itertools.product(*chunk_size),
                                          itertools.product(*chunk_size_idxes)):
            chunk_op = op.to_chunk_op(chunk_shape, chunk_idx, chunk_size)
            out_chunk = chunk_op.new_chunk(None, shape=chunk_shape, index=chunk_idx,
                                           order=tensor.order)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, tensor.shape, chunks=out_chunks, nsplits=chunk_size,
                                  order=tensor.order, **tensor.extra_params)


class TensorNoInput(TensorDataSource):
    """
    Tensor operand with no inputs.
    """

    def check_inputs(self, inputs):
        # no inputs
        if inputs and len(inputs) > 0:
            raise ValueError("Tensor data source has no inputs")

    def _new_chunks(self, inputs, kws=None, **kw):
        shape = kw.get('shape', None)
        self.extra_params['shape'] = shape  # set shape to make the operand key different
        return super()._new_chunks(inputs, kws=kws, **kw)

    def _new_tileables(self, inputs, kws=None, **kw):
        shape = kw.get('shape', None)
        self.extra_params['shape'] = shape  # set shape to make the operand key different
        return super()._new_tileables(inputs, kws=kws, **kw)

    def __call__(self, shape, chunk_size=None, order=None):
        shape = normalize_shape(shape)
        order = TensorOrder.C_ORDER if order is None else order
        return self.new_tensor(None, shape, raw_chunk_size=chunk_size, order=order)


class TensorHasInput(TensorDataSource):
    """
    Tensor operand with a single input.
    """

    @property
    def input(self):
        return self._input

    def check_inputs(self, inputs):
        # no inputs
        if len(inputs) != 1:
            raise ValueError("Tensor can only have 1 input")

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    @classmethod
    def tile(cls, op):
        output = op.outputs[0]

        out_chunks = []
        for c in op.input.chunks:
            out_chunk = op.copy().reset_key().new_chunk([c], shape=c.shape,
                                                        index=c.index, order=output.order)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, output.shape, order=output.order,
                                  chunks=out_chunks, nsplits=op.input.nsplits)

    def __call__(self, a, order=None):
        order = a.order if order is None else order
        return self.new_tensor([a], a.shape, order=order)


class TensorLike(TensorHasInput):
    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self.dtype is None:
            self.dtype = self.input.dtype
        if self.gpu is None:
            self.gpu = self.input.op.gpu

        # FIXME: remove when cupy supports other dtypes
        if self.gpu and self.dtype not in (np.float32, np.float64):
            raise NotImplementedError('Sparse tensor on GPU only supports float32 and float64')


class TensorFromHDF5Like(TensorNoInput):
    _filename = StringField('filename')
    _group = StringField('group')
    _dataset = StringField('dataset')
    _axis_offsets = TupleField('axis_offsets', FieldTypes.int64)

    def __init__(self, filename=None, group=None, dataset=None, **kw):
        super().__init__(_filename=filename, _group=group,
                         _dataset=dataset, **kw)

    @property
    def filename(self):
        return self._filename

    @property
    def group(self):
        return self._group

    @property
    def dataset(self):
        return self._dataset

    @property
    def axis_offsets(self):
        return self._axis_offsets

    @property
    def path(self):
        return self.get_path(self.group, self.dataset)

    def to_chunk_op(self, *args):
        _, chunk_index, nsplits = args
        chunk_op = super().to_chunk_op(*args)
        cum_offsets = [[0] + np.cumsum(ns).tolist() for ns in nsplits]
        axis_offsets = []
        for axis, idx in enumerate(chunk_index):
            axis_offsets.append(cum_offsets[axis][idx])
        chunk_op._axis_offsets = tuple(axis_offsets)
        return chunk_op

    @staticmethod
    def get_path(group, dataset):
        paths = []
        if group:
            paths.append(group)
        paths.append(dataset)
        return '/'.join(paths)
