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

import numpy as np

from ... import opcodes as OperandDef
from ...lib.sparse.array import get_sparse_module, SparseNDArray
from ...serialize import KeyField, Int64Field
from ...tensor.array_utils import as_same_device, device
from ...tensor.operands import TensorOrder
from ...tensor.utils import decide_unify_split
from ...tiles import TilesError
from ...utils import check_chunks_unknown_shape
from ..operands import LearnOperand, LearnOperandMixin, OutputType


class KNeighborsGraph(LearnOperand, LearnOperandMixin):
    _op_type_ = OperandDef.KNEIGHBORS_GRAPH

    _a_data = KeyField('a_data')
    _a_ind = KeyField('a_ind')
    _n_neighbors = Int64Field('n_neighbors')

    def __init__(self, a_data=None, a_ind=None, n_neighbors=None,
                 sparse=None, gpu=None, **kw):
        super().__init__(_a_data=a_data, _a_ind=a_ind, _n_neighbors=n_neighbors,
                         _sparse=sparse, _gpu=gpu, **kw)
        self._output_types = [OutputType.tensor]

    @property
    def a_data(self):
        return self._a_data

    @property
    def a_ind(self):
        return self._a_ind

    @property
    def n_neighbors(self):
        return self._n_neighbors

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self._a_data is not None:
            self._a_data = self._inputs[0]
        self._a_ind = self._inputs[-1]

    def __call__(self, A_data, A_ind, shape):
        inputs = []
        if A_data is not None:
            inputs.append(A_data)
        inputs.append(A_ind)
        return self.new_tileable(inputs, dtype=np.dtype(np.float64),
                                 shape=shape, order=TensorOrder.C_ORDER)

    @classmethod
    def tile(cls, op):
        check_chunks_unknown_shape(op.inputs, TilesError)
        A_data, A_ind = op.a_data, op.a_ind
        out = op.outputs[0]

        shape1 = A_ind.shape[1]
        if A_data is not None:
            # mode == 'distance'
            axis0_chunk_sizes = decide_unify_split(A_data.nsplits[0],
                                                   A_ind.nsplits[0])
            A_data = A_data.rechunk({0: axis0_chunk_sizes,
                                     1: shape1})._inplace_tile()
            A_ind = A_ind.rechunk({0: axis0_chunk_sizes,
                                   1: shape1})._inplace_tile()
        else:
            # mode == 'connectivity'
            A_ind = A_ind.rechunk({1: shape1})._inplace_tile()

        out_chunks = []
        for i, ind_c in enumerate(A_ind.chunks):
            chunk_op = op.copy().reset_key()
            chunk_inputs = [ind_c]
            if A_data is not None:
                data_c = A_data.cix[i, 0]
                chunk_inputs.insert(0, data_c)
            out_chunk = chunk_op.new_chunk(chunk_inputs, dtype=out.dtype,
                                           shape=(ind_c.shape[0], out.shape[1]),
                                           order=out.order, index=(i, 0))
            out_chunks.append(out_chunk)

        new_op = op.copy()
        params = out.params
        params['chunks'] = out_chunks
        params['nsplits'] = (A_ind.nsplits[0], (out.shape[1],))
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)
        out = op.outputs[0]
        n_samples1, n_samples2 = out.shape
        n_neighbors = op.n_neighbors
        n_nonzero = n_samples1 * n_neighbors

        with device(device_id):
            A_ind = inputs[-1]
            A_indptr = xp.arange(0, n_nonzero + 1, n_neighbors)

            if op.a_data is None:
                # mode == 'connectivity
                A_data = xp.ones(n_samples1 * n_neighbors)
            else:
                # mode == 'distance'
                A_data = xp.ravel(inputs[0])

            xps = get_sparse_module(A_ind)
            graph = xps.csr_matrix((A_data, A_ind.ravel(), A_indptr),
                                   shape=(n_samples1, n_samples2))
            ctx[out.key] = SparseNDArray(graph)
