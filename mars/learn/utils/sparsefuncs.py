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

from typing import Optional

import numpy as np

from ... import opcodes, tensor as mt
from ...core import OutputType, recursive_tile
from ...serialization.serializables import Int16Field, ReferenceField
from ...utils import has_unknown_shape
from ..operands import LearnOperand, LearnOperandMixin


class LearnCountNonzero(LearnOperand, LearnOperandMixin):
    _op_code_ = opcodes.COUNT_NONZERO

    axis = Int16Field("axis")
    sample_weight = ReferenceField("sample_weight")

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self.sample_weight is not None:
            self.sample_weight = inputs[-1]

    def __call__(self, x, sample_weight=None):
        self.sample_weight = sample_weight
        self._output_types = [
            OutputType.scalar if self.axis is None else OutputType.tensor
        ]
        dtype = np.dtype(int)
        inputs = [x]
        if sample_weight is not None:
            dtype = sample_weight.dtype
            inputs = [x, sample_weight]

        if self.axis is None:
            shape = ()
        else:
            shape = (x.shape[1 - self.axis],)

        return self.new_tileable(inputs, shape=shape, dtype=dtype)

    @classmethod
    def tile(cls, op: "LearnCountNonzero"):
        input_tensor = op.inputs[0]
        out_tensor = op.outputs[0]

        if op.sample_weight is not None:
            if has_unknown_shape(input_tensor):
                yield
            sample_weight = yield from recursive_tile(
                op.sample_weight.rechunk({0: input_tensor.nsplits[0]})
            )
        else:
            sample_weight = None

        chunks = []
        for input_chunk in input_tensor.chunks:
            if sample_weight is None:
                weight_chunk = None
            else:
                weight_chunk = sample_weight.cix[(input_chunk.index[0],)]

            new_op = op.copy().reset_key()
            new_op.sample_weight = weight_chunk

            inputs = [input_chunk] if not weight_chunk else [input_chunk, weight_chunk]
            if op.axis is None:
                shape = (1, 1)
            elif op.axis == 0:
                shape = (1, input_chunk.shape[1])
            else:
                shape = (input_chunk.shape[0], 1)
            chunks.append(
                new_op.new_chunk(
                    inputs, shape=shape, dtype=out_tensor.dtype, index=input_chunk.index
                )
            )

        new_op = op.copy().reset_key()
        if op.axis is None:
            nsplits = tuple((1,) * len(split) for split in input_tensor.nsplits)
            shape = tuple(len(split) for split in input_tensor.nsplits)
        elif op.axis == 0:
            nsplits = ((1,) * len(input_tensor.nsplits[0]), input_tensor.nsplits[1])
            shape = (len(input_tensor.nsplits[0]), input_tensor.shape[1])
        else:
            nsplits = (input_tensor.nsplits[0], (1,) * len(input_tensor.nsplits[1]))
            shape = (input_tensor.shape[0], len(input_tensor.nsplits[1]))

        tileable = new_op.new_tileable(
            out_tensor.inputs,
            chunks=chunks,
            nsplits=nsplits,
            shape=shape,
            dtype=out_tensor.dtype,
        )
        return [(yield from recursive_tile(mt.sum(tileable, axis=op.axis)))]

    @classmethod
    def execute(cls, ctx, op: "LearnCountNonzero"):
        axis = op.axis
        X = ctx[op.inputs[0].key]
        sample_weight = (
            ctx[op.sample_weight.key] if op.sample_weight is not None else None
        )

        # We rely here on the fact that np.diff(Y.indptr) for a CSR
        # will return the number of nonzero entries in each row.
        # A bincount over Y.indices will return the number of nonzeros
        # in each column. See ``csr_matrix.getnnz`` in scipy >= 0.14.
        if axis is None:
            if sample_weight is None:
                res = X.nnz
            else:
                res = np.dot(np.diff(X.indptr), sample_weight)
        elif axis == 1:
            out = np.diff(X.indptr)
            if sample_weight is None:
                # astype here is for consistency with axis=0 dtype
                res = out.astype("intp")
            else:
                res = out * sample_weight
        else:
            if sample_weight is None:
                res = np.bincount(X.indices, minlength=X.shape[1])
            else:
                weights = np.repeat(sample_weight, np.diff(X.indptr))
                res = np.bincount(X.indices, minlength=X.shape[1], weights=weights)
        if np.isscalar(res):
            res = np.array([res])
        out_shape = op.outputs[0].shape
        if any(np.isnan(s) for s in out_shape):
            new_shape = list(out_shape)
            for i, s in enumerate(out_shape):
                if np.isnan(s):
                    new_shape[i] = -1
            out_shape = tuple(new_shape)
        ctx[op.outputs[0].key] = res.reshape(out_shape)


def count_nonzero(X, axis: Optional[int] = None, sample_weight=None):
    """A variant of X.getnnz() with extension to weighting on axis 0

    Useful in efficiently calculating multilabel metrics.

    Parameters
    ----------
    X : CSR sparse matrix of shape (n_samples, n_labels)
        Input data.

    axis : None, 0 or 1
        The axis on which the data is aggregated.

    sample_weight : array-like of shape (n_samples,), default=None
        Weight for each row of X.
    """
    if axis == -1:
        axis = 1
    elif axis == -2:
        axis = 0
    if axis is not None and axis not in (0, 1):
        raise ValueError(f"Unsupported axis: {axis}")

    X = mt.asarray(X)
    if sample_weight is not None:
        sample_weight = mt.asarray(sample_weight)

    op = LearnCountNonzero(axis=axis)
    return op(X, sample_weight=sample_weight)
