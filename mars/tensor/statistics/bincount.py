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
import functools
from typing import Optional

import numpy as np
import pandas as pd

from ... import opcodes, options, get_context
from ...core import recursive_tile, OutputType
from ...core.operand import OperandStage
from ...serialization.serializables import Int64Field, ReferenceField
from ...utils import ceildiv, has_unknown_shape
from ..datasource import tensor as astensor
from ..operands import TensorOperandMixin, TensorMapReduceOperand


class TensorBinCount(TensorMapReduceOperand, TensorOperandMixin):
    _op_type_ = opcodes.BINCOUNT

    weights = ReferenceField("weights", default=None)
    minlength: Optional[int] = Int64Field("minlength", default=0)
    chunk_size_limit: int = Int64Field("chunk_size_limit")

    chunk_count: Optional[int] = Int64Field("chunk_count")
    tileable_right_bound: Optional[int] = Int64Field("tileable_right_bound")

    def __call__(self, x, weights=None):
        inputs = [x]
        self.weights = weights
        dtype = np.dtype(np.int_)
        if weights is not None:
            inputs.append(weights)
            dtype = weights.dtype
        return self.new_tensor(inputs, dtype=dtype, shape=(np.nan,))

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if len(inputs) > 1:
            self.weights = inputs[1]

    @classmethod
    def _tile_single(cls, op: "TensorBinCount"):
        out = op.outputs[0]
        new_chunk_op = op.copy().reset_key()
        chunk_inputs = [op.inputs[0].chunks[0]]
        if op.weights is not None:
            chunk_inputs.append(op.weights.chunks[0])
        new_chunk = new_chunk_op.new_chunk(chunk_inputs, index=(0,), **out.params)

        new_op = op.copy().reset_key()
        return new_op.new_tileables(
            op.inputs, chunks=[new_chunk], nsplits=((np.nan,),), **out.params
        )

    @classmethod
    def tile(cls, op: "TensorBinCount"):
        from ...dataframe.operands import DataFrameShuffleProxy
        from ...dataframe.utils import parse_index

        if has_unknown_shape(*op.inputs):
            yield

        ctx = get_context()
        a = op.inputs[0]
        out = op.outputs[0]

        if op.weights is not None and a.shape != op.weights.shape:
            raise ValueError("The weights and list don't have the same length.")

        input_max = yield from recursive_tile(a.max())
        yield input_max.chunks
        [max_val] = ctx.get_chunks_result([input_max.chunks[0].key])
        tileable_right_bound = max(op.minlength, int(max_val) + 1)

        chunk_count = max(1, ceildiv(tileable_right_bound, op.chunk_size_limit))

        if (
            len(op.inputs[0].chunks) == 1
            and (op.weights is None or len(op.weights.chunks) == 1)
            and chunk_count == 1
        ):
            return cls._tile_single(op)

        if op.weights is not None:
            weights = yield from recursive_tile(op.weights.rechunk(a.nsplits))
            weights_chunks = weights.chunks
        else:
            weights_chunks = itertools.repeat(None)

        map_chunks = []
        for a_chunk, weights_chunk in zip(a.chunks, weights_chunks):
            new_op = op.copy().reset_key()
            new_op.chunk_count = chunk_count
            new_op.tileable_right_bound = tileable_right_bound
            new_op.stage = OperandStage.map
            new_op._output_types = [OutputType.series]

            inputs = [a_chunk]
            if weights_chunk is not None:
                inputs.append(weights_chunk)
            map_chunks.append(
                new_op.new_chunk(
                    inputs,
                    dtype=out.dtype,
                    shape=(np.nan,),
                    index=a_chunk.index,
                    index_value=parse_index(pd.Index([0], dtype=np.int64), a_chunk.key),
                )
            )

        shuffle_op = DataFrameShuffleProxy(output_types=[OutputType.tensor]).new_chunk(
            map_chunks, dtype=out.dtype, shape=()
        )

        reduce_chunks = []
        reduce_nsplits = []
        left_offset = 0
        for chunk_idx in range(chunk_count):
            right_offset = min(tileable_right_bound, left_offset + op.chunk_size_limit)

            new_op = op.copy().reset_key()
            new_op.stage = OperandStage.reduce
            new_op.chunk_count = chunk_count
            new_op.tileable_right_bound = tileable_right_bound

            reduce_chunks.append(
                new_op.new_chunk(
                    [shuffle_op],
                    dtype=out.dtype,
                    shape=(right_offset - left_offset,),
                    index=(chunk_idx,),
                )
            )
            reduce_nsplits.append(right_offset - left_offset)
            left_offset = right_offset

        new_op = op.copy().reset_key()
        params = out.params.copy()
        params["shape"] = (tileable_right_bound,)
        return new_op.new_tileables(
            op.inputs,
            chunks=reduce_chunks,
            nsplits=(tuple(reduce_nsplits),),
            **params,
        )

    @classmethod
    def _execute_map(cls, ctx, op: "TensorBinCount"):
        input_val = ctx[op.inputs[0].key]
        if op.weights is not None:
            weights_val = ctx[op.weights.key]
            df = pd.DataFrame({"data": input_val, "weights": weights_val})
            res = df.groupby("data")["weights"].sum()
        else:
            res = pd.Series(input_val).groupby(input_val).count()

        if res.index.min() < 0:
            raise ValueError("'list' argument must have no negative elements")

        left_bound = 0
        for target_idx in range(op.chunk_count):
            right_bound = res.index.searchsorted(
                (1 + target_idx) * op.chunk_size_limit, "left"
            )
            sliced = res.iloc[left_bound:right_bound]
            if len(sliced) > 0:
                ctx[op.outputs[0].key, (target_idx,)] = sliced
            left_bound = right_bound

    @classmethod
    def _execute_reduce(cls, ctx, op: "TensorBinCount"):
        out = op.outputs[0]
        input_list = []
        for input_key in op.inputs[0].op.source_keys:
            sliced = ctx.get((input_key, out.index))
            if sliced is not None:
                input_list.append(sliced)

        left_bound = op.chunk_size_limit * out.index[0]
        right_bound = min(left_bound + op.chunk_size_limit, op.tileable_right_bound)
        if not input_list:
            ctx[op.outputs[0].key] = np.zeros(right_bound - left_bound)
        else:
            res = functools.reduce(
                lambda a, b: a.add(b, fill_value=0), input_list
            ).astype(out.dtype)
            res = res.reindex(pd.RangeIndex(left_bound, right_bound), fill_value=0)
            ctx[op.outputs[0].key] = res.values

    @classmethod
    def execute(cls, ctx, op: "TensorBinCount"):
        if op.stage == OperandStage.map:
            op._execute_map(ctx, op)
        elif op.stage == OperandStage.reduce:
            op._execute_reduce(ctx, op)
        else:
            input_val = ctx[op.inputs[0].key]
            weights_val = ctx[op.weights.key] if op.weights is not None else None
            ctx[op.outputs[0].key] = np.bincount(
                input_val, weights=weights_val, minlength=op.minlength
            )


def bincount(x, weights=None, minlength=0, chunk_size_limit=None):
    """
    Count number of occurrences of each value in array of non-negative ints.

    The number of bins (of size 1) is one larger than the largest value in
    `x`. If `minlength` is specified, there will be at least this number
    of bins in the output array (though it will be longer if necessary,
    depending on the contents of `x`).
    Each bin gives the number of occurrences of its index value in `x`.
    If `weights` is specified the input array is weighted by it, i.e. if a
    value ``n`` is found at position ``i``, ``out[n] += weight[i]`` instead
    of ``out[n] += 1``.

    Parameters
    ----------
    x : tensor or array_like, 1 dimension, nonnegative ints
        Input array.
    weights : tensor or array_like, optional
        Weights, array of the same shape as `x`.
    minlength : int, optional
        A minimum number of bins for the output array.

    Returns
    -------
    out : tensor of ints
        The result of binning the input array.
        The length of `out` is equal to ``np.amax(x)+1``.

    Raises
    ------
    ValueError
        If the input is not 1-dimensional, or contains elements with negative
        values, or if `minlength` is negative.
    TypeError
        If the type of the input is float or complex.

    See Also
    --------
    histogram, digitize, unique

    Examples
    --------
    >>> import mars.tensor as mt
    >>> mt.bincount(mt.arange(5)).execute()
    array([1, 1, 1, 1, 1])
    >>> mt.bincount(mt.tensor([0, 1, 1, 3, 2, 1, 7])).execute()
    array([1, 3, 1, 1, 0, 0, 0, 1])

    The input array needs to be of integer dtype, otherwise a
    TypeError is raised:

    >>> mt.bincount(mt.arange(5, dtype=float)).execute()
    Traceback (most recent call last):
      ....execute()
    TypeError: Cannot cast array data from dtype('float64') to dtype('int64')
    according to the rule 'safe'

    A possible use of ``bincount`` is to perform sums over
    variable-size chunks of an array, using the ``weights`` keyword.

    >>> w = mt.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
    >>> x = mt.array([0, 1, 1, 2, 2, 2])
    >>> mt.bincount(x,  weights=w).execute()
    array([ 0.3,  0.7,  1.1])
    """
    x = astensor(x)
    weights = astensor(weights) if weights is not None else None

    if not np.issubdtype(x.dtype, np.int_):
        raise TypeError(f"Cannot cast array data from {x.dtype} to {np.dtype(np.int_)}")
    if x.ndim != 1:
        raise ValueError("'x' must be 1 dimension")
    if minlength < 0:
        raise ValueError("'minlength' must not be negative")

    chunk_size_limit = (
        chunk_size_limit
        if chunk_size_limit is not None
        else options.bincount.chunk_size_limit
    )
    op = TensorBinCount(minlength=minlength, chunk_size_limit=chunk_size_limit)
    return op(x, weights=weights)
