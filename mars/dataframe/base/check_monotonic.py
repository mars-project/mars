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

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import OutputType
from ...core.operand import OperandStage
from ...serialization.serializables import BoolField
from ...tensor.core import TensorOrder
from ...tensor.merge import TensorConcatenate
from ..operands import DataFrameOperand, DataFrameOperandMixin


class DataFrameCheckMonotonic(DataFrameOperand, DataFrameOperandMixin):
    _op_code_ = opcodes.CHECK_MONOTONIC

    # 0 - increasing, 1 - decreasing
    _decreasing = BoolField('decreasing')
    _strict = BoolField('strict')

    def __init__(self, decreasing=None, strict=None, output_types=None, **kw):
        super().__init__(_decreasing=decreasing, _strict=strict,
                         _output_types=output_types, **kw)

    @property
    def decreasing(self):
        return self._decreasing

    @property
    def strict(self):
        return self._strict

    def __call__(self, df_obj):
        self._output_types = [OutputType.scalar]
        return self.new_tileable([df_obj], shape=(), dtype=np.dtype(bool))

    @classmethod
    def tile(cls, op: 'DataFrameCheckMonotonic'):
        map_chunks = []
        for c in op.inputs[0].chunks:
            new_op = DataFrameCheckMonotonic(
                decreasing=op.decreasing, strict=op.strict, stage=OperandStage.map,
                output_types=[OutputType.series], order=TensorOrder.C_ORDER)
            map_chunks.append(new_op.new_chunk([c], shape=(2,), dtype=np.dtype(bool)))

        concat_op = TensorConcatenate(axis=0, dtype=np.dtype(bool))
        concat_r_chunk = concat_op.new_chunk(
            map_chunks, shape=(len(map_chunks),), index=(0, 0), order=TensorOrder.C_ORDER)

        new_op = DataFrameCheckMonotonic(
            decreasing=op.decreasing, strict=op.strict, stage=OperandStage.reduce,
            output_types=[OutputType.scalar], order=TensorOrder.C_ORDER)
        r_chunk = new_op.new_chunk(
            [concat_r_chunk], shape=(), order=TensorOrder.C_ORDER, dtype=np.dtype(bool))

        new_op = op.copy().reset_key()
        params = op.outputs[0].params
        params['chunks'] = [r_chunk]
        params['nsplits'] = ()
        return new_op.new_tileables(op.inputs, **params)

    @classmethod
    def execute(cls, ctx, op: 'DataFrameCheckMonotonic'):
        in_data = ctx[op.inputs[0].key]
        if op.stage == OperandStage.map:
            is_mono = in_data.is_monotonic_increasing \
                if not op.decreasing else in_data.is_monotonic_decreasing
            if op.strict and is_mono:
                is_mono = in_data.is_unique

            if isinstance(in_data, pd.Index):
                edge_array = np.array([in_data[0], in_data[-1]])
            else:
                edge_array = np.array([in_data.iloc[0], in_data.iloc[-1]])

            ctx[op.outputs[0].key] = (
                np.array([is_mono]), edge_array,
            )
        else:
            in_series = pd.Series(in_data[1])
            is_edge_mono = in_series.is_monotonic_increasing \
                if not op.decreasing else in_series.is_monotonic_decreasing
            if op.strict and is_edge_mono:
                is_edge_mono = in_series.is_unique
            ctx[op.outputs[0].key] = in_data[0].all() and is_edge_mono


def check_monotonic(series_or_index, decreasing=False, strict=False):
    """
    Check if values in the object are monotonic increasing
    or decreasing.

    Parameters
    ----------
    decreasing : bool
        If True, check if values are monotonic decreasing,
        otherwise check if values are monotonic increasing
    strict : bool
        If True, values need to be unique to get a positive
        result

    Returns
    -------
    Scalar
    """
    op = DataFrameCheckMonotonic(decreasing=decreasing, strict=strict)
    return op(series_or_index)


def is_monotonic(series_or_index):
    """
    Return boolean scalar if values in the object are
    monotonic_increasing.

    Returns
    -------
    Scalar
    """
    return check_monotonic(series_or_index, decreasing=False, strict=False)


is_monotonic_increasing = is_monotonic


def is_monotonic_decreasing(series_or_index):
    """
    Return boolean scalar if values in the object are
    monotonic_decreasing.

    Returns
    -------
    Scalar
    """
    return check_monotonic(series_or_index, decreasing=True, strict=False)
