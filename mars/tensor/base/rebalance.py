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

from ... import opcodes
from ...core import recursive_tile
from ...core.context import get_context
from ...serialization.serializables import KeyField, Float64Field, Int64Field
from ...tensor.datasource import tensor as astensor
from ...utils import has_unknown_shape, ceildiv
from ..operands import TensorOperandMixin, TensorOperand


class RebalanceMixin:
    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, df_or_series):
        self._output_types = df_or_series.op.output_types
        return self.new_tileable([df_or_series], kws=[df_or_series.params])

    def _get_input_object(self):
        raise NotImplementedError

    @classmethod
    def tile(cls, op: "RebalanceMixin"):
        in_obj = op._get_input_object()
        ctx = get_context()

        if ctx is None and op.factor is not None:
            return [in_obj]

        if has_unknown_shape(in_obj):
            yield

        size = in_obj.shape[op.axis]
        if op.factor is not None:
            cluster_cpu_count = ctx.get_total_n_cpu()
            assert cluster_cpu_count > 0
            expect_n_chunk = int(cluster_cpu_count * op.factor)
        else:
            expect_n_chunk = op.num_partitions

        expect_chunk_size = max(ceildiv(size, expect_n_chunk), 1)
        r = yield from recursive_tile(
            in_obj.rechunk(
                {op.axis: expect_chunk_size},
                reassign_worker=op.reassign_worker))
        return r


class TensorRebalance(RebalanceMixin, TensorOperandMixin, TensorOperand):
    _op_type_ = opcodes.REBALANCE

    _input = KeyField('input')
    _factor = Float64Field('factor')
    _axis = Int64Field('axis')
    _num_partitions = Int64Field('num_partitions')

    def __init__(self, input=None, factor=None, axis=None,  # pylint: disable=redefined-builtin
                 num_partitions=None, output_types=None, **kw):
        super().__init__(_input=input, _factor=factor, _axis=axis, _num_partitions=num_partitions,
                         _output_types=output_types, **kw)

    @property
    def input(self):
        return self._input

    @property
    def factor(self):
        return self._factor

    @property
    def axis(self):
        return self._axis

    @property
    def num_partitions(self):
        return self._num_partitions

    def _get_input_object(self):
        return astensor(self.inputs[0])


def rebalance(tensor, factor=None, axis=0, num_partitions=None, reassign_worker=True):
    """
    Make Data more balanced across entire cluster.

    Parameters
    ----------
    factor : float
        Specified so that number of chunks after balance is
        total CPU count of cluster * factor.
    axis : int
        The axis to rebalance.
    num_partitions : int
        Specified so the number of chunks are at most
        num_partitions.
    reassign_worker : bool
        If True, workers will be reassigned.

    Returns
    -------
    Series or DataFrame
        Result of DataFrame or Series after rebalanced.
    """
    if num_partitions is None:
        factor = factor if factor is not None else 1.2

    op = TensorRebalance(input=tensor, factor=factor, axis=axis, num_partitions=num_partitions,
                         reassign_worker=reassign_worker)
    return op(tensor)
