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

from ... import opcodes
from ...context import get_context
from ...serialize import KeyField, Float64Field, Int64Field, BoolField
from ...tiles import TilesError
from ...utils import check_chunks_unknown_shape, ceildiv
from ..core import INDEX_TYPE
from ..initializer import DataFrame as asdataframe, Series as asseries, Index as asindex
from ..operands import DataFrameOperand, DataFrameOperandMixin


class DataFrameRebalance(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.REBALANCE

    _input = KeyField('input')
    _factor = Float64Field('factor')
    _num_partitions = Int64Field('num_partitions')
    _reassign_worker = BoolField('reassign_worker')

    def __init__(self, input=None, factor=None,  # pylint: disable=redefined-builtin
                 num_partitions=None, reassign_worker=None, output_types=None, **kw):
        super().__init__(_input=input, _factor=factor, _num_partitions=num_partitions,
                         _output_types=output_types, _reassign_worker=reassign_worker, **kw)

    @property
    def input(self):
        return self._input

    @property
    def factor(self):
        return self._factor

    @property
    def num_partitions(self):
        return self._num_partitions

    @property
    def reassign_worker(self):
        return self._reassign_worker

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, df_or_series):
        self._output_types = df_or_series.op.output_types
        return self.new_tileable([df_or_series], kws=[df_or_series.params])

    @classmethod
    def tile(cls, op: "DataFrameRebalance"):
        in_obj = op.input
        if isinstance(in_obj, INDEX_TYPE):
            convert = asindex
        else:
            convert = asdataframe if in_obj.ndim == 2 else asseries
        in_obj = convert(in_obj)

        ctx = get_context()

        if ctx is None and op.factor is not None:
            return [in_obj]

        check_chunks_unknown_shape([in_obj], TilesError)

        size = len(in_obj)
        if op.factor is not None:
            cluster_cpu_count = ctx.get_total_ncores()
            assert cluster_cpu_count > 0
            expect_n_chunk = int(cluster_cpu_count * op.factor)
        else:
            expect_n_chunk = op.num_partitions

        expect_chunk_size = max(ceildiv(size, expect_n_chunk), 1)
        return [in_obj.rechunk(
            {0: expect_chunk_size}, reassign_worker=op.reassign_worker)._inplace_tile()]


def rebalance(df_or_series, factor=None, num_partitions=None, reassign_worker=True):
    """
    Make Data more balanced across entire cluster.

    Parameters
    ----------
    factor : float
        Specified so that number of chunks after balance is
        total CPU count of cluster * factor.
    num_partitions: int
        Specified so the number of chunks are at most
        num_partitions.
    reassign_worker: bool
        If True, workers will be reassigned.

    Returns
    -------
    Series or DataFrame
        Result of DataFrame or Series after rebalanced.
    """
    if num_partitions is None:
        factor = factor if factor is not None else 1.2

    op = DataFrameRebalance(input=df_or_series, factor=factor, num_partitions=num_partitions,
                            reassign_worker=reassign_worker)
    return op(df_or_series)
