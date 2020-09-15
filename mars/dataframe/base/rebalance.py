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
from ...serialize import KeyField, Float64Field
from ...tiles import TilesError
from ...utils import check_chunks_unknown_shape
from ..initializer import DataFrame as asdataframe, Series as asseries
from ..operands import DataFrameOperand, DataFrameOperandMixin


class DataFrameRebalance(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.REBALANCE

    _input = KeyField('input')
    _factor = Float64Field('factor')

    def __init__(self, input=None, factor=None, output_types=None, **kw):  # pylint: disable=redefined-builtin
        super().__init__(_input=input, _factor=factor,
                         _output_types=output_types, **kw)

    @property
    def input(self):
        return self._input

    @property
    def factor(self):
        return self._factor

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, df_or_series):
        self._output_types = df_or_series.op.output_types
        return self.new_tileable([df_or_series], kws=[df_or_series.params])

    @classmethod
    def tile(cls, op: "DataFrameRebalance"):
        df_or_series = op.input
        convert = asdataframe if df_or_series.ndim == 2 else asseries
        df_or_series = convert(df_or_series)

        ctx = get_context()
        if ctx is None:
            return [df_or_series]

        check_chunks_unknown_shape([df_or_series], TilesError)

        cluster_cpu_count = ctx.get_total_ncores()
        assert cluster_cpu_count > 0

        size = len(df_or_series)
        expect_n_chunk = int(cluster_cpu_count * op.factor)
        expect_chunk_size = max(size // expect_n_chunk, 1)
        return [df_or_series.rechunk({0: expect_chunk_size}, reassign_worker=True)._inplace_tile()]


def rebalance(df_or_series, factor=1.2):
    """
    Make Data more balanced across entire cluster.

    Parameters
    ----------
    factor : float
        Specified so that number of chunks after balance is
        total CPU count of cluster * factor.

    Returns
    -------
    Series or DataFrame
        Result of DataFrame or Series after rebalanced.
    """
    op = DataFrameRebalance(input=df_or_series, factor=factor)
    return op(df_or_series)
