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
from ...serialization.serializables import KeyField, Float64Field, Int64Field
from ...tensor.base.rebalance import RebalanceMixin
from ..core import INDEX_TYPE
from ..initializer import DataFrame as asdataframe, Series as asseries, Index as asindex
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import validate_axis


class DataFrameRebalance(RebalanceMixin, DataFrameOperandMixin, DataFrameOperand):
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
        in_obj = self.input
        if isinstance(in_obj, INDEX_TYPE):
            convert = asindex
        else:
            convert = asdataframe if in_obj.ndim == 2 else asseries
        return convert(in_obj)


def rebalance(df_or_series, factor=None, axis=0, num_partitions=None, reassign_worker=True):
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
    axis = validate_axis(axis, df_or_series)
    if num_partitions is None:
        factor = factor if factor is not None else 1.2

    op = DataFrameRebalance(input=df_or_series, factor=factor, axis=axis,
                            num_partitions=num_partitions, reassign_worker=reassign_worker)
    return op(df_or_series)
