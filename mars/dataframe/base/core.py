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

from ...serialize import KeyField
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..core import DATAFRAME_TYPE, SERIES_TYPE


class DataFrameDeviceConversionBase(DataFrameOperand, DataFrameOperandMixin):
    _input = KeyField('input')

    @property
    def input(self):
        return self._input

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = inputs[0]

    def __call__(self, obj):
        if isinstance(obj, DATAFRAME_TYPE):
            self._object_type = ObjectType.dataframe
            return self.new_dataframe([obj], shape=obj.shape, dtypes=obj.dtypes,
                                      index_value=obj.index_value,
                                      columns_value=obj.columns_value)
        else:
            assert isinstance(obj, SERIES_TYPE)
            self._object_type = ObjectType.series
            return self.new_series([obj], shape=obj.shape, dtype=obj.dtype,
                                   index_value=obj.index_value, name=obj.name)

    @classmethod
    def tile(cls, op):
        out_chunks = []
        for c in op.input.chunks:
            chunk_op = op.copy().reset_key()
            out_chunk = chunk_op.new_chunk([c], **c.params)
            out_chunks.append(out_chunk)

        new_op = op.copy().reset_key()
        out = op.outputs[0]
        return new_op.new_dataframes(op.inputs, chunks=out_chunks,
                                     nsplits=op.inputs[0].nsplits,
                                     **out.params)
