# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

from mars.core import OutputType
from mars.core.operand import Operand
from mars.serialization.serializables import (
    Int64Field,
    Float64Field,
    ListField,
    DataTypeField,
    SeriesField,
    NDArrayField,
    StringField,
    FieldTypes,
)
from mars.tensor.operands import TensorOperandMixin
from mars.utils import tokenize


class MockOperand(Operand, TensorOperandMixin):
    _op_code_ = 102345

    str_field = StringField("str_field")
    int_field = Int64Field("int_field")
    float_field = Float64Field("float_field")
    dtype_field = DataTypeField("dtype_field")
    series_field = SeriesField("series_field")
    ndarray_field = NDArrayField("ndarray_field")
    int_list_field = ListField("int_list_field", field_type=FieldTypes.int64)
    float_list_field = ListField("float_list_field", field_type=FieldTypes.float64)
    str_list_field = ListField("str_list_field", field_type=FieldTypes.string)


class TokenizeOperandSuite:
    def setup(self):
        chunks = []
        for idx in range(1000):
            op = MockOperand(
                str_field="abcd" * 1024,
                int_field=idx,
                float_field=float(idx) * 1.42,
                dtype_field=np.dtype("<M8"),
                series_field=pd.Series([np.dtype(int)] * 1024, name="dtype"),
                ndarray_field=np.random.rand(1000),
                int_list_field=np.random.randint(0, 1000, size=(1000,)).tolist(),
                float_list_field=np.random.rand(1000).tolist(),
                str_list_field=[str(i * 2.8571) for i in range(100)],
            )
            chunks.append(op.new_chunk([], output_type=OutputType.tensor))
        self.test_data = chunks

    def time_tokenize(self):
        tokenize(self.test_data)


if __name__ == "__main__":
    suite = TokenizeOperandSuite()
    suite.setup()
    suite.time_tokenize()
