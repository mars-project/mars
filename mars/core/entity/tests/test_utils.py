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

from mars import tensor as mt
from mars.core import recursive_tile
from mars.tensor.operands import TensorOperand, TensorOperandMixin
from mars.utils import has_unknown_shape


class _TestOperand(TensorOperand, TensorOperandMixin):

    @classmethod
    def tile(cls, op: "_TestOperand"):
        data1, data2 = op.inputs

        data1 = mt.sort(data1)
        data2 = mt.sort(data2)
        data_all = mt.concatenate([data1, data2])
        s1 = mt.searchsorted(data1, data_all)
        s2 = mt.searchsorted(data2, data_all)
        result = yield from recursive_tile(mt.concatenate([s1, s2]))
        # data1 will be yield by s1
        assert not has_unknown_shape(data1)
        assert not has_unknown_shape(data2)
        assert not has_unknown_shape(data_all)
        return result


def test_recursive_tile(setup):
    d1 = mt.random.rand(10, chunk_size=5)
    d2 = mt.random.rand(10, chunk_size=5)
    op = _TestOperand()
    t = op.new_tensor([d1, d2], dtype=d1.dtype,
                      shape=(20,), order=d1.order)
    t.execute()
