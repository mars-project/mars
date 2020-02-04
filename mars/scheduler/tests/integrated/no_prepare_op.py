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

import numpy as np

from mars.tensor.arithmetic import TensorAdd


class NoPrepareOperand(TensorAdd):
    _op_type_ = 9870104312

    @classmethod
    def execute(cls, ctx, op):
        input_keys = [c.key for c in op.inputs]
        has_all_data = all(k in ctx for k in input_keys)
        if has_all_data:
            raise ValueError('Unexpected behavior')
        ctx[op.outputs[0].key] = np.array((len(op.inputs), 1))
