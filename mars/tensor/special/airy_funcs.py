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

import scipy.special as spspecial

from ..arithmetic.utils import arithmetic_operand
from ..utils import infer_dtype, implement_scipy
from .core import (
    TensorSpecialUnaryOp,
    _register_special_op,
)


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorAiry(TensorSpecialUnaryOp):
    _func_name = "airy"


@implement_scipy(spspecial.airy)
@infer_dtype(spspecial.airy)
def airy(x, **kwargs):
    op = TensorAiry(**kwargs)
    return op(x)


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorAirye(TensorSpecialUnaryOp):
    _func_name = "airye"


@implement_scipy(spspecial.airye)
@infer_dtype(spspecial.airye)
def airye(x, **kwargs):
    op = TensorAirye(**kwargs)
    return op(x)
