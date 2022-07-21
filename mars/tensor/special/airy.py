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
from .core import TensorSpecialUnaryOp, _register_special_op


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorAiry(TensorSpecialUnaryOp):
    _func_name = "airy"


@implement_scipy(spspecial.airy)
@infer_dtype(spspecial.airy, multi_outputs=True)
def airy(z, **kwargs):
    op = TensorAiry(**kwargs)
    return op(z)


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorAirye(TensorSpecialUnaryOp):
    _func_name = "airye"


@implement_scipy(spspecial.airye)
@infer_dtype(spspecial.airye, multi_outputs=True)
def airye(z, **kwargs):
    op = TensorAirye(**kwargs)
    return op(z)


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorAiZeros(TensorSpecialUnaryOp):
    _func_name = "ai_zeros"


@implement_scipy(spspecial.ai_zeros)
@infer_dtype(spspecial.ai_zeros, multi_outputs=True)
def ai_zeros(nt, **kwargs):
    op = TensorAiZeros(**kwargs)
    return op(nt)


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorBiZeros(TensorSpecialUnaryOp):
    _func_name = "bi_zeros"


@implement_scipy(spspecial.bi_zeros)
@infer_dtype(spspecial.bi_zeros, multi_outputs=True)
def bi_zeros(nt, **kwargs):
    op = TensorBiZeros(**kwargs)
    return op(nt)


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorItairy(TensorSpecialUnaryOp):
    _func_name = "itairy"


@implement_scipy(spspecial.itairy)
@infer_dtype(spspecial.itairy, multi_outputs=True)
def itairy(x, **kwargs):
    op = TensorItairy(**kwargs)
    return op(x)