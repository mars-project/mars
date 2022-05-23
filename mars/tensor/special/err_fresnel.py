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
class TensorErf(TensorSpecialUnaryOp):
    _func_name = "erf"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorErfc(TensorSpecialUnaryOp):
    _func_name = "erfc"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorErfcx(TensorSpecialUnaryOp):
    _func_name = "erfcx"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorErfi(TensorSpecialUnaryOp):
    _func_name = "erfi"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorErfinv(TensorSpecialUnaryOp):
    _func_name = "erfinv"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorErfcinv(TensorSpecialUnaryOp):
    _func_name = "erfcinv"


@implement_scipy(spspecial.erf)
@infer_dtype(spspecial.erf)
def erf(x, out=None, where=None, **kwargs):
    op = TensorErf(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.erfc)
@infer_dtype(spspecial.erfc)
def erfc(x, out=None, where=None, **kwargs):
    op = TensorErfc(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.erfcx)
@infer_dtype(spspecial.erfcx)
def erfcx(x, out=None, where=None, **kwargs):
    op = TensorErfcx(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.erfi)
@infer_dtype(spspecial.erfi)
def erfi(x, out=None, where=None, **kwargs):
    op = TensorErfi(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.erfinv)
@infer_dtype(spspecial.erfinv)
def erfinv(x, out=None, where=None, **kwargs):
    op = TensorErfinv(**kwargs)
    return op(x, out=out, where=where)


@implement_scipy(spspecial.erfcinv)
@infer_dtype(spspecial.erfcinv)
def erfcinv(x, out=None, where=None, **kwargs):
    op = TensorErfcinv(**kwargs)
    return op(x, out=out, where=where)
