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
    _register_special_op,
    TensorSpecialBinOp,
    TensorSpecialUnaryOp,
    TensorSpecialMultiOp,
)


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorEllipk(TensorSpecialUnaryOp):
    _func_name = "ellipk"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorEllipkm1(TensorSpecialUnaryOp):
    _func_name = "ellipkm1"


@_register_special_op
@arithmetic_operand(sparse_mode="binary_and")
class TensorEllipkinc(TensorSpecialBinOp):
    _func_name = "ellipkinc"


@_register_special_op
@arithmetic_operand(sparse_mode="unary")
class TensorEllipe(TensorSpecialUnaryOp):
    _func_name = "ellipe"


@_register_special_op
@arithmetic_operand(sparse_mode="binary_and")
class TensorEllipeinc(TensorSpecialBinOp):
    _func_name = "ellipeinc"


@_register_special_op
@arithmetic_operand(sparse_mode="binary_and")
class TensorElliprc(TensorSpecialBinOp):
    _func_name = "elliprc"


@_register_special_op
class TensorElliprd(TensorSpecialMultiOp):
    _ARG_COUNT = 3
    _func_name = "elliprd"


@_register_special_op
class TensorElliprf(TensorSpecialMultiOp):
    _ARG_COUNT = 3
    _func_name = "elliprf"


@_register_special_op
class TensorElliprg(TensorSpecialMultiOp):
    _ARG_COUNT = 3
    _func_name = "elliprg"


@_register_special_op
class TensorElliprj(TensorSpecialMultiOp):
    _ARG_COUNT = 4
    _func_name = "elliprj"


@implement_scipy(spspecial.ellipk)
@infer_dtype(spspecial.ellipk)
def ellipk(x, **kwargs):
    op = TensorEllipk(**kwargs)
    return op(x)


@implement_scipy(spspecial.ellipkm1)
@infer_dtype(spspecial.ellipkm1)
def ellipkm1(x, **kwargs):
    op = TensorEllipkm1(**kwargs)
    return op(x)


@implement_scipy(spspecial.ellipkinc)
@infer_dtype(spspecial.ellipkinc)
def ellipkinc(phi, m, **kwargs):
    op = TensorEllipkinc(**kwargs)
    return op(phi, m)


@implement_scipy(spspecial.ellipe)
@infer_dtype(spspecial.ellipe)
def ellipe(x, **kwargs):
    op = TensorEllipe(**kwargs)
    return op(x)


@implement_scipy(spspecial.ellipeinc)
@infer_dtype(spspecial.ellipeinc)
def ellipeinc(phi, m, **kwargs):
    op = TensorEllipeinc(**kwargs)
    return op(phi, m)


try:

    @implement_scipy(spspecial.elliprc)
    @infer_dtype(spspecial.elliprc)
    def elliprc(x, y, **kwargs):
        op = TensorElliprc(**kwargs)
        return op(x, y)

    @implement_scipy(spspecial.elliprd)
    @infer_dtype(spspecial.elliprd)
    def elliprd(x, y, z, **kwargs):
        op = TensorElliprd(**kwargs)
        return op(x, y, z)

    @implement_scipy(spspecial.elliprf)
    @infer_dtype(spspecial.elliprf)
    def elliprf(x, y, z, **kwargs):
        op = TensorElliprf(**kwargs)
        return op(x, y, z)

    @implement_scipy(spspecial.elliprg)
    @infer_dtype(spspecial.elliprg)
    def elliprg(x, y, z, **kwargs):
        op = TensorElliprg(**kwargs)
        return op(x, y, z)

    @implement_scipy(spspecial.elliprj)
    @infer_dtype(spspecial.elliprj)
    def elliprj(x, y, z, p, **kwargs):
        op = TensorElliprj(**kwargs)
        return op(x, y, z, p)

except AttributeError:
    # These functions are not implemented before scipy v1.8 so
    # spsecial.func may cause AttributeError
    elliprc = elliprd = elliprf = elliprg = elliprj = None
