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
from .core import TensorSpecialBinOp, _register_special_op


@_register_special_op
@arithmetic_operand(sparse_mode='binary_and')
class TensorJV(TensorSpecialBinOp):
    _func_name = 'jv'


@implement_scipy(spspecial.jv)
@infer_dtype(spspecial.jv)
def jv(v, z, **kwargs):
    op = TensorJV(**kwargs)
    return op(v, z)


@_register_special_op
@arithmetic_operand(sparse_mode='binary_and')
class TensorJVE(TensorSpecialBinOp):
    _func_name = 'jve'


@implement_scipy(spspecial.jve)
@infer_dtype(spspecial.jve)
def jve(v, z, **kwargs):
    op = TensorJVE(**kwargs)
    return op(v, z)


@_register_special_op
@arithmetic_operand(sparse_mode='binary_and')
class TensorYN(TensorSpecialBinOp):
    _func_name = 'yn'


@implement_scipy(spspecial.yn)
@infer_dtype(spspecial.yn)
def yn(n, x, **kwargs):
    op = TensorYN(**kwargs)
    return op(n, x)


@_register_special_op
@arithmetic_operand(sparse_mode='binary_and')
class TensorYV(TensorSpecialBinOp):
    _func_name = 'yv'


@implement_scipy(spspecial.yv)
@infer_dtype(spspecial.yv)
def yv(v, z, **kwargs):
    op = TensorYV(**kwargs)
    return op(v, z)


@_register_special_op
@arithmetic_operand(sparse_mode='binary_and')
class TensorYVE(TensorSpecialBinOp):
    _func_name = 'yve'


@implement_scipy(spspecial.yve)
@infer_dtype(spspecial.yve)
def yve(v, z, **kwargs):
    op = TensorYVE(**kwargs)
    return op(v, z)


@_register_special_op
@arithmetic_operand(sparse_mode='binary_and')
class TensorKN(TensorSpecialBinOp):
    _func_name = 'kn'


@implement_scipy(spspecial.kn)
@infer_dtype(spspecial.kn)
def kn(n, x, **kwargs):
    op = TensorKN(**kwargs)
    return op(n, x)


@_register_special_op
@arithmetic_operand(sparse_mode='binary_and')
class TensorKV(TensorSpecialBinOp):
    _func_name = 'kv'


@implement_scipy(spspecial.kv)
@infer_dtype(spspecial.kv)
def kv(v, z, **kwargs):
    op = TensorKV(**kwargs)
    return op(v, z)


@_register_special_op
@arithmetic_operand(sparse_mode='binary_and')
class TensorKVE(TensorSpecialBinOp):
    _func_name = 'kve'


@implement_scipy(spspecial.kve)
@infer_dtype(spspecial.kve)
def kve(v, z, **kwargs):
    op = TensorKVE(**kwargs)
    return op(v, z)


@_register_special_op
@arithmetic_operand(sparse_mode='binary_and')
class TensorIV(TensorSpecialBinOp):
    _func_name = 'iv'


@implement_scipy(spspecial.iv)
@infer_dtype(spspecial.iv)
def iv(v, z, **kwargs):
    op = TensorIV(**kwargs)
    return op(v, z)


@_register_special_op
@arithmetic_operand(sparse_mode='binary_and')
class TensorIVE(TensorSpecialBinOp):
    _func_name = 'ive'


@implement_scipy(spspecial.ive)
@infer_dtype(spspecial.ive)
def ive(v, z, **kwargs):
    op = TensorIVE(**kwargs)
    return op(v, z)


@_register_special_op
@arithmetic_operand(sparse_mode='binary_and')
class TensorHankel1(TensorSpecialBinOp):
    _func_name = 'hankel1'


@implement_scipy(spspecial.hankel1)
@infer_dtype(spspecial.hankel1)
def hankel1(v, z, **kwargs):
    op = TensorHankel1(**kwargs)
    return op(v, z)


@_register_special_op
@arithmetic_operand(sparse_mode='binary_and')
class TensorHankel1e(TensorSpecialBinOp):
    _func_name = 'hankel1e'


@implement_scipy(spspecial.hankel1e)
@infer_dtype(spspecial.hankel1e)
def hankel1e(v, z, **kwargs):
    op = TensorHankel1e(**kwargs)
    return op(v, z)


@_register_special_op
@arithmetic_operand(sparse_mode='binary_and')
class TensorHankel2(TensorSpecialBinOp):
    _func_name = 'hankel2'


@implement_scipy(spspecial.hankel2)
@infer_dtype(spspecial.hankel2)
def hankel2(v, z, **kwargs):
    op = TensorHankel2(**kwargs)
    return op(v, z)


@_register_special_op
@arithmetic_operand(sparse_mode='binary_and')
class TensorHankel2e(TensorSpecialBinOp):
    _func_name = 'hankel2e'


@implement_scipy(spspecial.hankel2e)
@infer_dtype(spspecial.hankel2e)
def hankel2e(v, z, **kwargs):
    op = TensorHankel2e(**kwargs)
    return op(v, z)
