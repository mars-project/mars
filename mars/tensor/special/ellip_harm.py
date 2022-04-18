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

from ..utils import infer_dtype, implement_scipy
from .core import _register_special_op, TensorSpecialMultiOp


@_register_special_op
class TensorEllipHarm(TensorSpecialMultiOp):
    _ARG_COUNT = 5
    _func_name = "ellip_harm"


@implement_scipy(spspecial.ellip_harm)
@infer_dtype(spspecial.ellip_harm)
def ellip_harm(h2, k2, n, p, s, signm=1, signn=1, **kwargs):
    op = TensorEllipHarm(**kwargs)
    return op(h2, k2, n, p, s, signm, signn)


@_register_special_op
class TensorEllipHarm2(TensorSpecialMultiOp):
    _ARG_COUNT = 5
    _func_name = "ellip_harm_2"


@implement_scipy(spspecial.ellip_harm_2)
@infer_dtype(spspecial.ellip_harm_2)
def ellip_harm_2(h2, k2, n, p, s, **kwargs):
    op = TensorEllipHarm2(**kwargs)
    return op(h2, k2, n, p, s)


@_register_special_op
class TensorEllipNormal(TensorSpecialMultiOp):
    _ARG_COUNT = 4
    _func_name = "ellip_normal"


@implement_scipy(spspecial.ellip_normal)
@infer_dtype(spspecial.ellip_normal)
def ellip_normal(h2, k2, n, p, **kwargs):
    op = TensorEllipNormal(**kwargs)
    return op(h2, k2, n, p)
