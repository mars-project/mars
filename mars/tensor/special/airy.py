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
from .core import TensorTupleOp, _register_special_op


@_register_special_op
class TensorAiry(TensorTupleOp):
    _func_name = "airy"
    _n_outputs = 4


@implement_scipy(spspecial.airy)
@infer_dtype(spspecial.airy, multi_outputs=True)
def airy(z, out=None, **kwargs):
    op = TensorAiry(**kwargs)
    return op(z, out=out)


@_register_special_op
class TensorAirye(TensorTupleOp):
    _func_name = "airye"
    _n_outputs = 4


@implement_scipy(spspecial.airye)
@infer_dtype(spspecial.airye, multi_outputs=True)
def airye(z, out=None, **kwargs):
    op = TensorAirye(**kwargs)
    return op(z, out=out)


@_register_special_op
class TensorItairy(TensorTupleOp):
    _func_name = "itairy"
    _n_outputs = 4


@implement_scipy(spspecial.itairy)
@infer_dtype(spspecial.itairy, multi_outputs=True)
def itairy(x, out=None, **kwargs):
    op = TensorItairy(**kwargs)
    return op(x, out=out)
