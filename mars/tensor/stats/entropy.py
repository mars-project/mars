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

from math import log

try:
    from scipy.stats import entropy as sp_entropy
except ImportError:
    sp_entropy = None

from ... import tensor as mt
from ...tensor import special as mt_special
from ..core import TENSOR_TYPE
from ..datasource import tensor as astensor
from ..utils import implement_scipy


@implement_scipy(sp_entropy)
def entropy(pk, qk=None, base=None):
    pk = astensor(pk)
    pk = 1.0 * pk / mt.sum(pk, axis=0)
    if qk is None:
        vec = mt_special.entr(pk)
    else:
        qk = astensor(qk)
        if len(qk) != len(pk):
            raise ValueError("qk and pk must have same length.")
        qk = 1.0 * qk / mt.sum(qk, axis=0)
        vec = mt_special.rel_entr(pk, qk)
    S = mt.sum(vec, axis=0)
    if base is not None:
        if isinstance(base, TENSOR_TYPE):
            S /= mt.log(base)
        else:
            S /= log(base)
    return S
