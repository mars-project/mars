#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from .sum import sum, TensorSum
from .nansum import nansum, TensorNanSum
from .prod import prod, TensorProd
from .nanprod import nanprod, TensorNanProd
from .max import max, TensorMax
from .nanmax import nanmax, TensorNanMax
from .min import min, TensorMin
from .nanmin import nanmin, TensorNanMin
from .all import all, TensorAll
from .any import any, TensorAny
from .mean import mean, TensorMean
from .nanmean import nanmean, TensorNanMean
from .argmax import argmax, TensorArgmax
from .nanargmax import nanargmax, TensorNanArgmax
from .argmin import argmin, TensorArgmin
from .nanargmin import nanargmin, TensorNanArgmin
from .cumsum import cumsum, TensorCumsum
from .cumprod import cumprod, TensorCumprod
from .var import var, TensorVar, TensorMoment
from .std import std
from .nanvar import nanvar, TensorNanVar, TensorNanMoment
from .nanstd import nanstd
from .nancumsum import nancumsum, TensorNanCumsum
from .nancumprod import nancumprod, TensorNanCumprod
from .count_nonzero import count_nonzero, TensorCountNonzero
from .allclose import allclose
from .array_equal import array_equal


def _install():
    from ..core import Tensor, TensorData

    for cls in (Tensor, TensorData):
        setattr(cls, 'sum', sum)
        setattr(cls, 'prod', prod)
        setattr(cls, 'max', max)
        setattr(cls, 'min', min)
        setattr(cls, 'all', all)
        setattr(cls, 'any', any)
        setattr(cls, 'mean', mean)
        setattr(cls, 'argmax', argmax)
        setattr(cls, 'argmin', argmin)
        setattr(cls, 'cumsum', cumsum)
        setattr(cls, 'cumprod', cumprod)
        setattr(cls, 'var', var)
        setattr(cls, 'std', std)


_install()
del _install
