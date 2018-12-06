#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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
from .mean import mean, TensorMean, TensorMeanChunk, TensorMeanCombine
from .nanmean import nanmean, TensorNanMean, TensorNanMeanChunk, TensorMeanCombine
from .argmax import argmax, TensorArgmax, TensorArgmaxChunk, TensorArgmaxCombine
from .nanargmax import nanargmax, TensorNanArgmax, \
    TensorNanArgmaxChunk, TensorNanArgmaxCombine
from .argmin import argmin, TensorArgmin, TensorArgminChunk, TensorArgminCombine
from .nanargmin import nanargmin, TensorNanArgmin, \
    TensorNanArgminChunk, TensorNanArgminCombine
from .cumsum import cumsum, TensorCumsum
from .cumprod import cumprod, TensorCumprod
from .var import var, TensorVar, TensorMoment, TensorMomentChunk, TensorMomentCombine
from .std import std
from .nanvar import nanvar, TensorNanVar, TensorNanMoment, \
    TensorNanMomentChunk, TensorNanMomentCombine
from .nanstd import nanstd
from .nancumsum import nancumsum, TensorNanCumsum
from .nancumprod import nancumprod, TensorNanCumprod
from .count_nonzero import count_nonzero, TensorCountNonzero
from .allclose import allclose
from .array_equal import array_equal


def _install():
    from ...core import Tensor

    setattr(Tensor, 'sum', sum)
    setattr(Tensor, 'prod', prod)
    setattr(Tensor, 'max', max)
    setattr(Tensor, 'min', min)
    setattr(Tensor, 'all', all)
    setattr(Tensor, 'any', any)
    setattr(Tensor, 'mean', mean)
    setattr(Tensor, 'argmax', argmax)
    setattr(Tensor, 'argmin', argmin)
    setattr(Tensor, 'cumsum', cumsum)
    setattr(Tensor, 'cumprod', cumprod)
    setattr(Tensor, 'var', var)
    setattr(Tensor, 'std', std)


_install()
del _install
