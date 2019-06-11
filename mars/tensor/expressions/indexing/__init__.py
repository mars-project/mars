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

from .take import take
from .compress import compress
from .extract import extract
from .choose import choose, TensorChoose
from .unravel_index import unravel_index, TensorUnravelIndex
from .nonzero import nonzero, TensorNonzero
from .flatnonzero import flatnonzero
from .slice import TensorSlice
from .getitem import TensorIndex, FancyIndexingDistributeMap, FancyIndexingDistributeReduce, \
    FancyIndexingConcatMap, FancyIndexingConcatReduce
from .setitem import TensorIndexSetValue


def _install():
    from ...core import Tensor, TensorData
    from .getitem import _getitem
    from .setitem import _setitem

    setattr(Tensor, '__getitem__', _getitem)
    setattr(TensorData, '__getitem__', _getitem)
    setattr(Tensor, '__setitem__', _setitem)
    setattr(Tensor, 'take', take)
    setattr(Tensor, 'compress',
            lambda a, condition, axis=None: compress(condition, a, axis=axis))
    setattr(Tensor, 'choose', choose)
    setattr(Tensor, 'nonzero', nonzero)


_install()
del _install
