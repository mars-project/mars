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

from .result_type import result_type
from .astype import TensorAstype
from .copyto import copyto, TensorCopyTo
from .transpose import transpose, TensorTranspose
from .where import where, TensorWhere
from .broadcast_to import broadcast_to, TensorBroadcastTo
from .broadcast_arrays import broadcast_arrays
from .expand_dims import expand_dims
from .rollaxis import rollaxis
from .swapaxes import swapaxes, TensorSwapAxes
from .moveaxis import moveaxis
from .ravel import ravel
from .atleast_1d import atleast_1d
from .atleast_2d import atleast_2d
from .atleast_3d import atleast_3d
from .argwhere import argwhere, TensorArgwhere
from .array_split import array_split
from .split import split, TensorSplit
from .hsplit import hsplit
from .vsplit import vsplit
from .dsplit import dsplit
from .roll import roll
from .squeeze import squeeze, TensorSqueeze
from .ptp import ptp
from .diff import diff
from .ediff1d import ediff1d
from .digitize import digitize, TensorDigitize
from .average import average
from .cov import cov
from .corrcoef import corrcoef
from .flip import flip
from .flipud import flipud
from .fliplr import fliplr
from .repeat import repeat, TensorRepeat
from .tile import tile
from .isin import isin, TensorIsIn


def _install():
    from ...core import Tensor, TensorData
    from .astype import _astype

    setattr(Tensor, 'astype', _astype)
    setattr(Tensor, 'swapaxes', swapaxes)
    setattr(Tensor, 'squeeze', squeeze)
    setattr(Tensor, 'repeat', repeat)
    setattr(TensorData, 'astype', _astype)
    setattr(TensorData, 'swapaxes', swapaxes)
    setattr(TensorData, 'squeeze', squeeze)
    setattr(TensorData, 'repeat', repeat)


_install()
del _install
