# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
from .ndim import ndim
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
from .flatten import flatten
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
from .diff import diff
from .ediff1d import ediff1d
from .flip import flip
from .flipud import flipud
from .fliplr import fliplr
from .repeat import repeat, TensorRepeat
from .tile import tile
from .isin import isin, TensorIsIn
from .searchsorted import searchsorted, TensorSearchsorted
from .unique import unique
from .sort import sort
from .argsort import argsort
from .partition import partition
from .argpartition import argpartition
from .topk import topk
from .argtopk import argtopk
from .copy import copy
from .trapz import trapz
from .shape import shape
from .to_gpu import to_gpu
from .to_cpu import to_cpu


def _install():
    from ..core import Tensor, TensorData
    from .astype import _astype

    for cls in (Tensor, TensorData):
        setattr(cls, 'astype', _astype)
        setattr(cls, 'swapaxes', swapaxes)
        setattr(cls, 'squeeze', squeeze)
        setattr(cls, 'repeat', repeat)
        setattr(cls, 'ravel', ravel)
        setattr(cls, 'flatten', flatten)
        setattr(cls, 'to_gpu', to_gpu)
        setattr(cls, 'to_cpu', to_cpu)


_install()
del _install
