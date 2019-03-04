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


from .array import tensor, array, asarray, ArrayDataSource, CSRMatrixDataSource
from .scalar import scalar, Scalar
from .empty import empty, empty_like, TensorEmpty, TensorEmptyLike
from .ones import ones, ones_like, TensorOnes, TensorOnesLike
from .zeros import zeros, zeros_like, TensorZeros, TensorZerosLike
from .full import full, TensorFull
from .arange import arange, TensorArange
from .diag import diag, TensorDiag
from .diagflat import diagflat
from .eye import eye, TensorEye
from .identity import identity
from .linspace import linspace, TensorLinspace
from .meshgrid import meshgrid
from .indices import indices, TensorIndices
from .tri import triu, tril, TensorTriu, TensorTril
from .fromdense import fromdense, DenseToSparse
from .fromsparse import fromsparse, SparseToDense
from .fromtiledb import fromtiledb, TensorTileDBDataSource
