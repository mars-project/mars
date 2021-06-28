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

import numpy as np
try:
    import scipy.sparse as sps
    import scipy.sparse.linalg as splinalg
except ImportError:  # pragma: no cover
    sps = None
    splinalg = None

from ...utils import lazy_import

splinalg = splinalg
cp = lazy_import('cupy', globals=globals(), rename='cp')
cps = lazy_import('cupy.sparse', globals=globals(), rename='cps')


def issparse(x):
    if cps and cps.issparse(x):
        # is cupy.sparse
        return True
    if sps and sps.issparse(x):
        # is scipy.sparse
        return True
    if np and isinstance(x, np.ndarray):
        return False
    if cp and isinstance(x, cp.ndarray):
        return False

    from .array import SparseNDArray
    return isinstance(x, SparseNDArray)


def is_sparse_or_dense(x):
    if issparse(x):
        return True
    m = get_array_module(x)
    if m.isscalar(x):
        return True
    return isinstance(x, m.ndarray)


def get_dense_module(x):
    from .array import SparseNDArray

    if cp:
        if isinstance(x, SparseNDArray):
            return get_array_module(x.raw)
        return get_array_module(x)

    return np


def get_array_module(x):
    if cp:
        return cp.get_array_module(x)
    return np


def get_sparse_module(x):
    m = get_array_module(x)
    if m is np:
        return sps
    return cps


def is_cupy(x):
    return get_array_module(x) is cp


def naked(x):
    if hasattr(x, 'spmatrix'):
        return x.spmatrix
    if not is_sparse_or_dense(x):
        raise TypeError('only sparse matrix or ndarray accepted')
    return x
