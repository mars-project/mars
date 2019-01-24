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

import itertools
from numbers import Integral

import numpy as np

from ...core import TENSOR_TYPE, CHUNK_TYPE
from ..utils import calc_sliced_size, broadcast_shape, replace_ellipsis, index_ndim
from ..datasource import tensor as astensor


_INDEX_ERROR_MSG = 'only integers, slices (`:`), ellipsis (`...`), ' \
                   'numpy.newaxis (`None`) and integer or boolean arrays are valid indices'


def get_index_and_shape(tensor_shape, index):
    shape = []
    idx = 0
    fancy_index = None
    fancy_index_shapes = []
    for ind in index:
        if isinstance(ind, (TENSOR_TYPE + CHUNK_TYPE)):
            if ind.dtype == np.bool_:
                shape.append(np.nan)
                for i, t_size, size in zip(itertools.count(0), ind.shape, tensor_shape[idx:ind.ndim + idx]):
                    if not np.isnan(t_size) and not np.isnan(size) and t_size != size:
                        raise IndexError(
                            'boolean index did not match indexed array along dimension {0}; '
                            'dimension is {1} but corresponding boolean dimension is {2}'.format(
                                idx + i, size, t_size)
                        )
                idx += ind.ndim
        elif isinstance(ind, np.ndarray):
            if fancy_index is None:
                fancy_index = idx
            fancy_index_shapes.append(ind.shape)
            idx += 1
        elif isinstance(ind, slice):
            if np.isnan(tensor_shape[idx]):
                shape.append(np.nan)
                idx += 1
            else:
                shape.append(calc_sliced_size(tensor_shape[idx], ind))
                idx += 1
        elif isinstance(ind, Integral):
            size = tensor_shape[idx]
            if not np.isnan(size) and ind >= size:
                raise IndexError('index {0} is out of bounds for axis {1} with size {2}'.format(
                    ind, idx, size
                ))
            idx += 1
        else:
            assert ind is None
            shape.append(1)

    if fancy_index is not None:
        try:
            fancy_index_shape = broadcast_shape(*fancy_index_shapes)
            shape = shape[:fancy_index] + list(fancy_index_shape) + shape[fancy_index:]
        except ValueError:
            raise IndexError(
                'shape mismatch: indexing arrays could not be broadcast together '
                'with shapes {0}'.format(' '.join(str(s) for s in fancy_index_shapes)))

    return index, shape


def preprocess_index(index):
    inds = []
    for ind in index:
        if isinstance(ind, (list, np.ndarray)):
            ind = np.array(ind)
            if ind.dtype.kind not in 'biu':
                raise IndexError(_INDEX_ERROR_MSG)
            if ind.dtype.kind == 'b':
                ind = astensor(ind)
        elif isinstance(ind, TENSOR_TYPE):
            if ind.dtype.kind not in 'biu':
                raise IndexError(_INDEX_ERROR_MSG)
        elif not isinstance(ind, (slice, Integral)) and ind is not None \
                and ind is not Ellipsis:
            raise IndexError(_INDEX_ERROR_MSG)
        inds.append(ind)

    return tuple(inds)


def process_index(tensor, item):
    if isinstance(item, list):
        arr = np.array(item)
        if arr.dtype == np.object_:
            item = tuple(item)
        elif arr.dtype.kind == 'f':
            raise IndexError(_INDEX_ERROR_MSG)
        else:
            item = (arr,)
    elif not isinstance(item, tuple):
        item = (item,)

    index = preprocess_index(item)
    index = replace_ellipsis(index, tensor.ndim)
    missing = tensor.ndim - sum(index_ndim(i) for i in index)
    if missing < 0:
        raise IndexError('too many indices for tensor')
    return index + (slice(None),) * missing
