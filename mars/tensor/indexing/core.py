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


import itertools
from numbers import Integral

import numpy as np

from ..core import TENSOR_TYPE, TENSOR_CHUNK_TYPE
from ..datasource import tensor as astensor
from ..utils import calc_sliced_size, broadcast_shape, replace_ellipsis, index_ndim


_INDEX_ERROR_MSG = 'only integers, slices (`:`), ellipsis (`...`), ' \
                   'numpy.newaxis (`None`) and integer or boolean arrays are valid indices'


def calc_shape(tensor_shape, index):
    shape = []
    in_axis = 0
    out_axis = 0
    fancy_index = None
    fancy_index_shapes = []
    for ind in index:
        if isinstance(ind, TENSOR_TYPE + TENSOR_CHUNK_TYPE + (np.ndarray,)) and ind.dtype == np.bool_:
            # bool
            shape.append(np.nan if not isinstance(ind, np.ndarray) else int(ind.sum()))
            for i, t_size, size in zip(itertools.count(0), ind.shape, tensor_shape[in_axis:ind.ndim + in_axis]):
                if not np.isnan(t_size) and not np.isnan(size) and t_size != size:
                    raise IndexError(
                        f'boolean index did not match indexed array along dimension {in_axis + i}; '
                        f'dimension is {size} but corresponding boolean dimension is {t_size}'
                    )
            in_axis += ind.ndim
            out_axis += 1
        elif isinstance(ind, TENSOR_TYPE + TENSOR_CHUNK_TYPE + (np.ndarray,)):
            first_fancy_index = False
            if fancy_index is None:
                first_fancy_index = True
                fancy_index = out_axis
            if isinstance(ind, np.ndarray) and np.any(ind >= tensor_shape[in_axis]):
                out_of_range_index = next(i for i in ind.flat if i >= tensor_shape[in_axis])
                raise IndexError(f'IndexError: index {out_of_range_index} is out of '
                                 f'bounds with size {tensor_shape[in_axis]}')
            fancy_index_shapes.append(ind.shape)
            in_axis += 1
            if first_fancy_index:
                out_axis += ind.ndim
        elif isinstance(ind, slice):
            if np.isnan(tensor_shape[in_axis]):
                shape.append(np.nan)
            else:
                shape.append(calc_sliced_size(tensor_shape[in_axis], ind))
            in_axis += 1
            out_axis += 1
        elif isinstance(ind, Integral):
            size = tensor_shape[in_axis]
            if not np.isnan(size) and ind >= size:
                raise IndexError(f'index {ind} is out of bounds for axis {in_axis} with size {size}')
            in_axis += 1
        else:
            assert ind is None
            shape.append(1)

    if fancy_index is not None:
        try:
            if any(np.isnan(np.prod(s)) for s in fancy_index_shapes):
                fancy_index_shape = (np.nan,) * len(fancy_index_shapes[0])
            else:
                fancy_index_shape = broadcast_shape(*fancy_index_shapes)
            shape = shape[:fancy_index] + list(fancy_index_shape) + shape[fancy_index:]
        except ValueError:
            raise IndexError(
                'shape mismatch: indexing arrays could not be broadcast together '
                'with shapes {0}'.format(' '.join(str(s) for s in fancy_index_shapes)))

    return shape


def preprocess_index(index, convert_bool_to_fancy=None):
    from .nonzero import nonzero

    inds = []
    fancy_indexes = []
    bool_indexes = []
    all_fancy_index_ndarray = True
    all_bool_index_ndarray = True
    for j, ind in enumerate(index):
        if isinstance(ind, (list, np.ndarray) + TENSOR_TYPE):
            if not isinstance(ind, TENSOR_TYPE):
                ind = np.array(ind)
            if ind.dtype.kind not in 'biu':
                raise IndexError(_INDEX_ERROR_MSG)
            if ind.dtype.kind == 'b':
                # bool indexing
                bool_indexes.append(j)
                if not isinstance(ind, np.ndarray):
                    all_bool_index_ndarray = False
            else:
                # fancy indexing
                fancy_indexes.append(j)
                if not isinstance(ind, np.ndarray):
                    all_fancy_index_ndarray = False
        elif not isinstance(ind, (slice, Integral)) and ind is not None \
                and ind is not Ellipsis:
            raise IndexError(_INDEX_ERROR_MSG)
        inds.append(ind)

    if convert_bool_to_fancy is None:
        convert_bool_to_fancy = \
            (fancy_indexes and len(bool_indexes) > 0) or len(bool_indexes) > 1

    if not all_fancy_index_ndarray or (convert_bool_to_fancy and not all_bool_index_ndarray):
        # if not all fancy indexes are ndarray,
        # or bool indexes need to be converted to fancy indexes,
        # and not all bool indexes are ndarray,
        # we will convert all of them to Tensor
        for fancy_index in fancy_indexes:
            inds[fancy_index] = astensor(inds[fancy_index])

    # convert bool index to fancy index when any situation below meets:
    # 1. fancy indexes and bool indexes both exists
    # 2. bool indexes more than 2
    if convert_bool_to_fancy:
        default_m = None
        if len(fancy_indexes) > 0:
            default_m = np.nonzero \
                if isinstance(inds[fancy_indexes[0]], np.ndarray) \
                else nonzero
        for bool_index in bool_indexes:
            ind = inds[bool_index]
            m = default_m
            if m is None:
                m = np.nonzero if isinstance(ind, np.ndarray) else nonzero
            ind = m(ind)[0]
            inds[bool_index] = ind

    return tuple(inds)


def process_index(tensor_ndim, item, convert_bool_to_fancy=None):
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

    index = preprocess_index(
        item, convert_bool_to_fancy=convert_bool_to_fancy)
    index = replace_ellipsis(index, tensor_ndim)
    missing = tensor_ndim - sum(index_ndim(i) for i in index)
    if missing < 0:
        raise IndexError('too many indices for tensor')
    return index + (slice(None),) * missing
