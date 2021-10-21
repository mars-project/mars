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

from datetime import datetime
import itertools
from numbers import Integral
from typing import Optional
import uuid

import numpy as np

from ...core import tile
from ...tensor.indexing.core import process_index, calc_shape
from ...tensor.indexing.getitem import TensorIndex


def indexing_to_chunk_indices(output_chunk):
    """
    Compute input_indices and value_indices when read from or write to
    a tensor chunk.

    Parameters
    ----------
    output_chunk:
        A chunk in the output of the `__setitem__` op.

    Returns
    -------
        The indices in the input chunk, and value_indices in the value block
        that will be assigned.
    """
    input_indices = []  # index in the chunk of the mutable tensor
    value_indices = []  # index in the chunk of the assigned value
    for d, s in zip(output_chunk.op.indexes, output_chunk.op.inputs[0].shape):
        # expand the index (slice)
        idx = np.r_[slice(*d.indices(s)) if isinstance(d, slice) else d]
        input_indices.append(idx)
        if not isinstance(d, Integral):
            value_indices.append(np.arange(len(idx)))
    return input_indices, value_indices


def compute_output_of_indexing(tensor, tensor_index):
    """
    Compute the output information of `__{set,get}item__` on tensor for every chunk.
    """
    tensor_index = process_index(tensor.ndim, tensor_index)
    output_shape = calc_shape(tensor.shape, tensor_index)

    index_tensor_op = TensorIndex(
        dtype=tensor.dtype, sparse=False, indexes=list(tensor_index)
    )
    index_tensor = tile(index_tensor_op.new_tensor([tensor], shape=tuple(output_shape)))
    output_chunks = index_tensor.chunks

    nsplits_acc = [
        np.cumsum(
            (0,)
            + tuple(
                c.shape[i]
                for c in output_chunks
                if all(idx == 0 for j, idx in enumerate(c.index) if j != i)
            )
        )
        for i in range(len(output_chunks[0].shape))
    ]
    return output_shape, output_chunks, nsplits_acc


def setitem_on_chunk_to_records(nsplits_acc, output_chunk, value, ts, is_scalar):
    """
    Turns a `__setitem__` on chunk to a list of index-value records.

    Parameters
    ----------
    nsplits_acc:
        Accumulate nsplits arrays of the output tensor chunks.

    Returns
    -------
        A list of `(index, value, timestamp)`, where `index` is the in-chunk index.
    """
    input_indices, value_indices = indexing_to_chunk_indices(output_chunk)

    # normalize assigned value
    if is_scalar:
        chunk_value = value
    else:
        chunk_value_slice = tuple(
            slice(
                nsplits_acc[i][output_chunk.index[i]],
                nsplits_acc[i][output_chunk.index[i] + 1],
            )
            for i in range(len(output_chunk.index))
        )
        chunk_value = value[chunk_value_slice]

    records = []
    for chunk_idx, value_idx in zip(
        itertools.product(*input_indices), itertools.product(*value_indices)
    ):
        new_value = chunk_value if is_scalar else chunk_value[value_idx]
        index_in_chunk = np.ravel_multi_index(
            chunk_idx, output_chunk.op.inputs[0].shape
        )
        records.append((index_in_chunk, new_value, ts))
    return records


def setitem_to_records(tensor, tensor_index, value, timestamp):
    """
    Compute the records of `__setitem__` on tensor for every chunk.

    Returns
    -------
        dict, a dict of chunk index to records in that chunk.
    """
    output_shape, output_chunks, nsplits_acc = compute_output_of_indexing(
        tensor, tensor_index
    )

    is_scalar = (
        np.isscalar(value)
        or isinstance(value, tuple)
        and tensor.dtype.fields is not None
    )
    if not is_scalar:
        value = np.broadcast_to(value, output_shape).astype(tensor.dtype)

    records = dict()
    for output_chunk in output_chunks:
        records_in_chunk = setitem_on_chunk_to_records(
            nsplits_acc, output_chunk, value, timestamp, is_scalar=is_scalar
        )
        records[output_chunk.op.inputs[0].index] = records_in_chunk
    return records


def getitem_on_chunk_to_records(nsplits_acc, output_chunk):
    """
    Turns a `__getitem__` on chunk to a list of index-value records.

    Parameters
    ----------
    nsplits_acc:
        Accumulate nsplits arrays of the output tensor chunks.

    Returns
    -------
        records: A list of `(index, value_index)`, where `index` is the in-chunk index, and
        `value_index` is the index in the final result block.
        chunk_value_shape: shape of result of this chunk.
        chunk_value_slice: index of result of this chunk in the whole result tensor.
    """
    input_indices, value_indices = indexing_to_chunk_indices(output_chunk)

    chunk_value_slice = tuple(
        slice(
            nsplits_acc[i][output_chunk.index[i]],
            nsplits_acc[i][output_chunk.index[i] + 1],
        )
        for i in range(len(output_chunk.index))
    )

    records = []
    for chunk_idx, value_idx in zip(
        itertools.product(*input_indices), itertools.product(*value_indices)
    ):
        index_in_chunk = np.ravel_multi_index(
            chunk_idx, output_chunk.op.inputs[0].shape
        )
        records.append((index_in_chunk, value_idx))
    return records, output_chunk.shape, chunk_value_slice


def getitem_to_records(tensor, tensor_index):
    """
    Compute the records of `__getitem__` on tensor for every chunk.

    Returns
    -------
        records and output_chunk dict, records is a dict of chunk index to records
        in that chunk.
    """
    output_shape, output_chunks, nsplits_acc = compute_output_of_indexing(
        tensor, tensor_index
    )

    records = dict()
    for output_chunk in output_chunks:
        records_in_chunk = getitem_on_chunk_to_records(nsplits_acc, output_chunk)
        records[output_chunk.op.inputs[0].index] = records_in_chunk
    return records, output_shape


def normalize_timestamp(timestamp=None):
    if timestamp is None:
        timestamp = np.datetime64(datetime.now())
    if isinstance(timestamp, datetime):
        timestamp = np.datetime64(timestamp)
    return timestamp


def normalize_name(name: Optional[str] = None):
    if not name:
        return str(uuid.uuid4())
    return name
