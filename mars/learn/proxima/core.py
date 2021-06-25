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
    import pyproxima2 as proxima
except ImportError:  # pragma: no cover
    proxima = None

from ... import tensor as mt
from ...tensor.merge import TensorConcatenate
from ...tensor.indexing import TensorSlice


available_numpy_dtypes = [
    np.dtype(np.float16),
    np.dtype(np.float32),
    np.dtype(np.int8),
    np.dtype(np.int16),
]


if proxima:
    _proxima_types = [
        proxima.IndexMeta.FT_FP16,
        proxima.IndexMeta.FT_FP32,
        proxima.IndexMeta.FT_INT8,
        proxima.IndexMeta.FT_INT16,
    ]
    assert len(_proxima_types) == len(available_numpy_dtypes)
    _type_mapping = {numpy_dtype: proxima_type
                     for numpy_dtype, proxima_type
                     in zip(available_numpy_dtypes, _proxima_types)}


def rechunk_tensor(tensor, chunk_size):
    # TODO(hks): Provide a unify rechunk logic with mmap.
    cur_chunks = []

    out_nchunks = tensor.shape[0] // chunk_size
    row_nsplits = [chunk_size] * out_nchunks
    rest = tensor.shape[0] % chunk_size
    if rest >= out_nchunks:
        row_nsplits.append(rest)
    else:
        for i in range(tensor.shape[0] % chunk_size):
            row_nsplits[-i-1] += 1

    tensor_cumnrows = np.cumsum([0] + list(tensor.nsplits[0]))
    offset = 0
    out_groups = []
    for split in row_nsplits:
        start_chunk_index = int(tensor_cumnrows.searchsorted(offset))
        start_chunk_index = start_chunk_index - 1 if start_chunk_index != 0 else 0
        end_chunk_index = int(tensor_cumnrows.searchsorted(offset + split) - 1)
        if start_chunk_index == end_chunk_index:
            t = tensor.chunks[start_chunk_index]
            slice_op = TensorSlice((slice(offset - tensor_cumnrows[start_chunk_index],
                                          split + offset - tensor_cumnrows[end_chunk_index]),
                                    slice(None)), dtype=t.dtype)
            out_groups.append([slice_op.new_chunk([t], shape=(split, t.shape[1]),
                                                  index=(len(cur_chunks), 0),
                                                  order=t.order)])
        else:
            chunks = []
            start_chunk = tensor.chunks[start_chunk_index]
            start_slice = int(offset - tensor_cumnrows[start_chunk_index])
            slice_op = TensorSlice((slice(start_slice, None),
                                    slice(None)), dtype=start_chunk.dtype)
            chunks.append(slice_op.new_chunk([start_chunk], shape=(start_chunk.shape[0] - start_slice,
                                                                   start_chunk.shape[1]),
                                             index=(0, 0),
                                             order=start_chunk.order))
            chunks.extend(tensor.chunks[start_chunk_index + 1: end_chunk_index])
            end_chunk = tensor.chunks[end_chunk_index]
            end_slice = int(split + offset - tensor_cumnrows[end_chunk_index])
            slice_op_end = TensorSlice((slice(None, end_slice),
                                        slice(None)), dtype=start_chunk.dtype)
            chunks.append(slice_op_end.new_chunk([end_chunk], shape=(end_slice, end_chunk.shape[1]),
                                                 index=(end_chunk_index - start_chunk_index, 0),
                                                 order=end_chunk.order))
            out_groups.append(chunks)

        offset += split

    return out_groups


def build_mmap_chunks(chunks, worker, file_prefix):
    write_mmap_chunks = []
    nrows = sum(c.shape[0] for c in chunks)
    array_shape = (nrows, chunks[0].shape[1])
    array_dtype = chunks[0].dtype
    create_mmap_op = TensorConcatenate(mmap=True, create_mmap_file=True,
                                       total_shape=array_shape,
                                       file_prefix=file_prefix,
                                       dtype=array_dtype)
    create_mmap_op.expect_worker = worker
    create_mmap_chunk = create_mmap_op.new_chunk(
        None, index=(0,), shape=(), dtype=array_dtype)
    start_index = 0
    for j, chk in enumerate(chunks):
        s = slice(start_index, start_index + chk.shape[0])
        start_index += chk.shape[0]
        write_mmap_op = TensorConcatenate(mmap=True, create_mmap_file=False,
                                          total_shape=array_shape,
                                          partition_slice=s,
                                          dtype=array_dtype)
        write_mmap_op.expect_worker = worker
        write_mmap_chunk = write_mmap_op.new_chunk([create_mmap_chunk, chk],
                                                   index=(j + 1, 0), shape=(),
                                                   dtype=array_dtype)
        write_mmap_chunks.append(write_mmap_chunk)
    return write_mmap_chunks


def validate_tensor(tensor):
    if hasattr(tensor, 'to_tensor'):
        tensor = tensor.to_tensor()
    else:
        tensor = mt.tensor(tensor)
    if tensor.ndim != 2:
        raise ValueError('Input tensor should be 2-d')
    return tensor


def get_proxima_type(np_dtype):
    try:
        return _type_mapping[np_dtype]
    except KeyError:
        raise TypeError(f"Does not support {np_dtype}, available types include "
                        f"{', '.join(t.name for t in _type_mapping)}")
