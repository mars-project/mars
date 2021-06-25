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

from ... import opcodes as OperandDef
from ...lib.filesystem import get_fs, FSMap
from ..core import TensorOrder
from .core import TensorFromHDF5Like


class TensorFromZarr(TensorFromHDF5Like):
    _op_type_ = OperandDef.TENSOR_FROM_ZARR

    @classmethod
    def execute(cls, ctx, op):
        import zarr

        axis_offsets = op.axis_offsets
        shape = op.outputs[0].shape

        fs = get_fs(op.filename, None)
        fs_map = FSMap(op.filename, fs)

        root = zarr.group(store=fs_map)
        path = cls.get_path(op.group, op.dataset)
        arr = root[path]

        data = arr[tuple(slice(offset, offset + size)
                         for offset, size in zip(axis_offsets, shape))]
        ctx[op.outputs[0].key] = data


def fromzarr(path, group=None, dataset=None, chunk_size=None):
    import zarr

    if isinstance(path, zarr.Array):
        arr = path
        if isinstance(arr.store, FSMap):
            root = arr.store.root
            path, dataset = root.rsplit('/', 1)
        else:
            path = arr.store.path
            if '/' in arr.path and group is None:
                group = arr.path.rsplit('/', 1)[0]
            dataset = arr.basename
            if not dataset:
                path, dataset = path.rsplit('/', 1)
        shape = arr.shape
    elif isinstance(path, str):
        fs = get_fs(path, None)
        fs_map = FSMap(path, fs)

        if group is None and dataset is None:
            arr = zarr.open(fs_map)
            if isinstance(arr, zarr.Array):
                return fromzarr(arr, chunk_size=chunk_size)

        g = zarr.group(store=fs_map)
        arr = g[TensorFromZarr.get_path(group, dataset)]
        shape = arr.shape
    else:
        raise TypeError('`path` passed has wrong type, '
                        'expect str, or zarr.Array'
                        f'got {type(path)}')

    chunk_size = chunk_size if chunk_size is not None else arr.chunks
    op = TensorFromZarr(filename=path, group=group, dataset=dataset,
                        dtype=arr.dtype)
    return op(shape, chunk_size=chunk_size, order=TensorOrder(arr.order))
