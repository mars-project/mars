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

from ... import opcodes as OperandDef
from ...filesystem import open_file
from .core import TensorFromHDF5Like, TensorOrder


class TensorHDF5DataSource(TensorFromHDF5Like):
    _op_type_ = OperandDef.TENSOR_FROM_HDF5

    @classmethod
    def execute(cls, ctx, op):
        import h5py

        axis_offsets = op.axis_offsets
        shape = op.outputs[0].shape

        with h5py.File(open_file(op.filename), mode='r') as f:
            ds = f[op.path]
            data = ds[tuple(slice(offset, offset + size)
                            for offset, size in zip(axis_offsets, shape))]
            ctx[op.outputs[0].key] = data


def fromhdf5(hdf5_file, group=None, dataset=None, chunk_size=None):
    import h5py

    if isinstance(hdf5_file, h5py.Dataset):
        filename = hdf5_file.file.filename
        group = hdf5_file.parent.name
        dataset = hdf5_file.name.rsplit('/', 1)[1]
        chunk_size = chunk_size if chunk_size is not None else hdf5_file.chunks
        shape = hdf5_file.shape
        dtype = hdf5_file.dtype
    elif isinstance(hdf5_file, h5py.File):
        filename = hdf5_file.filename
        if dataset is None:
            raise ValueError('`dataset` should be provided')
        try:
            h5_dataset = hdf5_file[TensorHDF5DataSource.get_path(group, dataset)]
        except KeyError:
            raise ValueError('dataset({}) does not exist'.format(dataset))
        chunk_size = chunk_size if chunk_size is not None else h5_dataset.chunks
        shape = h5_dataset.shape
        dtype = h5_dataset.dtype
    elif isinstance(hdf5_file, str):
        filename = hdf5_file
        try:
            with h5py.File(open_file(filename), mode='r') as f:
                if dataset is None:
                    raise ValueError('`dataset` should be provided')
                h5_dataset = f[TensorHDF5DataSource.get_path(group, dataset)]

                chunk_size = chunk_size if chunk_size is not None else h5_dataset.chunks
                shape = h5_dataset.shape
                dtype = h5_dataset.dtype
        except KeyError:
            raise ValueError('dataset({}) does not exist'.format(dataset))
    else:
        raise TypeError('`hdf5_file` passed has wrong type, '
                        'expect str, h5py.File or h5py.Dataset, '
                        'got {}'.format(type(hdf5_file)))

    op = TensorHDF5DataSource(filename=filename, group=group,
                              dataset=dataset, dtype=dtype)
    return op(shape, chunk_size=chunk_size, order=TensorOrder.C_ORDER)
