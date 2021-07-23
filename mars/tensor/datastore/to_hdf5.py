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

import threading
import time
from typing import List

import numpy as np

from ... import opcodes as OperandDef
from ...core.context import get_context
from ...oscar import ActorNotExist
from ...serialization.serializables import FieldTypes, KeyField, \
    StringField, DictField, TupleField
from ...lib.filesystem import open_file
from ...utils import has_unknown_shape
from ..datasource import tensor as astensor
from .core import TensorDataStore


class _HDF5Container:
    def __init__(self,
                 all_chunk_op_keys: List[str]):
        self._all_chunk_op_keys = set(all_chunk_op_keys)
        self._done_chunk_op_keys = set()
        self._lock = threading.Lock()

    def acquire(self):
        return self._lock.acquire()

    def release(self):
        return self._lock.release()

    def mark_done(self, op_key: str):
        self._done_chunk_op_keys.add(op_key)

    def is_done(self):
        return self._done_chunk_op_keys == self._all_chunk_op_keys


class TensorHDF5DataStore(TensorDataStore):
    _op_type_ = OperandDef.TENSOR_STORE_HDF5

    _input = KeyField('input')
    _filename = StringField('filename')
    _group = StringField('group')
    _dataset = StringField('dataset')
    _dataset_kwds = DictField('dataset_kwds', key_type=FieldTypes.string)
    _axis_offsets = TupleField('axis_offsets', FieldTypes.int32)
    _out_shape = TupleField('out_shape', FieldTypes.int32)
    _container_name = StringField('container_name')

    def __init__(self, filename=None, group=None, dataset=None,
                 dataset_kwds=None, container_name=None, **kw):
        super().__init__(_filename=filename, _group=group, _dataset=dataset,
                         _dataset_kwds=dataset_kwds,
                         _container_name=container_name, **kw)

    @property
    def input(self):
        return self._input

    @property
    def filename(self):
        return self._filename

    @property
    def group(self):
        return self._group

    @property
    def dataset(self):
        return self._dataset

    @property
    def dataset_kwds(self):
        return self._dataset_kwds

    @property
    def axis_offsets(self):
        return self._axis_offsets

    @property
    def out_shape(self):
        return self._out_shape

    @property
    def container_name(self):
        return self._container_name

    @property
    def path(self):
        paths = []
        if self._group is not None:
            paths.append(self.group)
        paths.append(self.dataset)
        return '/'.join(paths)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    @classmethod
    def tile(cls, op):
        if has_unknown_shape(*op.inputs):
            yield
        in_tensor = op.input

        with open_file(op.filename, 'w'):
            # create file if not exist
            pass

        nsplits = tuple([(0,) * len(ns) for ns in in_tensor.nsplits])
        if len(in_tensor.chunks) == 1:
            in_chunk = in_tensor.chunks[0]
            chunk_op = op.copy().reset_key()
            chunk_op._axis_offsets = (0,) * in_chunk.ndim
            chunk_op._out_shape = in_tensor.shape
            out_chunk = chunk_op.new_chunk([in_chunk], shape=(0,) * in_chunk.ndim,
                                           index=in_chunk.index)
            new_op = op.copy()
            return new_op.new_tensors(op.inputs, shape=(0,) * in_tensor.ndim,
                                      nsplits=nsplits, chunks=[out_chunk])

        container_name = f'{op.key}_{int(time.time())}'

        out_chunks = []
        acc = [[0] + np.cumsum(ns).tolist() for ns in in_tensor.nsplits]
        chunk_op_keys = []
        for chunk in in_tensor.chunks:
            chunk_op = op.copy().reset_key()
            chunk_op._out_shape = in_tensor.shape
            chunk_op._container_name = container_name
            chunk_op._axis_offsets = tuple(acc[ax][i] for ax, i in enumerate(chunk.index))
            out_chunk = chunk_op.new_chunk([chunk], shape=(0,) * chunk.ndim,
                                           index=chunk.index)
            out_chunks.append(out_chunk)
            chunk_op_keys.append(out_chunk.op.key)

        ctx = get_context()
        ctx.create_remote_object(container_name, _HDF5Container, chunk_op_keys)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=(0,) * in_tensor.ndim,
                                  nsplits=nsplits,
                                  chunks=out_chunks)

    @classmethod
    def execute(cls, ctx, op: "TensorHDF5DataStore"):
        import h5py

        to_store = ctx[op.inputs[0].key]
        axis_offsets = op.axis_offsets

        container_name = op.container_name
        container: _HDF5Container = None
        if container_name:
            container = ctx.get_remote_object(container_name)
            container.acquire()
        try:
            with h5py.File(open_file(op.filename, mode='r+b'), mode='r+') as f:
                try:
                    ds = f[op.path]
                except KeyError:
                    ds = f.create_dataset(op.path, shape=op.out_shape,
                                          dtype=to_store.dtype, **op.dataset_kwds)
                ds[tuple(slice(offset, offset + size)
                         for offset, size
                         in zip(axis_offsets, to_store.shape))] = to_store
                ctx[op.outputs[0].key] = np.empty((0,) * to_store.ndim,
                                                  dtype=to_store.dtype)
                if container:
                    container.mark_done(op.key)
        finally:
            if container:
                try:
                    container.release()
                    if container.is_done():
                        ctx.destroy_remote_object(container_name)
                except ActorNotExist:
                    # destroyed by other execution, just ignore
                    return


def tohdf5(hdf5_file, x, group=None, dataset=None, **kwds):
    import h5py

    x = astensor(x)
    if isinstance(hdf5_file, h5py.Dataset):
        filename = hdf5_file.file.filename
        group = hdf5_file.parent.name
        dataset = hdf5_file.name.rsplit('/', 1)[1]
    elif isinstance(hdf5_file, h5py.File):
        filename = hdf5_file.filename
        if dataset is None:
            raise ValueError('`dataset` should be provided')
    elif isinstance(hdf5_file, str):
        filename = hdf5_file
        if dataset is None:
            raise ValueError('`dataset` should be provided')
    else:
        raise TypeError('`hdf5_file` passed has wrong type, '
                        'expect str, h5py.File or h5py.Dataset, '
                        f'got {type(hdf5_file)}')

    op = TensorHDF5DataStore(filename=filename, group=group, dataset=dataset,
                             dataset_kwds=kwds)
    return op(x)
