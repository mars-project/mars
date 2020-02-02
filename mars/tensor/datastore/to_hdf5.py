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

import numpy as np

from ... import opcodes as OperandDef
from ...serialize import ValueType, KeyField, StringField, DictField, TupleField
from ...context import RunningMode
from ...filesystem import open_file
from ...tiles import TilesError
from ...utils import check_chunks_unknown_shape
from ...operands import SuccessorsExclusive
from ..datasource import tensor as astensor
from .core import TensorDataStore


class TensorHDF5DataStore(TensorDataStore):
    _op_type_ = OperandDef.TENSOR_STORE_HDF5

    _input = KeyField('input')
    _lock = KeyField('lock')
    _filename = StringField('filename')
    _group = StringField('group')
    _dataset = StringField('dataset')
    _dataset_kwds = DictField('dataset_kwds', key_type=ValueType.string)
    _axis_offsets = TupleField('axis_offsets', ValueType.int32)
    _out_shape = TupleField('out_shape', ValueType.int32)

    def __init__(self, lock=None, filename=None, group=None, dataset=None,
                 dataset_kwds=None, **kw):
        super().__init__(_lock=lock, _filename=filename, _group=group, _dataset=dataset,
                         _dataset_kwds=dataset_kwds, **kw)

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
    def path(self):
        paths = []
        if self._group is not None:
            paths.append(self.group)
        paths.append(self.dataset)
        return '/'.join(paths)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if self._lock is not None:
            self._lock = self._inputs[-1]

    @classmethod
    def tile(cls, op):
        check_chunks_unknown_shape(op.inputs, TilesError)
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

        # hdf5 cannot write concurrently,
        # thus create a SuccessorsExclusive to control the concurrency
        exclusive_chunk = SuccessorsExclusive(on=in_tensor.key).new_chunk(in_tensor.chunks)
        out_chunks = []
        acc = [[0] + np.cumsum(ns).tolist() for ns in in_tensor.nsplits]
        for chunk in in_tensor.chunks:
            chunk_op = op.copy().reset_key()
            chunk_op._lock = exclusive_chunk
            chunk_op._out_shape = in_tensor.shape
            chunk_op._axis_offsets = tuple(acc[ax][i] for ax, i in enumerate(chunk.index))
            chunk_op._prepare_inputs = [True, False]
            out_chunk = chunk_op.new_chunk([chunk, exclusive_chunk], shape=(0,) * chunk.ndim,
                                           index=chunk.index)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=(0,) * in_tensor.ndim,
                                  nsplits=nsplits,
                                  chunks=out_chunks)

    @classmethod
    def execute(cls, ctx, op):
        import h5py

        to_store = ctx[op.inputs[0].key]
        lock = None
        axis_offsets = op.axis_offsets

        # for the local runtime,
        # lock is created in the execution of SuccessorsExclusive.
        # meanwhile for distributed runtime
        # operand actor of SuccessorsExclusive will take over
        # the correspond exclusive behavior
        if ctx.running_mode == RunningMode.local and len(op.inputs) == 2:
            lock = ctx[op.inputs[1].key]
            lock.acquire()
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
        finally:
            if lock is not None:
                lock.release()


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
                        'got {}'.format(type(hdf5_file)))

    op = TensorHDF5DataStore(filename=filename, group=group, dataset=dataset,
                             dataset_kwds=kwds)
    return op(x)
