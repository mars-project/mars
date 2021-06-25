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

import pickle
from typing import Dict

import numpy as np

from ... import opcodes as OperandDef
from ...serialization.serializables import FieldTypes, KeyField, \
    StringField, BytesField, TupleField
from ...lib.filesystem import get_fs, FSMap
from ...utils import has_unknown_shape
from .core import TensorDataStore


class ZarrOptions(object):
    def __init__(self, options: Dict):
        self._options = options

    def todict(self):
        return self._options

    @staticmethod
    def _stringfy(v):
        return pickle.dumps(v) if not isinstance(v, str) else v

    def __mars_tokenize__(self):
        return list(self._options.keys()), \
               list(self._stringfy(v) for v in self._options.values())

    def __getstate__(self):
        return self._options

    def __setstate__(self, state):
        self._options = state


class TensorToZarrDataStore(TensorDataStore):
    _op_type_ = OperandDef.TENSOR_STORE_ZARR

    _input = KeyField('input')
    _path = StringField('path')
    _group = StringField('group')
    _dataset = StringField('dataset')
    _zarr_options = BytesField('zarr_options', on_serialize=pickle.dumps,
                               on_deserialize=pickle.loads)
    _axis_offsets = TupleField('axis_offsets', FieldTypes.int32)

    def __init__(self, path=None, group=None, dataset=None, zarr_options=None,
                 axis_offsets=None, **kw):
        super().__init__(_path=path, _group=group, _dataset=dataset,
                         _zarr_options=zarr_options, _axis_offsets=axis_offsets, **kw)

    @property
    def path(self):
        return self._path

    @property
    def group(self):
        return self._group

    @property
    def dataset(self):
        return self._dataset

    @property
    def zarr_options(self):
        return self._zarr_options

    @property
    def axis_offsets(self):
        return self._axis_offsets

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    @classmethod
    def tile(cls, op):
        import zarr

        if has_unknown_shape(*op.inputs):
            yield
        in_tensor = op.input

        # create dataset
        fs = get_fs(op.path, None)
        path = op.path
        if op.group is not None:
            path += '/' + op.group
        fs_map = FSMap(path, fs)
        zarr.open(fs_map, 'w', path=op.dataset,
                  dtype=in_tensor.dtype, shape=in_tensor.shape,
                  chunks=tuple(max(ns) for ns in in_tensor.nsplits),
                  **op.zarr_options.todict())

        cum_nsplits = [[0] + np.cumsum(ns).tolist() for ns in in_tensor.nsplits]
        out_chunks = []
        for chunk in in_tensor.chunks:
            chunk_op = op.copy().reset_key()
            chunk_op._axis_offsets = \
                tuple(cs[i] for i, cs in zip(chunk.index, cum_nsplits))
            out_chunks.append(chunk_op.new_chunk([chunk], shape=(0,) * chunk.ndim,
                                                 index=chunk.index))

        new_op = op.copy()
        out = op.outputs[0]
        nsplits = tuple((0,) * len(ns) for ns in in_tensor.nsplits)
        return new_op.new_tensors(op.inputs, shape=out.shape, order=out.order,
                                  nsplits=nsplits, chunks=out_chunks)

    @classmethod
    def execute(cls, ctx, op):
        import zarr

        fs = get_fs(op.path, None)
        fs_map = FSMap(op.path, fs)

        group = zarr.Group(store=fs_map, path=op.group)
        array = group[op.dataset]

        to_store = ctx[op.inputs[0].key]
        axis_offsets = op.axis_offsets
        shape = to_store.shape

        array[tuple(slice(offset, offset + size)
              for offset, size
              in zip(axis_offsets, shape))] = to_store

        ctx[op.outputs[0].key] = np.empty((0,) * to_store.ndim,
                                          dtype=to_store.dtype)


def tozarr(path, x, group=None, dataset=None, **zarr_options):
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
        for attr in ['compressor', 'filters']:
            if getattr(arr, attr):
                zarr_options[attr] = getattr(arr, attr)
    elif isinstance(path, str):
        if dataset is None:
            path, dataset = path.rsplit('/', 1)
    else:
        raise TypeError('`path` passed has wrong type, '
                        'expect str, or zarr.Array'
                        f'got {type(path)}')

    op = TensorToZarrDataStore(path=path, group=group, dataset=dataset,
                               zarr_options=ZarrOptions(zarr_options))
    return op(x)
