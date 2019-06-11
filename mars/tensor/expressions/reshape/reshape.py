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

import numpy as np
from mars.tensor.expressions.utils import decide_chunk_sizes

from .... import opcodes as OperandDef
from ....compat import six, izip
from ....serialize import KeyField, TupleField, ValueType
from ..datasource import tensor as astensor
from ..core import TensorOperandMixin, TensorHasInput, TensorShuffleProxy, \
    TensorShuffleMap, TensorShuffleReduce


class TensorReshape(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.RESHAPE

    _input = KeyField('input')
    _newshape = TupleField('newshape', ValueType.int64)

    def __init__(self, newshape=None, dtype=None, **kw):
        super(TensorReshape, self).__init__(_newshape=newshape, _dtype=dtype, **kw)

    @property
    def newshape(self):
        return self._newshape

    def _set_inputs(self, inputs):
        super(TensorReshape, self)._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, a):
        return self.new_tensor([a], self._newshape)

    @staticmethod
    def _gen_reshape_rechunk_nsplits(old_shape, new_shape, nsplits):
        old_idx = len(old_shape) - 1
        new_idx = len(new_shape) - 1
        rechunk_nsplists = [None for _ in old_shape]
        reshape_nsplists = [None for _ in new_shape]

        while old_idx >= 0 or new_idx >= 0:
            old_dim_size = old_shape[old_idx]
            new_dim_size = new_shape[new_idx]

            if old_dim_size == new_dim_size:
                # nothing need to do
                rechunk_nsplists[old_idx] = nsplits[old_idx]
                reshape_nsplists[new_idx] = nsplits[old_idx]
                old_idx -= 1
                new_idx -= 1
                continue

            if old_dim_size == 1:
                rechunk_nsplists[old_idx] = (1,)
                old_idx -= 1
            elif new_dim_size == 1:
                reshape_nsplists[new_idx] = (1,)
                new_idx -= 1
            elif old_dim_size < new_dim_size:
                left_old_idx = old_idx - 1
                while left_old_idx >= 0 and \
                        np.prod(old_shape[left_old_idx: old_idx + 1]) < new_dim_size:
                    left_old_idx -= 1
                if np.prod(old_shape[left_old_idx: old_idx + 1]) != new_dim_size:
                    raise ValueError('shapes not compatible')

                for i in range(left_old_idx + 1, old_idx + 1):
                    # rechunk the higher dimension into 1 chunk
                    # e.g. ((2, 2, 2), [(3, 3), (4, 4))] -> [6, 8]
                    rechunk_nsplists[i] = (old_shape[i],)

                chunk_reduce = np.prod([len(c) for c in nsplits[left_old_idx + 1: old_idx + 1]])
                # cause the higher dimension has been concatenated,
                # the lowest dimension should be expanded to reduce size
                rechunk_nsplists[left_old_idx] = \
                    TensorReshape._expand_nsplit_by_reduce(nsplits[left_old_idx], chunk_reduce)

                size_reduce = np.prod(old_shape[left_old_idx + 1: old_idx + 1])
                reshape_nsplists[new_idx] = tuple(size_reduce * c for c in rechunk_nsplists[left_old_idx])

                old_idx = left_old_idx - 1
                new_idx -= 1
            else:
                assert old_dim_size > new_dim_size
                lef_new_idx = new_idx - 1
                while lef_new_idx >= 0 and \
                        np.prod(new_shape[lef_new_idx: new_idx +1]) < old_dim_size:
                    lef_new_idx -= 1
                if np.prod(new_shape[lef_new_idx: new_idx + 1]) != old_dim_size:
                    raise ValueError('shapes not compatible')

                chunk_expand = np.prod(new_shape[lef_new_idx + 1: new_idx + 1])
                rechunk_nsplists[old_idx] = TensorReshape._reduce_nsplit_by_expand(nsplits[old_idx], chunk_expand)

                for i in range(lef_new_idx + 1, new_idx + 1):
                    reshape_nsplists[i] = (new_shape[i],)
                reshape_nsplists[lef_new_idx] = tuple(c // chunk_expand for c in rechunk_nsplists[old_idx])

                old_idx -= 1
                new_idx = lef_new_idx - 1

        assert np.prod([len(s) for s in rechunk_nsplists]) == \
               np.prod([len(s) for s in reshape_nsplists])
        return rechunk_nsplists, reshape_nsplists

    @staticmethod
    def _expand_nsplit_by_reduce(splits, reduced):
        if reduced == 1:
            return splits

        out = []
        for s in splits:
            x = s
            part = max(x / reduced, 1)
            while x >= 2 * part:
                out.append(int(part))
                x -= int(part)
            if x:
                out.append(x)
        assert sum(splits) == sum(out)
        return tuple(out)

    @staticmethod
    def _reduce_nsplit_by_expand(splits, expand):
        assert sum(splits) % expand == 0

        out = []
        residual = 0
        for chunk in splits:
            chunk += residual
            div = chunk // expand
            residual = chunk % expand
            good = expand * div
            if good:
                out.append(good)
        return tuple(out)

    @staticmethod
    def _tile_as_shuffle(op):
        in_tensor = op.input
        tensor = op.outputs[0]
        new_shape = op.newshape
        shuffle_inputs, shuffle_outputs = [], []
        axis_offsets = [[0] + np.cumsum(ns)[:-1].tolist() for ns in in_tensor.nsplits]

        max_chunk_size = max(max(tp) for tp in in_tensor.nsplits)
        out_nsplits = decide_chunk_sizes(new_shape, max_chunk_size, tensor.dtype.itemsize)
        chunk_size_idxes = (range(len(size)) for size in out_nsplits)

        for inp in in_tensor.chunks:
            offset = tuple(axis_offsets[axis][idx] for axis, idx in enumerate(inp.index))
            chunk_op = TensorReshapeMap(_input=inp, _axis_offsets=offset, _oldshape=in_tensor.shape,
                                        _newshape=new_shape, _new_chunk_size=(max_chunk_size,) * len(new_shape),
                                        _dtype=inp.dtype)
            shuffle_inputs.append(chunk_op.new_chunk([inp], shape=(np.nan,), index=inp.index))

        proxy_chunk = TensorShuffleProxy(dtype=in_tensor.dtype, _tensor_keys=[in_tensor.op.key]) \
            .new_chunk(shuffle_inputs, shape=())

        for chunk_shape, chunk_idx in izip(itertools.product(*out_nsplits),
                                           itertools.product(*chunk_size_idxes)):
            shuffle_key = ','.join(str(o) for o in chunk_idx)
            chunk_op = TensorReshapeReduce(_dtype=tensor.dtype, _shuffle_key=shuffle_key)
            shuffle_outputs.append(chunk_op.new_chunk([proxy_chunk], shape=chunk_shape, index=chunk_idx))

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, new_shape, chunks=shuffle_outputs,
                                  nsplits=out_nsplits)

    @classmethod
    def tile(cls, op):
        in_tensor = op.input
        tensor = op.outputs[0]

        try:
            rechunk_nsplits, reshape_nsplits = cls._gen_reshape_rechunk_nsplits(
                in_tensor.shape, tensor.shape, in_tensor.nsplits)
            rechunked_tensor = in_tensor.rechunk(rechunk_nsplits).single_tiles()
            in_idxes = itertools.product(*[range(len(s)) for s in rechunk_nsplits])
            out_idxes = itertools.product(*[range(len(s)) for s in reshape_nsplits])
            out_shape = itertools.product(*[s for s in reshape_nsplits])
            out_chunks = []
            for input_idx, out_idx, out_shape in izip(in_idxes, out_idxes, out_shape):
                in_chunk = rechunked_tensor.cix[input_idx]
                chunk_op = op.copy().reset_key()
                chunk_op._newshape = out_shape
                out_chunk = chunk_op.new_chunk([in_chunk], shape=out_shape, index=out_idx)
                out_chunks.append(out_chunk)

            new_op = op.copy()
            return new_op.new_tensors(op.inputs, tensor.shape,
                                      chunks=out_chunks, nsplits=reshape_nsplits)
        except ValueError:
            # TODO: make this as default when shuffle is mature
            if getattr(op.extra_params, '_reshape_with_shuffle', False):
                return cls._tile_as_shuffle(op)

            # shape incompatible, we will first do flatten, then reshape to the new shape
            return [in_tensor.reshape(-1).single_tiles().reshape(tensor.shape).single_tiles()]


class TensorReshapeMap(TensorShuffleMap, TensorOperandMixin):
    _op_type_ = OperandDef.RESHAPE_MAP

    _input = KeyField('input')
    _axis_offsets = TupleField('axis_offsets', ValueType.uint64)
    _oldshape = TupleField('oldshape', ValueType.uint64)
    _newshape = TupleField('newshape', ValueType.uint64)
    _new_chunk_size = TupleField('new_chunk_size', ValueType.uint64)

    def _set_inputs(self, inputs):
        super(TensorReshapeMap, self)._set_inputs(inputs)
        self._input = self._inputs[0]

    @property
    def axis_offsets(self):
        return self._axis_offsets

    @property
    def oldshape(self):
        return self._oldshape

    @property
    def newshape(self):
        return self._newshape

    @property
    def new_chunk_size(self):
        return self._new_chunk_size


class TensorReshapeReduce(TensorShuffleReduce, TensorOperandMixin):
    _op_type_ = OperandDef.RESHAPE_REDUCE

    _input = KeyField('input')

    def _set_inputs(self, inputs):
        super(TensorReshapeReduce, self)._set_inputs(inputs)
        self._input = self._inputs[0]


def reshape(a, newshape):
    """
    Gives a new shape to a tensor without changing its data.

    Parameters
    ----------
    a : array_like
        Tensor to be reshaped.
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D tensor of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the tensor and remaining dimensions.

    Returns
    -------
    reshaped_array : Tensor
        This will be a new view object if possible; otherwise, it will
        be a copy.

    See Also
    --------
    Tensor.reshape : Equivalent method.

    Notes
    -----
    It is not always possible to change the shape of a tensor without
    copying the data. If you want an error to be raised when the data is copied,
    you should assign the new shape to the shape attribute of the array::

    >>> import mars.tensor as mt

    >>> a = mt.arange(6).reshape((3, 2))
    >>> a.execute()
    array([[0, 1],
           [2, 3],
           [4, 5]])

    You can think of reshaping as first raveling the tensor (using the given
    index order), then inserting the elements from the raveled tensor into the
    new tensor using the same kind of index ordering as was used for the
    raveling.

    >>> mt.reshape(a, (2, 3)).execute()
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> mt.reshape(mt.ravel(a), (2, 3)).execute()
    array([[0, 1, 2],
           [3, 4, 5]])

    Examples
    --------
    >>> a = mt.array([[1,2,3], [4,5,6]])
    >>> mt.reshape(a, 6).execute()
    array([1, 2, 3, 4, 5, 6])

    >>> mt.reshape(a, (3,-1)).execute()       # the unspecified value is inferred to be 2
    array([[1, 2],
           [3, 4],
           [5, 6]])
    """
    a = astensor(a)
    if isinstance(newshape, six.integer_types):
        newshape = (newshape,)
    else:
        newshape = tuple(int(s) for s in newshape)

    if np.isnan(sum(a.shape)):
        raise ValueError('tensor shape is unknown, {0}'.format(a.shape))

    known_shape = [s for s in newshape if s >= 0]
    missing_dim = len(newshape) - len(known_shape)
    if missing_dim > 1:
        raise ValueError('can only specify one unknown dimension')
    if missing_dim == 1:
        known_size = np.prod(known_shape)
        newshape = tuple(int(a.size / known_size) if s < 0 and known_size > 0 else s
                         for s in newshape)

    if a.size != np.prod(newshape):
        raise ValueError('cannot reshape array of size {0} into shape {1}'.format(a.size, newshape))

    if a.shape == newshape:
        # does not need to reshape
        return a

    op = TensorReshape(newshape, dtype=a.dtype)
    return op(a)
