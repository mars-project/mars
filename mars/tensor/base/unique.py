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

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...serialize import BoolField, Int32Field
from ...lib.mmh3 import hash_from_buffer
from ..operands import TensorOperand, TensorOperandMixin, TensorShuffleMap, TensorShuffleReduce
from ..array_utils import as_same_device, device
from ..core import TensorOrder
from ..utils import validate_axis


class TensorUnique(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.UNIQUE

    _return_index = BoolField('return_index')
    _return_inverse = BoolField('return_inverse')
    _return_counts = BoolField('return_counts')
    _axis = Int32Field('axis')
    _aggregate_size = Int32Field('aggregate_size')

    def __init__(self, return_index=None, return_inverse=None, return_counts=None,
                 axis=None, dtype=None, gpu=None, aggregate_size=None, **kw):
        super(TensorUnique, self).__init__(_return_index=return_index,
                                           _return_inverse=return_inverse,
                                           _return_counts=return_counts, _axis=axis,
                                           _aggregate_size=aggregate_size,
                                           _dtype=dtype, _gpu=gpu, **kw)

    @property
    def output_limit(self):
        return 1 + bool(self._return_index) + \
               bool(self._return_inverse) + bool(self._return_counts)

    @property
    def return_index(self):
        return self._return_index

    @property
    def return_inverse(self):
        return self._return_inverse

    @property
    def return_counts(self):
        return self._return_counts

    @property
    def axis(self):
        return self._axis

    @property
    def aggregate_size(self):
        return self._aggregate_size

    @classmethod
    def _gen_kws(cls, op, input_obj, chunk=False):
        kws = []

        # unique tensor
        shape = list(input_obj.shape)
        shape[op.axis] = np.nan
        kw = {'shape': tuple(shape),
              'dtype': input_obj.dtype,
              'gpu': input_obj.op.gpu}
        if chunk:
            kw['index'] = (0,) * len(shape)
        kws.append(kw)

        # unique indices tensor
        if op.return_index:
            kw = {'shape': (np.nan,),
                  'dtype': np.dtype(np.intp),
                  'gpu': input_obj.op.gpu,
                  'type': 'indices'}
            if chunk:
                kw['index'] = (0,)
            kws.append(kw)

        # unique inverse tensor
        if op.return_inverse:
            kw = {'shape': (input_obj.size,),
                  'dtype': np.dtype(np.intp),
                  'gpu': input_obj.op.gpu,
                  'type': 'inverse'}
            if chunk:
                kw['index'] = (0,)
            kws.append(kw)

        # unique counts tensor
        if op.return_counts:
            kw = {'shape': (np.nan,),
                  'dtype': np.dtype(np.intp),
                  'gpu': input_obj.op.gpu,
                  'type': 'counts'}
            if chunk:
                kw['index'] = (0,)
            kws.append(kw)

        return kws

    def __call__(self, ar):
        from .atleast_1d import atleast_1d

        ar = atleast_1d(ar)
        if self.axis is None:
            if ar.ndim > 1:
                ar = ar.flatten()
            self._axis = 0
        else:
            self._axis = validate_axis(ar.ndim, self._axis)

        kws = self._gen_kws(self, ar)
        tensors = self.new_tensors([ar], kws=kws, order=TensorOrder.C_ORDER)
        if len(tensors) == 1:
            return tensors[0]
        return tensors

    @classmethod
    def _tile_one_chunk(cls, op):
        outs = op.outputs
        ins = op.inputs

        chunk_op = op.copy().reset_key()
        in_chunk = ins[0].chunks[0]
        kws = cls._gen_kws(chunk_op, in_chunk, chunk=True)
        out_chunks = chunk_op.new_chunks([in_chunk], kws=kws, order=outs[0].order)
        new_op = op.copy()
        kws = [out.params.copy() for out in outs]
        for kw, out_chunk in zip(kws, out_chunks):
            kw['chunks'] = [out_chunk]
            kw['nsplits'] = tuple((s,) for s in out_chunk.shape)
        return new_op.new_tensors(ins, kws=kws, order=outs[0].order)

    @classmethod
    def _tile_via_shuffle(cls, op):
        # rechunk the axes except the axis to do unique into 1 chunk
        inp = op.inputs[0]
        if inp.ndim > 1:
            new_chunk_size = dict()
            for axis in range(inp.ndim):
                if axis == op.axis:
                    continue
                new_chunk_size[axis] = inp.shape[axis]
            inp = inp.rechunk(new_chunk_size).single_tiles()

    @classmethod
    def tile(cls, op):
        if len(op.inputs[0].chunks):
            return cls._tile_one_chunk(op)
        else:
            return cls._tile_via_shuffle(op)

    @classmethod
    def execute(cls, ctx, op):
        (ar,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            results = xp.unique(ar, return_index=op.return_index,
                                return_inverse=op.return_inverse,
                                return_counts=op.return_counts,
                                axis=op.axis)
            outs = op.outputs
            if len(outs) == 1:
                ctx[outs[0].key] = results
                return

            assert len(outs) == len(results)
            for out, result in zip(outs, results):
                ctx[out.key] = result


class TensorUniqueShuffleMap(TensorShuffleMap, TensorOperandMixin):
    _op_type_ = OperandDef.UNIQUE_MAP

    _return_index = BoolField('return_index')
    _return_counts = BoolField('return_counts')
    _axis = Int32Field('axis')
    _aggregate_size = Int32Field('aggregate_size')

    def __init__(self, return_index=None, return_counts=None,
                 axis=None, dtype=None, gpu=None, aggregate_size=None, **kw):
        super(TensorShuffleMap, self).__init__(_return_index=return_index,
                                               _return_counts=return_counts, _axis=axis,
                                               _aggregate_size=aggregate_size,
                                               _dtype=dtype, _gpu=gpu, **kw)

    @classmethod
    def execute(cls, ctx, op):
        (ar,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            results = xp.unique(ar, return_index=op.return_index,
                                return_counts=op.return_counts,
                                axis=op.axis)
            results = (results,) if not isinstance(results, tuple) else results
            results_iter = iter(results)
            unique_ar = next(results_iter)
            indices_ar = next(results_iter) if op.return_index else None
            counts_ar = next(results_iter) if op.return_counts else None

            inverse = pd.Data


def unique(ar, return_index=False, return_inverse=False, return_counts=False,
           axis=None, aggregate_size=None):
    """
    Find the unique elements of a tensor.

    Returns the sorted unique elements of a tensor. There are three optional
    outputs in addition to the unique elements:

    * the indices of the input tensor that give the unique values
    * the indices of the unique tensor that reconstruct the input tensor
    * the number of times each unique value comes up in the input tensor

    Parameters
    ----------
    ar : array_like
        Input tensor. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.
    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened tensor) that result in the unique tensor.
    return_inverse : bool, optional
        If True, also return the indices of the unique tensor (for the specified
        axis, if provided) that can be used to reconstruct `ar`.
    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.
    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened. If an integer,
        the subarrays indexed by the given axis will be flattened and treated
        as the elements of a 1-D tensor with the dimension of the given axis,
        see the notes for more details.  Object tensors or structured tensors
        that contain objects are not supported if the `axis` kwarg is used. The
        default is None.
    aggregate_size: int or None, optional
        How many chunks will be after unique, default as #input.chunks / options.combine_size

    Returns
    -------
    unique : Tensor
        The sorted unique values.
    unique_indices : Tensor, optional
        The indices of the first occurrences of the unique values in the
        original tensor. Only provided if `return_index` is True.
    unique_inverse : Tensor, optional
        The indices to reconstruct the original tensor from the
        unique tensor. Only provided if `return_inverse` is True.
    unique_counts : Tensor, optional
        The number of times each of the unique values comes up in the
        original tensor. Only provided if `return_counts` is True.

    Examples
    --------
    >>> np.unique([1, 1, 2, 2, 3, 3])
    array([1, 2, 3])
    >>> a = np.array([[1, 1], [2, 3]])
    >>> np.unique(a)
    array([1, 2, 3])

    Return the unique rows of a 2D array

    >>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    >>> np.unique(a, axis=0)
    array([[1, 0, 0], [2, 3, 4]])

    Return the indices of the original array that give the unique values:

    >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
    >>> u, indices = np.unique(a, return_index=True)
    >>> u
    array(['a', 'b', 'c'],
           dtype='|S1')
    >>> indices
    array([0, 1, 3])
    >>> a[indices]
    array(['a', 'b', 'c'],
           dtype='|S1')

    Reconstruct the input array from the unique values:

    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = np.unique(a, return_inverse=True)
    >>> u
    array([1, 2, 3, 4, 6])
    >>> indices
    array([0, 1, 4, 3, 1, 2, 1])
    >>> u[indices]
    array([1, 2, 6, 4, 2, 3, 2])
    """
    op = TensorUnique(return_index=return_index, return_inverse=return_inverse,
                      return_counts=return_counts, axis=axis,
                      aggregate_size=aggregate_size)
    return op(ar)
