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

import itertools

import numpy as np

from ... import opcodes as OperandDef
from ...config import options
from ...operands import OperandStage
from ...lib import sparse
from ...lib.sparse.core import get_array_module as get_sparse_array_module
from ...serialize import BoolField, Int32Field, Int64Field
from ...tiles import TilesError
from ...utils import get_shuffle_input_keys_idxes, check_chunks_unknown_shape
from ..operands import TensorMapReduceOperand, TensorOperandMixin, TensorShuffleProxy
from ..array_utils import as_same_device, device
from ..core import TensorOrder
from ..utils import validate_axis, hash_on_axis


class TensorUnique(TensorMapReduceOperand, TensorOperandMixin):
    _op_type_ = OperandDef.UNIQUE

    _return_index = BoolField('return_index')
    _return_inverse = BoolField('return_inverse')
    _return_counts = BoolField('return_counts')
    _axis = Int32Field('axis')
    _aggregate_size = Int32Field('aggregate_size')

    _aggregate_id = Int32Field('aggregate_id')
    _start_pos = Int64Field('start_pos')

    def __init__(self, return_index=None, return_inverse=None, return_counts=None,
                 axis=None, start_pos=None, stage=None, shuffle_key=None,
                 dtype=None, gpu=None, aggregate_id=None, aggregate_size=None, **kw):
        super().__init__(_return_index=return_index, _return_inverse=return_inverse,
                         _return_counts=return_counts, _axis=axis, _start_pos=start_pos,
                         _aggregate_id=aggregate_id, _aggregate_size=aggregate_size,
                         _stage=stage, _shuffle_key=shuffle_key, _dtype=dtype, _gpu=gpu, **kw)

    @property
    def output_limit(self):
        if self.stage == OperandStage.map:
            return 1
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

    @property
    def aggregate_id(self):
        return self._aggregate_id

    @property
    def start_pos(self):
        return self._start_pos

    @classmethod
    def _gen_kws(cls, op, input_obj, chunk=False, chunk_index=None):
        kws = []

        # unique tensor
        shape = list(input_obj.shape)
        shape[op.axis] = np.nan
        kw = {'shape': tuple(shape),
              'dtype': input_obj.dtype,
              'gpu': input_obj.op.gpu}
        if chunk:
            idx = [0, ] * len(shape)
            idx[op.axis] = chunk_index or 0
            kw['index'] = tuple(idx)
        kws.append(kw)

        # unique indices tensor
        if op.return_index:
            kw = {'shape': (np.nan,),
                  'dtype': np.dtype(np.intp),
                  'gpu': input_obj.op.gpu,
                  'type': 'indices'}
            if chunk:
                kw['index'] = (chunk_index or 0,)
            kws.append(kw)

        # unique inverse tensor
        if op.return_inverse:
            kw = {'shape': (input_obj.shape[op.axis],),
                  'dtype': np.dtype(np.intp),
                  'gpu': input_obj.op.gpu,
                  'type': 'inverse'}
            if chunk:
                kw['index'] = (chunk_index or 0,)
            kws.append(kw)

        # unique counts tensor
        if op.return_counts:
            kw = {'shape': (np.nan,),
                  'dtype': np.dtype(np.int_),
                  'gpu': input_obj.op.gpu,
                  'type': 'counts'}
            if chunk:
                kw['index'] = (chunk_index or 0,)
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
                if np.isnan(inp.shape[axis]):
                    raise TilesError('input tensor has unknown shape '
                                     'on axis {}'.format(axis))
                new_chunk_size[axis] = inp.shape[axis]
            check_chunks_unknown_shape([inp], TilesError)
            inp = inp.rechunk(new_chunk_size)._inplace_tile()

        aggregate_size = op.aggregate_size
        if aggregate_size is None:
            aggregate_size = max(inp.chunk_shape[op.axis] // options.combine_size, 1)

        unique_on_chunk_sizes = inp.nsplits[op.axis]
        start_poses = np.cumsum((0,) + unique_on_chunk_sizes).tolist()[:-1]
        map_chunks = []
        for c in inp.chunks:
            map_op = TensorUnique(stage=OperandStage.map,
                                  return_index=op.return_index,
                                  return_inverse=op.return_inverse,
                                  return_counts=op.return_counts,
                                  axis=op.axis, aggregate_size=aggregate_size,
                                  start_pos=start_poses[c.index[op.axis]],
                                  dtype=inp.dtype)
            shape = list(c.shape)
            shape[op.axis] = np.nan
            map_chunks.append(map_op.new_chunk([c], shape=tuple(shape), index=c.index))

        shuffle_chunk = TensorShuffleProxy(dtype=inp.dtype, _tensor_keys=[inp.op.key]) \
            .new_chunk(map_chunks, shape=())

        reduce_chunks = [list() for _ in range(len(op.outputs))]
        for i in range(aggregate_size):
            reduce_op = TensorUnique(stage=OperandStage.reduce,
                                     return_index=op.return_index,
                                     return_inverse=op.return_inverse,
                                     return_counts=op.return_counts,
                                     axis=op.axis, aggregate_id=i,
                                     shuffle_key=str(i))
            kws = cls._gen_kws(op, inp, chunk=True, chunk_index=i)
            chunks = reduce_op.new_chunks([shuffle_chunk], kws=kws,
                                          order=op.outputs[0].order)
            for j, c in enumerate(chunks):
                reduce_chunks[j].append(c)

        if op.return_inverse:
            inverse_pos = 2 if op.return_index else 1
            map_inverse_chunks = reduce_chunks[inverse_pos]
            inverse_shuffle_chunk = TensorShuffleProxy(
                dtype=map_inverse_chunks[0].dtype).new_chunk(map_inverse_chunks, shape=())
            inverse_chunks = []
            for j, cs in enumerate(unique_on_chunk_sizes):
                chunk_op = TensorUniqueInverseReduce(dtype=map_inverse_chunks[0].dtype,
                                                     shuffle_key=str(j))
                inverse_chunk = chunk_op.new_chunk([inverse_shuffle_chunk], shape=(cs,),
                                                   index=(j,))
                inverse_chunks.append(inverse_chunk)
            reduce_chunks[inverse_pos] = inverse_chunks

        kws = [out.params for out in op.outputs]
        for kw, chunks in zip(kws, reduce_chunks):
            kw['chunks'] = chunks
        unique_nsplits = list(inp.nsplits)
        unique_nsplits[op.axis] = (np.nan,) * len(reduce_chunks[0])
        kws[0]['nsplits'] = tuple(unique_nsplits)
        i = 1
        if op.return_index:
            kws[i]['nsplits'] = ((np.nan,) * len(reduce_chunks[i]),)
            i += 1
        if op.return_inverse:
            kws[i]['nsplits'] = (inp.nsplits[op.axis],)
            i += 1
        if op.return_counts:
            kws[i]['nsplits'] = ((np.nan,) * len(reduce_chunks[i]),)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, kws=kws)

    @classmethod
    def tile(cls, op):
        if len(op.inputs[0].chunks) == 1:
            return cls._tile_one_chunk(op)
        else:
            return cls._tile_via_shuffle(op)

    @classmethod
    def _execute_map(cls, ctx, op):
        (ar,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)
        n_reducer = op.aggregate_size

        with device(device_id):
            results = xp.unique(ar, return_index=op.return_index,
                                return_inverse=op.return_inverse,
                                return_counts=op.return_counts,
                                axis=op.axis)
            results = (results,) if not isinstance(results, tuple) else results
            results_iter = iter(results)
            unique_ar = next(results_iter)
            indices_ar = next(results_iter) + op.start_pos if op.return_index else None
            inverse_ar = next(results_iter) if op.return_inverse else None
            counts_ar = next(results_iter) if op.return_counts else None

            if xp is sparse:
                dense_xp = get_sparse_array_module(unique_ar)
            else:
                dense_xp = xp
            unique_index = dense_xp.arange(unique_ar.shape[op.axis]) \
                if inverse_ar is not None else None
            if unique_ar.size > 0:
                unique_reducers = dense_xp.asarray(
                    hash_on_axis(unique_ar, op.axis, n_reducer))
            else:
                unique_reducers = dense_xp.empty_like(unique_ar)
            ind_ar = dense_xp.arange(ar.shape[op.axis])

            for reducer in range(n_reducer):
                res = []
                cond = unique_reducers == reducer
                # unique
                slc = (slice(None),) * op.axis + (cond,)
                res.append(unique_ar[slc])
                # indices
                if indices_ar is not None:
                    res.append(indices_ar[cond])
                # inverse
                if inverse_ar is not None:
                    index_selected = unique_index[cond]
                    inv_cond = xp.isin(inverse_ar, index_selected)
                    inv_selected = xp.searchsorted(index_selected, inverse_ar[inv_cond])
                    ind_selected = ind_ar[inv_cond]
                    res.append(xp.stack([ind_selected, inv_selected]))
                # counts
                if counts_ar is not None:
                    res.append(counts_ar[cond])
                ctx[(op.outputs[0].key, str(reducer))] = tuple(res)

    @classmethod
    def _execute_reduce(cls, ctx, op):
        in_chunk = op.inputs[0]
        input_keys, input_indexes = get_shuffle_input_keys_idxes(in_chunk)

        inputs = list(zip(*(ctx[(input_key, str(op.aggregate_id))]
                            for input_key in input_keys)))
        flatten, device_id, xp = as_same_device(
            list(itertools.chain(*inputs)), device=op.device, ret_extra=True)
        n_ret = len(inputs[0])
        inputs = [flatten[i * n_ret: (i + 1) * n_ret] for i in range(len(inputs))]

        inputs_iter = iter(inputs)
        unique_arrays = next(inputs_iter)
        indices_arrays = next(inputs_iter) if op.return_index else None
        inverse_arrays = next(inputs_iter) if op.return_inverse else None
        counts_arrays = next(inputs_iter) if op.return_counts else None

        with device(device_id):
            ar = xp.concatenate(unique_arrays, axis=op.axis)
            result_return_inverse = op.return_inverse or op.return_counts
            axis = op.axis
            if ar.size == 0 or ar.shape[axis] == 0:
                # empty array on the axis
                results = [xp.empty(ar.shape)]
                i = 1
                for it in (op.return_index, op.return_inverse, op.return_counts):
                    if it:
                        results.append(xp.empty([], dtype=op.outputs[i].dtype))
                        i += 1
                results = tuple(results)
            else:
                results = xp.unique(ar, return_index=op.return_index,
                                    return_inverse=result_return_inverse,
                                    axis=axis)
            results = (results,) if not isinstance(results, tuple) else results
            results_iter = iter(results)
            outputs_iter = iter(op.outputs)
            # unique array
            ctx[next(outputs_iter).key] = next(results_iter)

            if op.output_limit == 1:
                return

            # calc indices
            if op.return_index:
                ctx[next(outputs_iter).key] = \
                    xp.concatenate(indices_arrays)[next(results_iter)]
            # calc inverse
            try:
                inverse_result = next(results_iter)
                if op.return_inverse:
                    unique_sizes = tuple(ua.shape[op.axis] for ua in unique_arrays)
                    cum_unique_sizes = np.cumsum((0,) + unique_sizes)
                    indices_out_key = next(outputs_iter).key
                    for i, inverse_array in enumerate(inverse_arrays):
                        p = inverse_result[cum_unique_sizes[i]: cum_unique_sizes[i + 1]]
                        r = xp.empty(inverse_array.shape, dtype=inverse_array.dtype)
                        if inverse_array.size > 0:
                            r[0] = inverse_array[0]
                            r[1] = p[inverse_array[1]]
                        # return unique length and
                        ctx[(indices_out_key, str(input_indexes[i][op.axis]))] = \
                            results[0].shape[op.axis], r
                # calc counts
                if op.return_counts:
                    result_counts = xp.zeros(results[0].shape[op.axis], dtype=int)
                    t = np.stack([inverse_result, np.concatenate(counts_arrays)])

                    def acc(a):
                        i, v = a
                        result_counts[i] += v

                    np.apply_along_axis(acc, 0, t)
                    ctx[next(outputs_iter).key] = xp.asarray(result_counts)
            except StopIteration:
                pass

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        elif op.stage == OperandStage.reduce:
            cls._execute_reduce(ctx, op)
        else:
            (ar,), device_id, xp = as_same_device(
                [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

            with device(device_id):
                kw = dict(return_index=op.return_index,
                          return_inverse=op.return_inverse,
                          return_counts=op.return_counts)
                if ar.dtype != object:
                    # axis cannot pass when dtype is object
                    kw['axis'] = op.axis
                results = xp.unique(ar, **kw)
                outs = op.outputs
                if len(outs) == 1:
                    ctx[outs[0].key] = results
                    return

                assert len(outs) == len(results)
                for out, result in zip(outs, results):
                    ctx[out.key] = result


class TensorUniqueInverseReduce(TensorMapReduceOperand, TensorOperandMixin):
    _op_type_ = OperandDef.UNIQUE_INVERSE_REDUCE

    def __init__(self, shuffle_key=None, dtype=None, gpu=None, **kw):
        super().__init__(_stage=OperandStage.reduce, _shuffle_key=shuffle_key,
                         _dtype=dtype, _gpu=gpu, **kw)

    @classmethod
    def execute(cls, ctx, op):
        out = op.outputs[0]
        input_keys, _ = get_shuffle_input_keys_idxes(op.inputs[0])
        inputs = [ctx[(inp_key, op.shuffle_key)] for inp_key in input_keys]
        unique_sizes = [inp[0] for inp in inputs]
        cum_unique_sizes = np.cumsum([0] + unique_sizes)
        invs, device_id, xp = as_same_device([inp[1] for inp in inputs],
                                             device=op.device, ret_extra=True)
        with device(device_id):
            ret = xp.empty(out.shape, dtype=out.dtype)
            for i, inv in enumerate(invs):
                ret[inv[0]] = cum_unique_sizes[i] + inv[1]
            ctx[out.key] = ret


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
    >>> import mars.tensor as mt

    >>> mt.unique([1, 1, 2, 2, 3, 3]).execute()
    array([1, 2, 3])
    >>> a = mt.array([[1, 1], [2, 3]])
    >>> mt.unique(a).execute()
    array([1, 2, 3])

    Return the unique rows of a 2D tensor

    >>> a = mt.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    >>> mt.unique(a, axis=0).execute()
    array([[1, 0, 0], [2, 3, 4]])

    Return the indices of the original tensor that give the unique values:

    >>> a = mt.array(['a', 'b', 'b', 'c', 'a'])
    >>> u, indices = mt.unique(a, return_index=True)
    >>> u.execute()
    array(['a', 'b', 'c'],
           dtype='|S1')
    >>> indices.execute()
    array([0, 1, 3])
    >>> a[indices].execute()
    array(['a', 'b', 'c'],
           dtype='|S1')

    Reconstruct the input array from the unique values:

    >>> a = mt.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = mt.unique(a, return_inverse=True)
    >>> u.execute()
    array([1, 2, 3, 4, 6])
    >>> indices.execute()
    array([0, 1, 4, 3, 1, 2, 1])
    >>> u[indices].execute()
    array([1, 2, 6, 4, 2, 3, 2])
    """
    op = TensorUnique(return_index=return_index, return_inverse=return_inverse,
                      return_counts=return_counts, axis=axis,
                      aggregate_size=aggregate_size)
    return op(ar)
