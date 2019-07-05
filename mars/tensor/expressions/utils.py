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

from collections import Iterable
from math import ceil
from numbers import Integral
import operator
import inspect
import itertools
from functools import wraps

import numpy as np

from ...compat import zip_longest, izip, six, reduce, lkeys, OrderedDict


def normalize_shape(shape):
    if isinstance(shape, Iterable):
        return tuple(shape)
    else:
        return shape,


def normalize_chunk_sizes(shape, chunk_size):
    shape = normalize_shape(shape)
    if not isinstance(chunk_size, tuple):
        if isinstance(chunk_size, Iterable):
            chunk_size = tuple(chunk_size)
        elif isinstance(chunk_size, six.integer_types):
            chunk_size = (chunk_size,) * len(shape)

    if len(shape) != len(chunk_size):
        raise ValueError('Chunks must have the same dimemsion, '
                         'got shape: %s, chunks: %s' % (shape, chunk_size))

    chunk_sizes = []
    for size, chunk in izip(shape, chunk_size):
        if isinstance(chunk, Iterable):
            if not isinstance(chunk, tuple):
                chunk = tuple(chunk)

            if sum(chunk) != size:
                raise ValueError('chunks shape should be of the same length, '
                                 'got shape: %s, chunks: %s' % (size, chunk))
            chunk_sizes.append(chunk)
        else:
            assert isinstance(chunk, six.integer_types)

            sizes = tuple(chunk for _ in range(int(size / chunk))) + \
                    (tuple() if size % chunk == 0 else (size % chunk,))
            chunk_sizes.append(sizes)

    return tuple(chunk_sizes)


def broadcast_shape(*shapes):
    if len(shapes) == 1:
        return shapes[0]

    out_shapes = []
    for ss in zip_longest(*[reversed(s) for s in shapes], fillvalue=-1):
        shape = max(ss)
        if any(i != -1 and i != 1 and i != shape and not np.isnan(i) for i in ss):
            raise ValueError('Operands could not be broadcast together '
                             'with shape {0}'.format(' '.join(map(str, shapes))))
        out_shapes.append(shape)
    return tuple(reversed(out_shapes))


def get_chunk_slices(nsplits, idx):
    return tuple(slice(sum(nsplit[:idx]), sum(nsplit[:idx+1]))
                 for idx, nsplit in zip(idx, nsplits))


def random_state_data(n, random_state=None):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    random_data = random_state.bytes(624 * n * 4)  # `n * 624` 32-bit integers
    l = list(np.frombuffer(random_data, dtype=np.uint32).reshape((n, -1)))
    assert len(l) == n
    return l


def validate_axis(ndim, axis):
    if axis >= ndim or axis < -ndim:
        raise np.AxisError(axis, ndim=ndim)

    return axis if axis >= 0 else ndim + axis


def inject_dtype(dtype):
    def inner(func):
        @wraps(func)
        def call(*tensors, **kw):
            kw['dtype'] = np.dtype(dtype)
            ret = func(*tensors, **kw)
            if ret is NotImplemented:
                reverse_func = getattr(inspect.getmodule(func), 'r{0}'.format(func.__name__), None)
                if reverse_func is not None:
                    ret = reverse_func(*tensors[::-1], **kw)
                if ret is NotImplemented:
                    raise TypeError(
                        "unsupported operand type(s) for {0}: '{1}' and '{2}".format(
                            func.__name__, *[type(t) for t in tensors]))
            return ret

        return call

    return inner


def infer_dtype(np_func, empty=True, reverse=False, check=True):
    def make_arg(arg):
        if empty:
            return np.empty((1,) * max(1, arg.ndim), dtype=arg.dtype)
        else:
            if hasattr(arg, 'op') and hasattr(arg.op, 'data'):
                arg = arg.op.data
            return arg[(0,) * max(1, arg.ndim)]

    def inner(func):
        @wraps(func)
        def h(*tensors, **kw):
            usr_dtype = np.dtype(kw.pop('dtype')) if 'dtype' in kw else None
            args = [make_arg(t) if hasattr(t, 'ndim') and hasattr(t, 'dtype') else t
                    for t in tensors]
            if reverse:
                args = args[::-1]
            np_kw = dict((k, make_arg(v) if hasattr(v, 'ndim') and hasattr(v, 'dtype') else v)
                         for k, v in six.iteritems(kw) if k != 'out')
            try:
                with np.errstate(all='ignore'):
                    dtype = np_func(*args, **np_kw).dtype
            except:  # noqa: E722
                dtype = None

            if usr_dtype and dtype:
                if check and not np.can_cast(dtype, usr_dtype):
                    raise TypeError(' No loop matching the specified signature '
                                    'and casting was found for ufunc %s' % np_func)
                kw['dtype'] = usr_dtype
            else:
                kw['dtype'] = dtype

            ret = func(*tensors, **kw)
            if ret is NotImplemented:
                reverse_func = getattr(inspect.getmodule(func), 'r{0}'.format(func.__name__), None) \
                    if not reverse else None
                if reverse_func is not None:
                    ret = reverse_func(*tensors[::-1], **kw)
                if ret is NotImplemented:
                    raise TypeError(
                        "unsupported operand type(s) for {0}: '{1}' and '{2}".format(
                            func.__name__, *[type(t) for t in tensors]))
            return ret

        return h

    return inner


def index_ndim(index):
    from ..core import Tensor

    if isinstance(index, Tensor) and index.dtype == np.bool_:
        # boolean indexing will occupy the ndim
        return index.ndim

    return 1 if index is not None else 0


def replace_ellipsis(index, ndim):
    all_illipsis = list(i for i, idx in enumerate(index) if idx is Ellipsis)
    if len(all_illipsis) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if not all_illipsis:
        return index

    illipsis_index = all_illipsis[0]
    n_extra = ndim - sum([index_ndim(i) for i in index]) + 1
    return index[:illipsis_index] + (slice(None),) * n_extra + index[illipsis_index+1:]


def calc_sliced_size(size, sliceobj):
    if np.isnan(size):
        return np.nan

    start, stop, step = sliceobj.indices(size)
    return int(ceil(abs((stop - start) / float(step))))


def slice_split(index, sizes):
    size = sum(sizes)

    if isinstance(index, Integral):
        i = 0
        ind = index
        lens = list(sizes)
        while ind >= lens[0]:
            i += 1
            ind -= lens.pop(0)
        return {i: ind}

    assert isinstance(index, slice)
    start, stop, step = index.indices(size)

    slice_all = slice(None)

    if index == slice_all:
        return dict((k, slice_all) for k in range(len(sizes)))

    d = dict()
    if step > 0:
        for i, length in enumerate(sizes):
            if start < length and stop > 0:
                d[i] = slice(start, min(stop, length), step)
                start = (start - length) % step
            else:
                start = start - length
            stop -= length
    else:
        rstart = start  # running start
        chunk_boundaries = np.cumsum(sizes)
        for i, chunk_stop in reversed(list(enumerate(chunk_boundaries))):
            # create a chunk start and stop
            if i == 0:
                chunk_start = 0
            else:
                chunk_start = chunk_boundaries[i - 1]

            # if our slice is in this chunk
            if (chunk_start <= rstart < chunk_stop) and (rstart > stop):
                d[i] = slice(rstart - chunk_stop,
                             max(chunk_start - chunk_stop - 1,
                                 stop - chunk_stop),
                             step)

                # compute the next running start point,
                offset = (rstart - (chunk_start - 1)) % step
                rstart = chunk_start + offset - 1

    # replace 0:20:1 with : if appropriate
    for k, v in d.items():
        if v == slice(0, sizes[k], 1):
            d[k] = slice(None, None, None)

    if not d:  # special case x[:0]
        d[0] = slice(0, 0, 1)

    return d


def is_asc_sorted(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if len(arr) == 0:
        return True
    return np.all(arr[:-1] <= arr[1:])


def split_indexes_into_chunks(nsplits, indexes, ret_is_asc=True):
    indexes = np.asarray(indexes)
    chunk_idxes = np.empty_like(indexes)
    cum_nsplits = [np.cumsum(nsplit) for nsplit in nsplits]
    for i, cum_nsplit, index in zip(itertools.count(0), cum_nsplits, indexes):
        # handle negative value in index
        if hasattr(index, 'flags') and not index.flags.writeable:
            index = index.copy()
        index = np.add(index, cum_nsplit[-1], out=index, where=index < 0)
        sorted_idx = np.argsort(index)

        if np.any(index >= cum_nsplit[-1]):
            idx = index[index >= cum_nsplit[-1]][0]
            raise IndexError('index {0} is out of bounds with size {1}'.format(
                idx, cum_nsplit[-1]))

        chunk_idx = np.searchsorted(cum_nsplit, index[sorted_idx], side='right')
        chunk_idxes[i, sorted_idx] = chunk_idx

    chunk_idxes_asc = False
    if ret_is_asc:
        chunk_idxes_asc = is_asc_sorted(np.lexsort(chunk_idxes[::-1]))

    chunk_index_to_indexes = OrderedDict()
    chunk_index_to_poses = OrderedDict()
    poses = np.arange(len(indexes[0]))
    for idx in itertools.product(*(range(len(nsplit)) for nsplit in nsplits)):
        cond = (chunk_idxes == np.array(idx).reshape((len(idx), 1))).all(axis=0)
        filtered = indexes[:, cond]
        for i in range(len(indexes)):
            filtered[i] = filtered[i] - (cum_nsplits[i][idx[i]-1] if idx[i] > 0 else 0)
        chunk_index_to_indexes[idx] = filtered
        chunk_index_to_poses[idx] = poses[cond]

    if ret_is_asc:
        return chunk_index_to_indexes, chunk_index_to_poses, chunk_idxes_asc
    return chunk_index_to_indexes, chunk_index_to_poses


def calc_pos(fancy_index_shape, pos):
    if isinstance(pos, dict):
        pos = np.concatenate(list(pos.values()))
    select_pos = np.empty(fancy_index_shape, dtype=int)
    select_pos.flat[pos] = np.arange(select_pos.size)
    return select_pos


def decide_unify_split(*splits):
    # TODO (jisheng): In the future, we need more sophisticated way to decide the rechunk split
    # right now, for (2, 2) and (3, 1), we get the rechunk split as (2, 1, 1)
    if not splits:
        return ()
    raw_splits = splits
    splits = set(s for s in splits if len(s) > 1)
    if len(splits) == 1:
        return splits.pop()
    if len(splits) == 0:
        return raw_splits[0]

    if any(np.isnan(sum(s)) for s in splits):
        raise ValueError('Tensor chunk sizes are unknown: {0}'.format(splits))
    if len(set(sum(s) for s in splits)) > 1:
        raise ValueError('Splits do not hava same size: {0}'.format(splits))

    q = [list(s) for s in splits]
    size = sum(q[0])
    cum = 0

    res = []
    while cum < size:
        m = min(s[0] for s in q)
        res.append(m)
        for s in q:
            s[0] -= m
            if s[0] == 0:
                s.pop(0)

        cum += m

    return tuple(res)


def unify_nsplits(*tensor_axes):
    from .rechunk import rechunk

    tensor_splits = [dict((a, split) for a, split in izip(axes, t.nsplits) if split != (1,))
                     for t, axes in tensor_axes]
    common_axes = reduce(operator.and_, [set(lkeys(ts)) for ts in tensor_splits])
    axes_unified_splits = dict((ax, decide_unify_split(*(t[ax] for t in tensor_splits)))
                               for ax in common_axes)

    if len(common_axes) == 0:
        return tuple(t[0] for t in tensor_axes)

    res = []
    for t, axes in tensor_axes:
        new_chunk = dict((i, axes_unified_splits[ax]) for ax, i in zip(axes, range(t.ndim))
                         if ax in axes_unified_splits)
        res.append(rechunk(t, new_chunk).single_tiles())

    return tuple(res)


def unify_chunks(*tensors):
    tensor_axes = [(t, range(t.ndim)) if not isinstance(t, tuple) else t
                   for t in tensors]

    if len(tensor_axes) < 2:
        return tuple(t[0] if isinstance(t, tuple) else t for t in tensors)

    return unify_nsplits(*tensor_axes)


def check_out_param(out, t, casting):
    from .base import broadcast_to

    if not hasattr(out, 'shape'):
        raise TypeError('return arrays must be a tensor')

    try:
        broadcast_to(t, out.shape)
    except ValueError:
        raise ValueError("operands could not be broadcast together "
                         "with shapes ({0}) ({1})".format(','.join(str(s) for s in t.shape),
                                                          ','.join(str(s) for s in out.shape)))

    if not np.can_cast(t.dtype, out.dtype, casting):
        raise TypeError("output (typecode '{0}') could not be coerced "
                        "to provided output paramter (typecode '{1}') "
                        "according to the casting rule ''{2}''".format(t.dtype.char, out.dtype.char, casting))


def recursive_tile(tensor):
    q = [tensor]
    while q:
        t = q[-1]
        cs = [c for c in t.inputs if c.is_coarse()]
        if cs:
            q.extend(cs)
            continue
        t.single_tiles()
        q.pop()

    return tensor


def dictify_chunk_size(shape, chunk_size):
    """
    Given chunk_size which may be a tuple or dict, return a dict type all the same.

    :param shape: tensor's shape
    :param chunk_size: if dict provided, it's dimension id to chunk size;
                       if provided, it's the chunk size for each dimension.
    :return: dict form of chunk_size
    """
    if chunk_size is not None:
        if isinstance(chunk_size, Iterable):
            if not isinstance(chunk_size, dict):
                chunk_size = {i: c for i, c in enumerate(chunk_size)}
        elif isinstance(chunk_size, six.integer_types):
            chunk_size = {i: chunk_size for i in range(len(shape))}
        else:
            raise TypeError('chunks must be iterable, got {0}'.format(type(chunk_size)))

    if chunk_size is None:
        chunk_size = dict()

    return chunk_size


def decide_chunk_sizes(shape, chunk_size, itemsize):
    """
    Decide how a given tensor can be split into chunk.

    :param shape: tensor's shape
    :param chunk_size: if dict provided, it's dimension id to chunk size;
                       if provided, it's the chunk size for each dimension.
    :param itemsize: element size
    :return: the calculated chunk size for each dimension
    :rtype: tuple
    """

    from ...config import options

    chunk_size = dictify_chunk_size(shape, chunk_size)
    nleft = len(shape) - len(chunk_size)
    if nleft < 0:
        raise ValueError("chunks have more dimensions than input tensor")
    if nleft == 0:
        return normalize_chunk_sizes(shape, tuple(chunk_size[j] for j in range(len(shape))))

    max_chunk_size = options.tensor.chunk_store_limit

    # normalize the dimension which specified first
    dim_to_normalized = {i: normalize_chunk_sizes((shape[i],), (c,))[0]
                         for i, c in six.iteritems(chunk_size)}

    left = {j: [] for j in range(len(shape)) if j not in dim_to_normalized}
    left_unsplit = {j: shape[j] for j in left}
    while True:
        nbytes_occupied = np.prod([max(c) for c in six.itervalues(dim_to_normalized)]) * itemsize
        dim_size = np.maximum(int(np.power(max_chunk_size / nbytes_occupied, 1 / float(len(left)))), 1)
        for j, ns in six.iteritems(left.copy()):
            unsplit = left_unsplit[j]
            ns.append(int(np.minimum(unsplit, dim_size)))
            left_unsplit[j] -= ns[-1]
            if left_unsplit[j] <= 0:
                dim_to_normalized[j] = tuple(ns)
                del left[j]

        if len(left) == 0:
            break

    return tuple(dim_to_normalized[i] for i in range(len(dim_to_normalized)))


def check_random_state(seed):
    """
    Turn seed into a mt.random.RandomState instance

    :param seed:
        If seed is None, return the RandomState singleton used by mt.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    :return:
    """
    from . import random as mtrand
    from numpy import random as np_mtrand

    if seed is None or seed is mtrand or seed is np_mtrand:
        return mtrand._random_state
    if isinstance(seed, (Integral, np.integer)):
        return mtrand.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return mtrand.RandomState.from_numpy(seed)
    if isinstance(seed, mtrand.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a mt.random.RandomState'
                     ' instance' % seed)


def concat_tileable_chunks(tensor):
    from .merge.concatenate import TensorConcatenate

    assert not tensor.is_coarse()

    op = TensorConcatenate(dtype=tensor.op.dtype)
    chunk = TensorConcatenate(dtype=op.dtype).new_chunk(
        tensor.chunks, shape=tensor.shape)
    return op.new_tensor([tensor], tensor.shape, chunks=[chunk],
                         nsplits=tuple((s,) for s in tensor.shape))


def create_fetch_tensor(chunk_size, shape, dtype, tensor_key=None, tensor_id=None, chunk_keys=None):
    from ...config import options
    from .fetch import TensorFetch

    # compute chunks
    chunk_size = chunk_size or options.tensor.chunk_size
    chunk_size = decide_chunk_sizes(shape, chunk_size, dtype.itemsize)
    chunk_size_idxes = (range(len(size)) for size in chunk_size)

    fetch_op = TensorFetch(dtype=dtype).reset_key()

    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)
    if chunk_keys is None:
        chunk_keys = itertools.repeat(None)

    chunks = []
    for chunk_shape, chunk_idx, chunk_key in izip(itertools.product(*chunk_size),
                                                  itertools.product(*chunk_size_idxes),
                                                  chunk_keys):
        chunk = fetch_op.copy().reset_key().new_chunk(None, shape=chunk_shape, index=chunk_idx,
                                                      _key=chunk_key)
        chunks.append(chunk)
    return fetch_op.copy().new_tensor(None, shape=shape, dtype=dtype, nsplits=chunk_size,
                                      chunks=chunks, _key=tensor_key, _id=tensor_id)


def setitem_as_records(nsplits_acc, output_chunk, value, ts):
    '''
    Turns a `__setitem__`  to a list of index-value records.

    Parameters:
    :arg nsplits_acc:
        Accumulate nsplits arrays of the output tensor chunks.

    :arg output_chunk:
        A chunk in the output of the `__setitem__` op.

    :arg value:
        The scalar or ndarray value that are set to the tensor.

    :arg ts:
        The timestamp value will be contained in the records.

    :returns:
        A list of `[index, value, timestamp]`.
    '''
    # prepare chunk value
    if np.isscalar(value):
        chunk_value = value
    else:
        chunk_value_slice = tuple(slice(nsplits_acc[i][output_chunk.index[i]],
                                        nsplits_acc[i][output_chunk.index[i] + 1])
                                    for i in range(len(output_chunk.index)))
        chunk_value = value[chunk_value_slice]

    input_chunk = output_chunk.op.input

    input_indices = []  # index in the chunk of the mutable tensor
    value_indices = []  # index in the chunk of the assigned value
    for d, s in zip(output_chunk.op.indexes, input_chunk.shape):
        # expand the index (slice)
        idx = np.r_[slice(*d.indices(s)) if isinstance(d, slice) else d]
        input_indices.append(idx)
        if not isinstance(d, Integral):
            value_indices.append(np.arange(len(idx)))

    records = []
    for chunk_idx, value_idx in zip(itertools.product(*input_indices),
                                    itertools.product(*value_indices)):
        if np.isscalar(chunk_value):
            new_value = chunk_value
        else:
            new_value = chunk_value[value_idx]
        records.append((np.ravel_multi_index(chunk_idx, input_chunk.shape), ts, new_value))
    return records


def get_fetch_op_cls(op):
    from ...operands import ShuffleProxy
    from .fetch import TensorFetchShuffle, TensorFetch
    if isinstance(op, ShuffleProxy):
        return TensorFetchShuffle
    else:
        return TensorFetch


def get_fuse_op_cls():
    from .fuse import TensorFuseChunk

    return TensorFuseChunk


def filter_inputs(inputs):
    from ...core import Base, Entity

    return [inp for inp in inputs if isinstance(inp, (Base, Entity))]
