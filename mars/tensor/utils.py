#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import inspect
import itertools
import operator
from collections import OrderedDict
from collections.abc import Iterable
from functools import lru_cache, reduce, wraps
from math import ceil
from numbers import Integral
from typing import Dict, List, Union

import numpy as np
try:
    import tiledb
except (ImportError, OSError):  # pragma: no cover
    tildb = None

from ..core import ExecutableTuple, recursive_tile
from ..utils import lazy_import
from ..lib.mmh3 import hash_from_buffer

cp = lazy_import('cupy', globals=globals(), rename='cp')


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
        elif isinstance(chunk_size, int):
            chunk_size = (chunk_size,) * len(shape)

    if len(shape) != len(chunk_size):
        raise ValueError('Chunks must have the same dimemsion, '
                         f'got shape: {shape}, chunks: {chunk_size}')

    chunk_sizes = []
    for size, chunk in zip(shape, chunk_size):
        if isinstance(chunk, Iterable):
            if not isinstance(chunk, tuple):
                chunk = tuple(chunk)

            if sum(chunk) != size:
                raise ValueError('chunks shape should be of the same length, '
                                 f'got shape: {size}, chunks: {chunk}')
            chunk_sizes.append(chunk)
        else:
            assert isinstance(chunk, int)

            if size == 0:
                sizes = (0,)
            else:
                sizes = tuple(chunk for _ in range(int(size / chunk))) + \
                    (tuple() if size % chunk == 0 else (size % chunk,))
            chunk_sizes.append(sizes)

    return tuple(chunk_sizes)


def broadcast_shape(*shapes):
    if len(shapes) == 1:
        return shapes[0]

    out_shapes = []
    for ss in itertools.zip_longest(*[reversed(s) for s in shapes], fillvalue=-1):
        shape = max(s for s in ss if s != -1)
        if any(i != -1 and i != 1 and i != shape and not np.isnan(i) for i in ss):
            raise ValueError('Operands could not be broadcast together '
                             'with shape {0}'.format(' '.join(map(str, shapes))))
        out_shapes.append(shape)
    return tuple(reversed(out_shapes))


def get_chunk_slices(nsplits, idx):
    return tuple(slice(sum(nsplit[:idx]), sum(nsplit[:idx + 1]))
                 for idx, nsplit in zip(idx, nsplits))


def gen_random_seeds(n, random_state):
    assert isinstance(random_state, np.random.RandomState)
    return tuple(np.frombuffer(random_state.bytes(n * 4), dtype=np.uint32).tolist())


def validate_axis(ndim, axis, argname=None):
    if axis >= ndim or axis < -ndim:
        raise np.AxisError(axis, ndim=ndim, msg_prefix=argname)

    return axis if axis >= 0 else ndim + axis


def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    """
    Normalizes an axis argument into a tuple of non-negative integer axes.

    This handles shorthands such as ``1`` and converts them to ``(1,)``,
    as well as performing the handling of negative indices covered by
    `normalize_axis_index`.

    By default, this forbids axes from being specified multiple times.

    Used internally by multi-axis-checking logic.

    Parameters
    ----------
    axis : int, iterable of int
        The un-normalized index or indices of the axis.
    ndim : int
        The number of dimensions of the array that `axis` should be normalized
        against.
    argname : str, optional
        A prefix to put before the error message, typically the name of the
        argument.
    allow_duplicate : bool, optional
        If False, the default, disallow an axis from being specified twice.

    Returns
    -------
    normalized_axes : tuple of int
        The normalized axis index, such that `0 <= normalized_axis < ndim`

    Raises
    ------
    AxisError
        If any axis provided is out of range
    ValueError
        If an axis is repeated

    See also
    --------
    normalize_axis_index : normalizing a single scalar axis
    """
    # Optimization to speed-up the most common cases.
    if type(axis) not in (tuple, list):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass
    # Going via an iterator directly is slower than via list comprehension.
    axis = tuple([validate_axis(ndim, ax, argname) for ax in axis])
    if not allow_duplicate and len(set(axis)) != len(axis):
        if argname:
            raise ValueError(f'repeated axis in `{argname}` argument')
        else:
            raise ValueError('repeated axis')
    return axis


def validate_order(dtype, order):
    if getattr(dtype, 'fields', None) is None:
        if order is not None:
            raise ValueError('Cannot specify order when the array has no fields')
        else:
            return

    need_check = True
    if order is None:
        order = list(dtype.names)
        need_check = False
    elif isinstance(order, (list, tuple)):
        order = list(order)
    else:
        order = [order]
    if need_check:
        for o in order:
            if o not in dtype.fields:
                raise ValueError(f'unknown field name: {o}')
    return order


def inject_dtype(dtype):
    def inner(func):
        @wraps(func)
        def call(*tensors, **kw):
            kw['dtype'] = np.dtype(dtype)
            ret = func(*tensors, **kw)
            if ret is NotImplemented:
                reverse_func = getattr(inspect.getmodule(func), f'r{func.__name__}', None)
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

    tensor_ufunc = '__tensor_ufunc__'

    def is_arg(arg):
        if hasattr(arg, tensor_ufunc):
            return False
        return hasattr(arg, 'ndim') and hasattr(arg, 'dtype')

    def inner(func):
        @wraps(func)
        def h(*tensors, **kw):
            usr_dtype = np.dtype(kw.pop('dtype')) if 'dtype' in kw else None
            args = [make_arg(t) if is_arg(t) else t for t in tensors]
            if reverse:
                args = args[::-1]
            np_kw = dict((k, make_arg(v) if hasattr(v, 'op') else v) for k, v in kw.items()
                         if is_arg(v) and k != 'out')

            dtype = None
            if not any(hasattr(arg, tensor_ufunc)
                       for arg in itertools.chain(args, np_kw.values())):
                # skip infer if encounter mars DataFrame etc
                # that implements __tensor_ufunc__
                try:
                    with np.errstate(all='ignore'):
                        dtype = np_func(*args, **np_kw).dtype
                except:  # noqa: E722
                    dtype = None

            if usr_dtype and dtype:
                can_cast_kwargs = {}
                if kw.get('casting') is not None:
                    can_cast_kwargs['casting'] = kw.get('casting')
                if check and not np.can_cast(dtype, usr_dtype, **can_cast_kwargs):
                    raise TypeError('No loop matching the specified signature '
                                    f'and casting was found for ufunc {np_func}')
                kw['dtype'] = usr_dtype
            else:
                kw['dtype'] = dtype

            ret = func(*tensors, **kw)
            if ret is NotImplemented:
                reverse_func = getattr(inspect.getmodule(func), f'r{func.__name__}', None) \
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
    from .core import Tensor

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
    return index[:illipsis_index] + (slice(None),) * n_extra + index[illipsis_index + 1:]


def calc_sliced_size(size, sliceobj):
    if np.isnan(size):
        return np.nan

    start, stop, step = sliceobj.indices(size)
    return int(ceil(abs((stop - start) / float(step))))


def calc_object_length(obj, size=None):
    if np.isscalar(obj):
        return 1
    elif isinstance(obj, slice):
        return calc_sliced_size(size, obj)
    else:
        return len(obj)


def slice_split(index: Union[int, slice],
                sizes: List[int]) -> Dict[int, Union[int, slice]]:
    size = sum(sizes)

    if isinstance(index, Integral):
        index = index if index >= 0 else size + index
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
    arr = np.asarray(arr)
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
            raise IndexError(f'index {idx} is out of bounds with size {cum_nsplit[-1]}')

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
            filtered[i] = filtered[i] - (cum_nsplits[i][idx[i] - 1] if idx[i] > 0 else 0)
        chunk_index_to_indexes[idx] = filtered
        chunk_index_to_poses[idx] = poses[cond]

    if ret_is_asc:
        return chunk_index_to_indexes, chunk_index_to_poses, chunk_idxes_asc
    return chunk_index_to_indexes, chunk_index_to_poses


def calc_pos(fancy_index_shape, pos, xp=np):
    if isinstance(pos, dict):
        pos = xp.concatenate(list(pos.values()))
    select_pos = xp.empty(fancy_index_shape, dtype=int)
    select_pos.flat[pos] = xp.arange(select_pos.size)
    return select_pos


def decide_unify_split(*splits):
    # TODO (jisheng): In the future, we need more sophisticated way to decide the rechunk split
    # right now, for (2, 2) and (3, 1), we get the rechunk split as (2, 1, 1)
    if not splits:
        return ()
    raw_splits = splits
    # support broadcasting rules
    # decide_unify_splits((1,), (5,))  --> (5,)
    splits = set(s for s in splits if ((len(s) > 1) or (len(s) == 1 and s[0] != 1)))
    if len(splits) == 1:
        return splits.pop()
    if len(splits) == 0:
        return raw_splits[0]

    if any(np.isnan(sum(s)) for s in splits):
        raise ValueError(f'Tensor chunk sizes are unknown: {splits}')
    if len(set(sum(s) for s in splits)) > 1:
        raise ValueError(f'Splits not of same size: {splits}')

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
    tensor_splits = [dict((a, split) for a, split in zip(axes, t.nsplits) if split != (1,))
                     for t, axes in tensor_axes if t.nsplits]
    common_axes = reduce(operator.and_, [set(ts.keys()) for ts in tensor_splits]) if tensor_splits else set()
    axes_unified_splits = dict((ax, decide_unify_split(*(t[ax] for t in tensor_splits)))
                               for ax in common_axes)

    if len(common_axes) == 0:
        return tuple(t[0] for t in tensor_axes)

    res = []
    for t, axes in tensor_axes:
        new_chunk = dict((i, axes_unified_splits[ax]) for ax, i in zip(axes, range(t.ndim))
                         if ax in axes_unified_splits)
        t = yield from recursive_tile(t.rechunk(new_chunk))
        res.append(t)

    return tuple(res)


def unify_chunks(*tensors):
    tensor_axes = [(t, range(t.ndim)) if not isinstance(t, tuple) else t
                   for t in tensors]

    if len(tensor_axes) < 2:
        return tuple(t[0] if isinstance(t, tuple) else t for t in tensors)

    return (yield from unify_nsplits(*tensor_axes))


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
        raise TypeError(f"output (typecode '{t.dtype.char}') could not be coerced "
                        f"to provided output parameter (typecode '{out.dtype.char}') "
                        f"according to the casting rule ''{casting}''")


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
        elif isinstance(chunk_size, int):
            chunk_size = {i: chunk_size for i in range(len(shape))}
        else:
            raise TypeError(f'chunks must be iterable, got {type(chunk_size)}')

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

    from ..config import options

    chunk_size = dictify_chunk_size(shape, chunk_size)
    nleft = len(shape) - len(chunk_size)
    if nleft < 0:
        raise ValueError("chunks have more dimensions than input tensor")
    if nleft == 0:
        return normalize_chunk_sizes(shape, tuple(chunk_size[j] for j in range(len(shape))))

    max_chunk_size = options.chunk_store_limit

    # normalize the dimension which specified first
    dim_to_normalized = {i: normalize_chunk_sizes((shape[i],), (c,))[0]
                         for i, c in chunk_size.items()}

    left = {j: [] for j in range(len(shape)) if j not in dim_to_normalized}
    left_unsplit = {j: shape[j] for j in left}
    while True:
        nbytes_occupied = np.prod([max(c) for c in dim_to_normalized.values()]) * itemsize
        dim_size = np.maximum(int(np.power(max_chunk_size / nbytes_occupied, 1 / float(len(left)))), 1)
        for j, ns in left.copy().items():
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
    raise ValueError(f'{seed} cannot be used to seed a mt.random.RandomState'
                     ' instance')


def filter_inputs(inputs):
    from ..core import ENTITY_TYPE

    return [inp for inp in inputs if isinstance(inp, ENTITY_TYPE)]


# As TileDB Ctx's creation is a bit time-consuming,
# we just cache the Ctx
# also remember the arguments should be hashable
@lru_cache(10)
def _create_tiledb_ctx(conf_tuple):
    if conf_tuple is not None:
        return tiledb.Ctx(dict(conf_tuple))
    return tiledb.Ctx()


def get_tiledb_ctx(conf):
    key = tuple(conf.items()) if conf is not None else None
    return _create_tiledb_ctx(key)


# this function is only used for pandas' compatibility
def to_numpy(pdf):
    try:
        return pdf.to_numpy()
    except AttributeError:  # pragma: no cover
        return pdf.values


def check_order(order_str, available_options='KACF',
                err_msg='order not understood'):
    order_str = order_str.upper()
    if order_str not in available_options:
        raise TypeError(err_msg)


def get_order(order_str, to_keep_order, available_options='KACF',
              err_msg='order not understood'):
    from .core import TensorOrder

    check_order(order_str, available_options=available_options,
                err_msg=err_msg)

    if order_str in 'KA':
        return to_keep_order
    elif order_str == 'C':
        return TensorOrder.C_ORDER
    else:
        return TensorOrder.F_ORDER


def reverse_order(old_order):
    from .core import TensorOrder

    assert isinstance(old_order, TensorOrder)
    return TensorOrder.C_ORDER if old_order == TensorOrder.F_ORDER else TensorOrder.F_ORDER


def hash_on_axis(ar, axis, n_dest):
    ar = np.asarray(ar)
    # cannot be scalar
    assert ar.ndim > 0
    axis = validate_axis(ar.ndim, axis)

    if n_dest == 1:
        return np.zeros(ar.shape[axis], dtype=np.uint32)

    if ar.ndim > 2:
        ret = np.empty(ar.shape[axis], dtype=np.uint32)

        def _hash_to_dest(data):
            i = data[0]
            idx = (slice(None),) * axis + (i,)
            ret[i] = hash_from_buffer(memoryview(ar[idx])) % n_dest

        np.apply_along_axis(_hash_to_dest, 0, np.arange(ar.shape[axis])[np.newaxis, :])
        return ret
    else:
        def _hash_to_dest(data):
            return hash_from_buffer(memoryview(data)) % n_dest

        if ar.ndim == 1:
            ar = ar.reshape(ar.size, 1)
        return np.apply_along_axis(_hash_to_dest, 1 - axis, ar)


def fetch_corner_data(tensor, session=None):
    print_option = np.get_printoptions()
    # only fetch corner data when data > threshold
    threshold = print_option['threshold']
    # number of edge items to print
    edgeitems = print_option['edgeitems']

    # we fetch corner data based on the fact that
    # the tensor must have been executed,
    # thus the size could not be NaN
    if tensor.size > threshold:
        # two edges for each exis
        indices_iter = list(itertools.product(*(range(2) for _ in range(tensor.ndim))))
        corners = np.empty(shape=(2,) * tensor.ndim, dtype=object)
        shape = [0 for _ in range(tensor.ndim)]
        for indices in indices_iter:
            slc = []
            for ax, i in enumerate(indices):
                size = tensor.shape[ax]
                if size > edgeitems * 2 + 2:
                    # fetch two more elements
                    if i == 0:
                        slc.append(slice(edgeitems + 1))
                    else:
                        slc.append(slice(-edgeitems - 1, None))
                    shape[ax] += edgeitems + 1
                else:
                    i_sep = size // 2
                    if i == 0:
                        slc.append(slice(i_sep))
                        shape[ax] += i_sep
                    else:
                        slc.append(slice(i_sep, None))
                        shape[ax] += size - i_sep
            corners[indices] = tensor[tuple(slc)]
        # fetch together
        fetched = ExecutableTuple(corners.flat).fetch(session=session)
        for indices, f in zip(indices_iter, fetched):
            corners[indices] = f
        return np.block(corners.tolist())
    else:
        return tensor.fetch(session=session)


def implement_scipy(scipy_fun):
    import re
    import textwrap

    def wrapper(fun):
        if scipy_fun is None:
            return None
        if not fun.__doc__:
            doc_str = textwrap.dedent(scipy_fun.__doc__)
            lines = []
            for line in doc_str.splitlines(keepends=False):
                # skip function headers
                if line.startswith(scipy_fun.__name__ + '('):
                    continue
                # skip version marks
                if line.strip().startswith('.. versionadded::'):
                    continue
                # skip examples
                if line.strip() == 'Examples':
                    break
                lines.append(line)
            doc_str = '\n'.join(lines).strip()
            # remove trailing empty sections
            fun.__doc__ = re.sub(r'[A-Za-z]+\n-+$', '', doc_str).strip()
        return fun

    return wrapper
