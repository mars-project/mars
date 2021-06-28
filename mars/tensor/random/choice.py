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

from numbers import Integral

import numpy as np

from ... import opcodes as OperandDef
from ...config import options
from ...core import recursive_tile
from ...serialization.serializables import FieldTypes, AnyField, KeyField, \
    BoolField, TupleField
from ...utils import has_unknown_shape, ceildiv
from ..operands import TensorOperandMixin
from ..core import TENSOR_TYPE, TENSOR_CHUNK_TYPE, TensorOrder
from ..datasource import arange, array
from ..utils import decide_chunk_sizes, normalize_chunk_sizes, gen_random_seeds
from ..array_utils import as_same_device, device
from .core import TensorRandomOperand, RandomState


class TensorChoice(TensorRandomOperand, TensorOperandMixin):
    _op_type_ = OperandDef.RAND_CHOICE

    _a = AnyField('a')
    _size = TupleField('size', FieldTypes.int64)
    _replace = BoolField('replace')
    _p = KeyField('p')

    def __init__(self, a=None, size=None, replace=None, p=None,
                 seed=None, **kw):
        super().__init__(_a=a, _size=size, _replace=replace, _p=p,
                         seed=seed, **kw)

    @property
    def a(self):
        return self._a

    @property
    def size(self):
        return self._size

    @property
    def replace(self):
        return self._replace

    @property
    def p(self):
        return self._p

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if isinstance(self._a, (TENSOR_TYPE, TENSOR_CHUNK_TYPE)):
            self._a = self._inputs[0]
        if isinstance(self._p, (TENSOR_TYPE, TENSOR_CHUNK_TYPE)):
            self._p = self._inputs[-1]

    def __call__(self, a, p, chunk_size=None):
        inputs = []
        if isinstance(a, TENSOR_TYPE):
            inputs.append(a)
        if isinstance(p, TENSOR_TYPE):
            inputs.append(p)
        return self.new_tensor(inputs, shape=self._size,
                               raw_chunk_size=chunk_size,
                               order=TensorOrder.C_ORDER)

    @classmethod
    def _tile_one_chunk(cls, op, a, p):
        out = op.outputs[0]
        chunk_op = op.copy().reset_key()
        chunk_op._seed = gen_random_seeds(1, np.random.RandomState(op.seed))[0]
        chunk_inputs = []
        if isinstance(a, TENSOR_TYPE):
            chunk_op._a = a.chunks[0]
            chunk_inputs.append(chunk_op.a)
        else:
            chunk_op._a = a
        if isinstance(p, TENSOR_TYPE):
            chunk_op._p = p.chunks[0]
            chunk_inputs.append(chunk_op.p)
        else:
            chunk_op._p = p
        chunk = chunk_op.new_chunk(chunk_inputs, shape=out.shape,
                                   index=(0,) * out.ndim,
                                   order=out.order)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=out.shape,
                                  order=out.order, chunks=[chunk],
                                  nsplits=tuple((s,) for s in out.shape))

    @classmethod
    def _tile_sample_with_replacement(cls, op, a, nsplits):
        out_shape = tuple(sum(ns) for ns in nsplits)
        out_size = np.prod(out_shape).item()
        most_chunk_size = np.prod([max(ns) for ns in nsplits]).item()

        is_a_int = False
        if isinstance(a, Integral):
            is_a_int = True
            a_size = a
        else:
            a = array(a)
            a_size = a.size

        rs = RandomState.from_numpy(np.random.RandomState(op.seed))

        if is_a_int:
            # the indices is just the result
            ret = rs.randint(a_size, size=out_shape, chunk_size=nsplits)
        else:
            # gen indices first, need to be flattened
            indices = rs.randint(a_size, size=out_size, chunk_size=most_chunk_size)
            # get result via fancy indexing
            ret = a[indices]
            if len(out_shape) > 1:
                # reshape back if out's ndim > 1
                ret = ret.reshape(out_shape)
            ret = ret.rechunk(nsplits)

        return [(yield from recursive_tile(ret))]

    @classmethod
    def _tile_sample_without_replacement(cls, op, a, nsplits):
        from ..base import searchsorted
        from ..merge.stack import TensorStack
        from ..indexing.getitem import TensorIndex

        out = op.outputs[0]
        out_shape = tuple(sum(ns) for ns in nsplits)
        # to sample count
        m = np.prod(out_shape).item()

        if isinstance(a, Integral):
            a_size = a
            a = arange(a)
        else:
            a = array(a)
            a_size = a.size
        a = yield from recursive_tile(a)

        if any(cs < m for cs in a.nsplits[0]):
            # make sure all chunk > m
            n_chunk = min(max(a.size // (m + 1), 1), a.chunk_shape[0])
            chunk_size = ceildiv(a.size, n_chunk)
            chunk_sizes = normalize_chunk_sizes(a.size, chunk_size)[0]
            if chunk_sizes[-1] < m and len(chunk_sizes) > 1:
                # the last chunk may still less than m
                # merge it into previous one
                chunk_sizes[-2] += chunk_sizes[-1]
                chunk_sizes = chunk_sizes[:-1]
            a = yield from recursive_tile(a.rechunk({0: chunk_sizes}))
            if len(chunk_sizes) == 1:
                return cls._tile_one_chunk(op, a, None)

        # for each chunk in a, do regular sampling
        sampled_chunks = []
        sample_seeds = gen_random_seeds(len(a.chunks), np.random.RandomState(op.seed))
        for seed, chunk in zip(sample_seeds, a.chunks):
            chunk_op = op.copy().reset_key()
            chunk_op._a = chunk
            chunk_op._size = (m,)
            chunk_op._seed = seed
            sampled_chunk = chunk_op.new_chunk([chunk], shape=(m,),
                                               order=out.order,
                                               index=chunk.index)
            sampled_chunks.append(sampled_chunk)

        if len(sampled_chunks) == 1:
            out_chunk = sampled_chunks[0]
        else:
            stacked_chunk = TensorStack(axis=0, dtype=sampled_chunks[0].dtype).new_chunk(
                sampled_chunks, shape=(len(a.chunks), m), order=TensorOrder.C_ORDER)

            # gen indices with length m from 0...a.size
            state = RandomState.from_numpy(np.random.RandomState(op.seed))
            indices = state.randint(a_size, size=(m,))
            cum_offsets = np.cumsum(a.nsplits[0])
            ind = yield from recursive_tile(searchsorted(cum_offsets, indices, side='right'))
            ind_chunk = ind.chunks[0]

            # do fancy index to find result
            arange_tensor = yield from recursive_tile(arange(m))
            indexes = [ind_chunk, arange_tensor.chunks[0]]
            out_chunk = TensorIndex(dtype=stacked_chunk.dtype, indexes=indexes).new_chunk(
                [stacked_chunk] + list(indexes), shape=(m,), order=TensorOrder.C_ORDER)

        ret = op.copy().new_tensor(op.inputs, shape=(m,), order=out.order,
                                   nsplits=((m,),), chunks=[out_chunk])
        if len(out_shape) > 0:
            ret = yield from recursive_tile(ret.reshape(out_shape))
        ret = yield from recursive_tile(ret.rechunk(nsplits))
        return [ret]

    @classmethod
    def tile(cls, op):
        if has_unknown_shape(*op.inputs):
            yield

        out = op.outputs[0]
        chunk_size = out.extra_params.raw_chunk_size or options.chunk_size
        nsplits = decide_chunk_sizes(out.shape, chunk_size, out.dtype.itemsize)
        inputs = op.inputs

        a, p = op.a, op.p
        if p is not None:
            # we cannot handle p in a parallel fashion
            inputs = []
            if isinstance(a, TENSOR_TYPE):
                a = yield from recursive_tile(a.rechunk(a.shape))
                inputs.append(a)
            p = yield from recursive_tile(p.rechunk(p.shape))
            inputs.append(p)

            # ignore nsplits if p is specified
            nsplits = ((s,) for s in out.shape)

        # all inputs and outputs has 1 chunk
        if all(len(inp.chunks) == 1 for inp in inputs) and \
                all(len(ns) == 1 for ns in nsplits):
            return cls._tile_one_chunk(op, a, p)

        if op.replace:
            return (yield from cls._tile_sample_with_replacement(
                op, a, nsplits))
        else:
            return (yield from cls._tile_sample_without_replacement(
                op, a, nsplits))

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)
        if isinstance(op.a, TENSOR_CHUNK_TYPE):
            a = inputs[0]
        else:
            a = op.a
        if isinstance(op.p, TENSOR_CHUNK_TYPE):
            p = inputs[-1]
        else:
            p = op.p

        with device(device_id):
            rs = xp.random.RandomState(op.seed)
            ctx[op.outputs[0].key] = rs.choice(
                a, size=op.size, replace=op.replace, p=p)


def choice(random_state, a, size=None, replace=True, p=None, chunk_size=None, gpu=None):
    """
    Generates a random sample from a given 1-D array

    Parameters
    -----------
    a : 1-D array-like or int
        If a tensor, a random sample is generated from its elements.
        If an int, the random sample is generated as if a were mt.arange(a)
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    replace : boolean, optional
        Whether the sample is with or without replacement
    p : 1-D array-like, optional
        The probabilities associated with each entry in a.
        If not given the sample assumes a uniform distribution over all
        entries in a.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default

    Returns
    --------
    samples : single item or tensor
        The generated random samples

    Raises
    -------
    ValueError
        If a is an int and less than zero, if a or p are not 1-dimensional,
        if a is an array-like of size 0, if p is not a vector of
        probabilities, if a and p have different lengths, or if
        replace=False and the sample size is greater than the population
        size

    See Also
    ---------
    randint, shuffle, permutation

    Examples
    ---------
    Generate a uniform random sample from mt.arange(5) of size 3:

    >>> import mars.tensor as mt

    >>> mt.random.choice(5, 3).execute()
    array([0, 3, 4])
    >>> #This is equivalent to mt.random.randint(0,5,3)

    Generate a non-uniform random sample from np.arange(5) of size 3:

    >>> mt.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0]).execute()
    array([3, 3, 0])

    Generate a uniform random sample from mt.arange(5) of size 3 without
    replacement:

    >>> mt.random.choice(5, 3, replace=False).execute()
    array([3,1,0])
    >>> #This is equivalent to np.random.permutation(np.arange(5))[:3]

    Generate a non-uniform random sample from mt.arange(5) of size
    3 without replacement:

    >>> mt.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0]).execute()
    array([2, 3, 0])

    Any of the above can be repeated with an arbitrary array-like
    instead of just integers. For instance:

    >>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
    >>> np.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
    array(['pooh', 'pooh', 'pooh', 'Christopher', 'piglet'],
          dtype='|S11')
    """

    if isinstance(a, Integral):
        if a <= 0:
            raise ValueError('a must be greater than 0')
        a_size = a
        dtype = np.random.choice(1, size=(), p=np.array([1]) if p is not None else p).dtype
    else:
        a = array(a)
        if a.ndim != 1:
            raise ValueError('a must be one dimensional')
        a_size = a.size
        dtype = a.dtype

    if p is not None:
        if not isinstance(p, TENSOR_TYPE):
            p = np.asarray(p)
            if not np.isclose(p.sum(), 1, rtol=1e-7, atol=0):
                raise ValueError('probabilities do not sum to 1')
            p = array(p, chunk_size=p.size)
        if p.ndim != 1:
            raise ValueError('p must be one dimensional')

    if size is None:
        size = ()
        length = 1
    else:
        try:
            tuple(size)
            length = np.prod(size)
        except TypeError:
            length = size
    if replace is False and length > a_size:
        raise ValueError("Cannot take a larger sample than population when 'replace=False'")

    size = random_state._handle_size(size)
    seed = gen_random_seeds(1, random_state.to_numpy())[0]
    op = TensorChoice(a=a, p=p, seed=seed,
                      replace=replace, size=size, dtype=dtype, gpu=gpu)
    return op(a, p, chunk_size=chunk_size)
