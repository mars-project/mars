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


import operator
import heapq
import itertools

import numpy as np

from ....config import options
from ....compat import six, izip, lzip, reduce
from ....operands import Rechunk
from ..utils import decide_chunk_sizes, calc_sliced_size
from ..core import TensorOperandMixin


class TensorRechunk(Rechunk, TensorOperandMixin):
    def __init__(self, chunk_size=None, threshold=None, chunk_size_limit=None, dtype=None, sparse=False, **kw):
        super(TensorRechunk, self).__init__(_chunk_size=chunk_size, _threshold=threshold,
                                            _chunk_size_limit=chunk_size_limit, _dtype=dtype, _sparse=sparse, **kw)

    def _set_inputs(self, inputs):
        super(TensorRechunk, self)._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, tensor):
        return self.new_tensor([tensor], tensor.shape)

    @classmethod
    def tile(cls, op):
        new_chunk_size = op.chunk_size
        steps = plan_rechunks(op.inputs[0], new_chunk_size,
                              threshold=op.threshold,
                              chunk_size_limit=op.chunk_size_limit)
        tensor = op.outputs[0]
        for c in steps:
            tensor = compute_rechunk(tensor.inputs[0], c)

        return [tensor]


def rechunk(tensor, chunk_size, threshold=None, chunk_size_limit=None):
    chunk_size = _get_nsplits(tensor, chunk_size)
    if chunk_size == tensor.nsplits:
        return tensor

    op = TensorRechunk(chunk_size, threshold, chunk_size_limit,
                       dtype=tensor.dtype, sparse=tensor.issparse())
    return op(tensor)


# -----------------------------------------------------------------------------------
# We adopt the iterative rechunk logic from dask, see: https://github.com/dask/dask
# -----------------------------------------------------------------------------------


def _get_nsplits(tensor, new_chunk_size):
    if isinstance(new_chunk_size, dict):
        chunk_size = list(tensor.nsplits)
        for idx, c in six.iteritems(new_chunk_size):
            chunk_size[idx] = c
    else:
        chunk_size = new_chunk_size

    return decide_chunk_sizes(tensor.shape, chunk_size, tensor.dtype.itemsize)


def _largest_chunk_size(nsplits):
    return reduce(operator.mul, map(max, nsplits))


def _chunk_number(nsplits):
    return reduce(operator.mul, map(len, nsplits))


def _estimate_graph_size(old_chunk_size, new_chunk_size):
    return reduce(operator.mul,
                  (len(oc) + len(nc) for oc, nc in zip(old_chunk_size, new_chunk_size)))


def plan_rechunks(tensor, new_chunk_size, threshold=None, chunk_size_limit=None):
    threshold = threshold or options.tensor.rechunk.threshold
    chunk_size_limit = chunk_size_limit or options.tensor.rechunk.chunk_size_limit

    if len(new_chunk_size) != tensor.ndim:
        raise ValueError('Provided chunks should have %d dimensions, got %d' % (
            tensor.ndim, len(new_chunk_size)))

    steps = []

    chunk_size_limit /= tensor.dtype.itemsize
    chunk_size_limit = max([int(chunk_size_limit),
                            _largest_chunk_size(tensor.nsplits),
                            _largest_chunk_size(new_chunk_size)])

    graph_size_threshold = threshold * (_chunk_number(tensor.nsplits) + _chunk_number(new_chunk_size))

    chunk_size = curr_chunk_size = tensor.nsplits
    first_run = True
    while True:
        graph_size = _estimate_graph_size(chunk_size, new_chunk_size)
        if graph_size < graph_size_threshold:
            break
        if not first_run:
            chunk_size = _find_split_rechunk(curr_chunk_size, new_chunk_size, graph_size * threshold)
        chunks_size, memory_limit_hit = _find_merge_rechunk(chunk_size, new_chunk_size, chunk_size_limit)
        if chunk_size == curr_chunk_size or chunk_size == new_chunk_size:
            break
        steps.append(chunk_size)
        curr_chunk_size = chunk_size
        if not memory_limit_hit:
            break
        first_run = False

    return steps + [new_chunk_size]


def _find_split_rechunk(old_chunk_size, new_chunk_size, graph_size_limit):
    """
        Find an intermediate rechunk that would merge some adjacent blocks
        together in order to get us nearer the *new_chunk_size* target, without
        violating the *graph_size_limit* (in number of elements).
        """
    ndim = len(old_chunk_size)

    old_largest_width = [max(c) for c in old_chunk_size]
    new_largest_width = [max(c) for c in new_chunk_size]

    graph_size_effect = {
        dim: len(nc) / len(oc)
        for dim, (oc, nc) in enumerate(zip(old_chunk_size, new_chunk_size))
    }

    block_size_effect = {
        dim: new_largest_width[dim] / old_largest_width[dim]
        for dim in range(ndim)
    }

    # Our goal is to reduce the number of nodes in the rechunk graph
    # by merging some adjacent chunks, so consider dimensions where we can
    # reduce the # of chunks
    merge_candidates = [dim for dim in range(ndim)
                        if graph_size_effect[dim] <= 1.0]

    # Merging along each dimension reduces the graph size by a certain factor
    # and increases memory largest block size by a certain factor.
    # We want to optimize the graph size while staying below the given
    # graph_size_limit.  This is in effect a knapsack problem, except with
    # multiplicative values and weights.  Just use a greedy algorithm
    # by trying dimensions in decreasing value / weight order.
    def key(k):
        gse = graph_size_effect[k]
        bse = block_size_effect[k]
        if bse == 1:
            bse = 1 + 1e-9
        return np.log(gse) / np.log(bse)

    sorted_candidates = sorted(merge_candidates, key=key)

    largest_block_size = reduce(operator.mul, old_largest_width)

    chunk_size = list(old_chunk_size)
    memory_limit_hit = False

    for dim in sorted_candidates:
        # Examine this dimension for possible graph reduction
        new_largest_block_size = (
            largest_block_size * new_largest_width[dim] // old_largest_width[dim])
        if new_largest_block_size <= graph_size_limit:
            # Full replacement by new chunks is possible
            chunk_size[dim] = new_chunk_size[dim]
            largest_block_size = new_largest_block_size
        else:
            # Try a partial rechunk, dividing the new chunks into
            # smaller pieces
            largest_width = old_largest_width[dim]
            chunk_limit = int(graph_size_limit * largest_width / largest_block_size)
            c = _divide_to_width(new_chunk_size[dim], chunk_limit)
            if len(c) <= len(old_chunk_size[dim]):
                # We manage to reduce the number of blocks, so do it
                chunk_size[dim] = c
                largest_block_size = largest_block_size * max(c) // largest_width

            memory_limit_hit = True

    assert largest_block_size == _largest_chunk_size(chunk_size)
    assert largest_block_size <= graph_size_limit
    return tuple(chunk_size), memory_limit_hit


def _merge_to_number(desired_chunk_size, max_number):
    """ Minimally merge the given chunks so as to drop the number of
        chunks below *max_number*, while minimizing the largest width.
        """
    if len(desired_chunk_size) <= max_number:
        return desired_chunk_size

    distinct = set(desired_chunk_size)
    if len(distinct) == 1:
        # Fast path for homogeneous target, also ensuring a regular result
        w = distinct.pop()
        n = len(desired_chunk_size)
        total = n * w

        desired_width = total // max_number
        width = w * (desired_width // w)
        adjust = (total - max_number * width) // w

        return (width + w,) * adjust + (width,) * (max_number - adjust)

    nmerges = len(desired_chunk_size) - max_number

    heap = [(desired_chunk_size[i] + desired_chunk_size[i + 1], i, i + 1)
            for i in range(len(desired_chunk_size) - 1)]
    heapq.heapify(heap)

    chunk_size = list(desired_chunk_size)

    while nmerges > 0:
        # Find smallest interval to merge
        width, i, j = heapq.heappop(heap)
        # If interval was made invalid by another merge, recompute
        # it, re-insert it and retry.
        if chunk_size[j] == 0:
            j += 1
            while chunk_size[j] == 0:
                j += 1
            heapq.heappush(heap, (chunk_size[i] + chunk_size[j], i, j))
            continue
        elif chunk_size[i] + chunk_size[j] != width:
            heapq.heappush(heap, (chunk_size[i] + chunk_size[j], i, j))
            continue
        # Merge
        assert chunk_size[i] != 0
        chunk_size[i] = 0  # mark deleted
        chunk_size[j] = width
        nmerges -= 1

    return tuple(filter(None, chunk_size))


def _divide_to_width(desired_chunk_size, max_width):
    """ Minimally divide the given chunks so as to make the largest chunk
    width less or equal than *max_width*.
    """
    chunk_size = []
    for c in desired_chunk_size:
        nb_divides = int(np.ceil(c / max_width))
        for i in range(nb_divides):
            n = c // (nb_divides - i)
            chunk_size.append(n)
            c -= n
        assert c == 0
    return tuple(chunk_size)


def _find_merge_rechunk(old_chunk_size, new_chunk_size, chunk_size_limit):
    """
        Find an intermediate rechunk that would merge some adjacent blocks
        together in order to get us nearer the *new_chunks* target, without
        violating the *chunk_size_limit* (in number of elements).
        """
    ndim = len(old_chunk_size)

    old_largest_width = [max(c) for c in old_chunk_size]
    new_largest_width = [max(c) for c in new_chunk_size]

    graph_size_effect = {
        dim: len(nc) / len(oc)
        for dim, (oc, nc) in enumerate(zip(old_chunk_size, new_chunk_size))
    }

    block_size_effect = {
        dim: new_largest_width[dim] / old_largest_width[dim]
        for dim in range(ndim)
    }

    # Our goal is to reduce the number of nodes in the rechunk graph
    # by merging some adjacent chunks, so consider dimensions where we can
    # reduce the # of chunks
    merge_candidates = [dim for dim in range(ndim)
                        if graph_size_effect[dim] <= 1.0]

    # Merging along each dimension reduces the graph size by a certain factor
    # and increases memory largest block size by a certain factor.
    # We want to optimize the graph size while staying below the given
    # chunk_size_limit.  This is in effect a knapsack problem, except with
    # multiplicative values and weights.  Just use a greedy algorithm
    # by trying dimensions in decreasing value / weight order.
    def key(k):
        gse = graph_size_effect[k]
        bse = block_size_effect[k]
        if bse == 1:
            bse = 1 + 1e-9
        return np.log(gse) / np.log(bse)

    sorted_candidates = sorted(merge_candidates, key=key)

    largest_block_size = reduce(operator.mul, old_largest_width)

    chunk_size = list(old_chunk_size)
    memory_limit_hit = False

    for dim in sorted_candidates:
        # Examine this dimension for possible graph reduction
        new_largest_block_size = (
            largest_block_size * new_largest_width[dim] // old_largest_width[dim])
        if new_largest_block_size <= chunk_size_limit:
            # Full replacement by new chunks is possible
            chunk_size[dim] = new_chunk_size[dim]
            largest_block_size = new_largest_block_size
        else:
            # Try a partial rechunk, dividing the new chunks into
            # smaller pieces
            largest_width = old_largest_width[dim]
            chunk_limit = int(chunk_size_limit * largest_width / largest_block_size)
            c = _divide_to_width(new_chunk_size[dim], chunk_limit)
            if len(c) <= len(old_chunk_size[dim]):
                # We manage to reduce the number of blocks, so do it
                chunk_size[dim] = c
                largest_block_size = largest_block_size * max(c) // largest_width

            memory_limit_hit = True

    assert largest_block_size == _largest_chunk_size(chunk_size)
    assert largest_block_size <= chunk_size_limit
    return tuple(chunk_size), memory_limit_hit


def compute_rechunk(tensor, chunk_size):
    from ..indexing.slice import TensorSlice
    from ..merge.concatenate import TensorConcatenate

    nsplits = tensor.nsplits
    truncated = [[0, None] for _ in range(tensor.ndim)]
    result_slices = []
    for dim, old_chunk_size, new_chunk_size in izip(itertools.count(0), nsplits, chunk_size):
        slices = []
        for rest in new_chunk_size:
            dim_slices = []
            while rest > 0:
                old_idx, old_start = truncated[dim]
                old_size = old_chunk_size[old_idx] - (old_start or 0)
                if old_size < rest:
                    dim_slices.append((old_idx, slice(old_start, None, None)))
                    rest -= old_size
                    truncated[dim] = old_idx + 1, None
                else:
                    end = rest if old_start is None else rest + old_start
                    if end >= old_chunk_size[old_idx]:
                        truncated[dim] = old_idx + 1, None
                    else:
                        truncated[dim] = old_idx, end
                    end = None if end == old_chunk_size[old_idx] else end
                    dim_slices.append((old_idx, slice(old_start, end)))
                    rest -= old_size
            slices.append(dim_slices)
        result_slices.append(slices)

    result_chunks = []
    idxes = itertools.product(*[range(len(c)) for c in chunk_size])
    chunk_slices = itertools.product(*result_slices)
    chunk_shapes = itertools.product(*chunk_size)
    for idx, chunk_slice, chunk_shape in izip(idxes, chunk_slices, chunk_shapes):
        to_merge = []
        merge_idxes = itertools.product(*[range(len(i)) for i in chunk_slice])
        for merge_idx, index_slices in izip(merge_idxes, itertools.product(*chunk_slice)):
            chunk_index, chunk_slice = lzip(*index_slices)
            old_chunk = tensor.cix[chunk_index]
            merge_chunk_shape = tuple(calc_sliced_size(s, chunk_slice[0]) for s in old_chunk.shape)
            merge_chunk_op = TensorSlice(chunk_slice, dtype=old_chunk.dtype, sparse=old_chunk.op.sparse)
            merge_chunk = merge_chunk_op.new_chunk([old_chunk], merge_chunk_shape, index=merge_idx)
            to_merge.append(merge_chunk)
        if len(to_merge) == 1:
            chunk_op = to_merge[0].op.copy()
            out_chunk = chunk_op.new_chunk(to_merge[0].op.inputs, chunk_shape, index=idx)
            result_chunks.append(out_chunk)
        else:
            chunk_op = TensorConcatenate(dtype=to_merge[0].dtype, sparse=to_merge[0].op.sparse)
            out_chunk = chunk_op.new_chunk(to_merge, chunk_shape, index=idx)
            result_chunks.append(out_chunk)

    op = TensorRechunk(chunk_size, sparse=tensor.issparse())
    return op.new_tensor([tensor], tensor.shape, dtype=tensor.dtype,
                         nsplits=chunk_size, chunks=result_chunks)
