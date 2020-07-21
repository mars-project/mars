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


import operator
import itertools
from functools import reduce

import numpy as np

from ...config import options
from ..utils import decide_chunk_sizes

# -----------------------------------------------------------------------------------
# We adopt the iterative rechunk logic from dask, see: https://github.com/dask/dask
# -----------------------------------------------------------------------------------


def get_nsplits(tileable, new_chunk_size, itemsize):
    if isinstance(new_chunk_size, dict):
        chunk_size = list(tileable.nsplits)
        for idx, c in new_chunk_size.items():
            chunk_size[idx] = c
    else:
        chunk_size = new_chunk_size

    return decide_chunk_sizes(tileable.shape, chunk_size, itemsize)


def _largest_chunk_size(nsplits):
    return reduce(operator.mul, map(max, nsplits))


def _chunk_number(nsplits):
    return reduce(operator.mul, map(len, nsplits))


def _estimate_graph_size(old_chunk_size, new_chunk_size):
    return reduce(operator.mul,
                  (len(oc) + len(nc) for oc, nc in zip(old_chunk_size, new_chunk_size)))


def plan_rechunks(tileable, new_chunk_size, itemsize, threshold=None, chunk_size_limit=None):
    threshold = threshold or options.rechunk.threshold
    chunk_size_limit = chunk_size_limit or options.rechunk.chunk_size_limit

    if len(new_chunk_size) != tileable.ndim:
        raise ValueError('Provided chunks should have %d dimensions, got %d' % (
            tileable.ndim, len(new_chunk_size)))

    steps = []

    if itemsize > 0:
        chunk_size_limit /= itemsize
    chunk_size_limit = max([int(chunk_size_limit),
                            _largest_chunk_size(tileable.nsplits),
                            _largest_chunk_size(new_chunk_size)])

    graph_size_threshold = threshold * (_chunk_number(tileable.nsplits) + _chunk_number(new_chunk_size))

    chunk_size = curr_chunk_size = tileable.nsplits
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


def compute_rechunk_slices(tileable, chunk_size):
    nsplits = tileable.nsplits
    truncated = [[0, None] for _ in range(tileable.ndim)]
    result_slices = []
    for dim, old_chunk_size, new_chunk_size in zip(itertools.count(0), nsplits, chunk_size):
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
    return result_slices
