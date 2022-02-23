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

import itertools
from dataclasses import dataclass
from typing import Tuple, Union, List

import numpy as np

from ...typing import ChunkType, TileableType
from ..utils import decide_chunk_sizes


chunk_size_type = Union[int, Tuple[int], Tuple[Tuple[int], ...]]


def get_nsplits(
    tileable: TileableType, new_chunk_size: chunk_size_type, itemsize: int
) -> Tuple[Tuple[int], ...]:
    if isinstance(new_chunk_size, dict):
        chunk_size = list(tileable.nsplits)
        for idx, c in new_chunk_size.items():
            chunk_size[idx] = c
    else:
        chunk_size = new_chunk_size

    return decide_chunk_sizes(tileable.shape, chunk_size, itemsize)


@dataclass
class RechunkInfo:
    out_index: Tuple[int]
    shape: Tuple[int]
    input_chunks: List[ChunkType]
    input_slices: List[Tuple[slice]]
    input_chunk_shape: List[int]


def gen_rechunk_infos(
    inp: TileableType, chunk_size: Tuple[Tuple[int], ...]
) -> List[RechunkInfo]:
    cum_in_nsplits = [np.cumsum(ns) for ns in inp.nsplits]
    cum_out_nsplits = [np.cumsum(ns) for ns in chunk_size]
    out_starts = [[0] + cum_ns[:-1].tolist() for cum_ns in cum_out_nsplits]
    out_ends = cum_out_nsplits
    out_start_indexes = [
        np.searchsorted(cum_ns, starts)
        for cum_ns, starts in zip(cum_in_nsplits, out_starts)
    ]
    out_end_indexes = [
        np.searchsorted(cum_ns, ends) for cum_ns, ends in zip(cum_in_nsplits, out_ends)
    ]

    chunk_index_iter = itertools.product(*(range(len(s)) for s in chunk_size))
    rechunk_infos = []
    for chunk_index in chunk_index_iter:
        shape = tuple(chunk_size[dim][i] for dim, i in enumerate(chunk_index))
        inp_chunk_slices = [list() for _ in range(len(chunk_index))]
        inp_chunk_indexes = [list() for _ in range(len(chunk_index))]
        for dim, i in enumerate(chunk_index):
            size_start = out_starts[dim][i]
            size_end = out_ends[dim][i]
            start_index = out_start_indexes[dim][i]
            end_index = out_end_indexes[dim][i]
            for inp_i in range(start_index, end_index + 1):
                inp_start = cum_in_nsplits[dim][inp_i - 1] if inp_i > 0 else 0
                inp_end = cum_in_nsplits[dim][inp_i]
                slice_start = max(inp_start, size_start) - inp_start
                slice_end = min(inp_end, size_end) - inp_start
                if slice_start == 0 and slice_end == inp_end - inp_start:
                    # slice all
                    slc = slice(None)
                else:
                    slc = slice(slice_start, slice_end)
                inp_chunk_slices[dim].append(slc)
                inp_chunk_indexes[dim].append(inp_i)

        inp_chunks = []
        inp_slices = []
        rechunk_info = RechunkInfo(
            out_index=chunk_index,
            shape=shape,
            input_chunks=inp_chunks,
            input_slices=inp_slices,
            input_chunk_shape=list(len(s) for s in inp_chunk_indexes),
        )
        for inp_chunk_index, inp_chunk_slice in zip(
            itertools.product(*inp_chunk_indexes),
            itertools.product(*inp_chunk_slices),
        ):
            inp_chunk = inp.cix[tuple(inp_chunk_index)]
            inp_chunks.append(inp_chunk)
            inp_slices.append(inp_chunk_slice)
        rechunk_infos.append(rechunk_info)

    return rechunk_infos
