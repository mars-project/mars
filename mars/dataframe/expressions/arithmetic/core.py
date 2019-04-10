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

from ...core import DATAFRAME_TYPE
from ..core import DataFrameOperandMixin
from ..utils import split_monotonic_index_min_max


class DataFrameBinOp(DataFrameOperandMixin):
    __slots__ = ()

    @classmethod
    def _check_overlap(cls, chunk_index_min_max):
        for j in range(len(chunk_index_min_max) - 1):
            # overlap only if the prev max is close and curr min is close
            # and they are identical
            prev_max, prev_max_close = chunk_index_min_max[j][2:]
            curr_min, curr_min_close = chunk_index_min_max[j + 1][:2]
            if prev_max_close and curr_min_close and prev_max == curr_min:
                return True
        return False

    @classmethod
    def _get_chunk_index_min_max(cls, df, index_type, axis):
        index = getattr(df, index_type)
        if not index.is_monotonic_increasing_or_decreasing:
            return

        chunk_index_min_max = []
        for i in range(df.chunk_shape[axis]):
            chunk_idx = [0, 0]
            chunk_idx[axis] = i
            chunk = df.cix[tuple(chunk_idx)]
            chunk_index = getattr(chunk, index_type)
            min_val = chunk_index.min_val
            min_val_close = chunk_index.min_val_close
            max_val = chunk_index.max_val
            max_val_close = chunk_index.max_val_close
            if not min_val or not max_val:
                return
            chunk_index_min_max.append((min_val, min_val_close, max_val, max_val_close))

        if index.is_monotonic_decreasing:
            chunk_index_min_max = list(reversed(chunk_index_min_max))

        if cls._check_overlap(chunk_index_min_max):
            return
        return chunk_index_min_max

    @classmethod
    def _tile_both_dataframes(cls, op):
        # if both of the inputs are DataFrames, axis is just ignored
        left, right = op.inputs
        nsplits = [[], []]

        # if both of their index are identical

        # first, we decide the chunk size on each axis
        # we perform the same logic for both index and columns
        for axis, index_type in enumerate(['index_value', 'columns']):
            # if both of the indexes are monotonic increasing or decreasing
            left_chunk_index_min_max = cls._get_chunk_index_min_max(left, index_type, axis)
            right_chunk_index_min_max = cls._get_chunk_index_min_max(right, index_type, axis)
            if left_chunk_index_min_max and right_chunk_index_min_max:
                # no need to do shuffle on this axis
                pass

    @classmethod
    def tile(cls, op):
        if all(isinstance(inp, DATAFRAME_TYPE) for inp in op.inputs):
            return cls._tile_both_dataframes(op)
