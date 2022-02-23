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

from typing import Tuple

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...core import OutputType
from ...serialization.serializables import AnyField
from ...tensor.rechunk.core import get_nsplits, gen_rechunk_infos, chunk_size_type
from ...typing import TileableType
from ...utils import has_unknown_shape
from ..initializer import DataFrame as asdataframe, Series as asseries, Index as asindex
from ..operands import DataFrameOperand, DataFrameOperandMixin, DATAFRAME_TYPE
from ..utils import indexing_index_value, merge_index_value


class DataFrameRechunk(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.RECHUNK

    chunk_size = AnyField("chunk_size")

    def __call__(self, x):
        if isinstance(x, DATAFRAME_TYPE):
            return self.new_dataframe(
                [x],
                shape=x.shape,
                dtypes=x.dtypes,
                columns_value=x.columns_value,
                index_value=x.index_value,
            )
        else:
            self.output_types = x.op.output_types
            f = (
                self.new_series
                if self.output_types[0] == OutputType.series
                else self.new_index
            )
            return f(
                [x],
                shape=x.shape,
                dtype=x.dtype,
                index_value=x.index_value,
                name=x.name,
            )

    @classmethod
    def tile(cls, op: "DataFrameRechunk"):
        from ..indexing.iloc import (
            DataFrameIlocGetItem,
            SeriesIlocGetItem,
            IndexIlocGetItem,
        )
        from ..merge.concat import DataFrameConcat

        if has_unknown_shape(*op.inputs):
            yield

        out = op.outputs[0]
        inp = op.inputs[0]
        if inp.ndim == 2:
            inp = asdataframe(inp)
        elif inp.op.output_types[0] == OutputType.series:
            inp = asseries(inp)
        else:
            inp = asindex(inp)
        chunk_size = _get_chunk_size(inp, op.chunk_size)
        if chunk_size == inp.nsplits:
            return [inp]

        rechunk_infos = gen_rechunk_infos(inp, chunk_size)
        out_chunks = []
        for rechunk_info in rechunk_infos:
            chunk_index = rechunk_info.out_index
            shape = rechunk_info.shape
            inp_chunks = rechunk_info.input_chunks
            inp_chunk_slices = rechunk_info.input_slices
            inp_slice_chunks = []
            for inp_chunk, inp_chunk_slice in zip(inp_chunks, inp_chunk_slices):
                if all(slc == slice(None) for slc in inp_chunk_slice):
                    inp_slice_chunks.append(inp_chunk)
                else:
                    index_value = indexing_index_value(
                        inp_chunk.index_value, inp_chunk_slice[0], rechunk=True
                    )
                    if inp_chunk.ndim == 1:
                        # Series or Index
                        slc_chunk_op_type = (
                            SeriesIlocGetItem
                            if op.output_types[0] == OutputType.series
                            else IndexIlocGetItem
                        )
                        slc_chunk = slc_chunk_op_type(
                            indexes=inp_chunk_slice,
                            output_types=op.output_types,
                            sparse=inp_chunk.op.sparse,
                        ).new_chunk(
                            [inp_chunk],
                            index_value=index_value,
                            dtype=inp_chunk.dtype,
                            name=inp_chunk.name,
                            index=inp_chunk.index,
                        )
                    else:
                        # DataFrame
                        columns_value = indexing_index_value(
                            inp_chunk.columns_value,
                            inp_chunk_slice[1],
                            store_data=True,
                            rechunk=True,
                        )
                        dtypes = inp_chunk.dtypes.iloc[inp_chunk_slice[1]]
                        slc_chunk = DataFrameIlocGetItem(
                            indexes=inp_chunk_slice,
                            output_types=[OutputType.dataframe],
                            sparse=inp_chunk.op.sparse,
                        ).new_chunk(
                            [inp_chunk],
                            index_value=index_value,
                            columns_value=columns_value,
                            dtypes=dtypes,
                            index=inp_chunk.index,
                        )
                    inp_slice_chunks.append(slc_chunk)

            chunk_shape = rechunk_info.input_chunk_shape
            inp_chunks_arr = np.empty(chunk_shape, dtype=object)
            inp_chunks_arr.ravel()[:] = inp_slice_chunks
            params = dict(index=chunk_index, shape=shape)
            if inp_chunks_arr.ndim == 1:
                params["index_value"] = merge_index_value(
                    {i: c.index_value for i, c in enumerate(inp_chunks_arr)}
                )
                params["dtype"] = inp_slice_chunks[0].dtype
                params["name"] = inp_slice_chunks[0].name
            else:
                params["index_value"] = merge_index_value(
                    {i: c.index_value for i, c in enumerate(inp_chunks_arr[:, 0])}
                )
                params["columns_value"] = merge_index_value(
                    {i: c.columns_value for i, c in enumerate(inp_chunks_arr[0])},
                    store_data=True,
                )
                params["dtypes"] = pd.concat([c.dtypes for c in inp_chunks_arr[0]])
            out_chunk = DataFrameConcat(
                output_types=[out.op.output_types[0]]
            ).new_chunk(inp_slice_chunks, kws=[params])
            out_chunks.append(out_chunk)

        new_op = op.copy()
        params = out.params
        params["nsplits"] = chunk_size
        params["chunks"] = out_chunks
        df_or_series = new_op.new_tileable(op.inputs, kws=[params])

        if op.reassign_worker:
            for c in df_or_series.chunks:
                c.op.reassign_worker = True

        return [df_or_series]


def _get_chunk_size(
    a: TileableType, chunk_size: chunk_size_type
) -> Tuple[Tuple[int], ...]:
    if isinstance(a, DATAFRAME_TYPE):
        itemsize = max(getattr(dt, "itemsize", 8) for dt in a.dtypes)
    else:
        itemsize = a.dtype.itemsize
    return get_nsplits(a, chunk_size, itemsize)


def rechunk(a: TileableType, chunk_size: chunk_size_type, reassign_worker=False):
    if not any(pd.isna(s) for s in a.shape) and not a.is_coarse():
        if not has_unknown_shape(a):
            # do client check only when no unknown shape,
            # real nsplits will be recalculated inside `tile`
            chunk_size = _get_chunk_size(a, chunk_size)
            if chunk_size == a.nsplits:
                return a

    op = DataFrameRechunk(
        chunk_size=chunk_size,
        reassign_worker=reassign_worker,
    )
    return op(a)
