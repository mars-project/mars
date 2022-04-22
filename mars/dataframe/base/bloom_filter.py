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

from typing import Union, List, Any

import numpy as np
import pandas as pd

from ...config import options
from ...core import OutputType
from ...core.context import Context
from ...lib.bloom_filter import BloomFilter
from ...serialization.serializables import (
    AnyField,
    Int64Field,
    Float64Field,
    StringField,
)
from ... import opcodes as OperandDef
from ...typing import TileableType
from ..operands import DataFrameOperandMixin, DataFrameOperand
from ..utils import parse_index


class DataFrameBloomFilter(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_BLOOM_FILTER

    left_on = AnyField("left_on")
    right_on = AnyField("right_on")
    on = AnyField("on")
    # for build
    max_elements = Int64Field("max_elements")
    error_rate = Float64Field("error_rate")

    execution_stage = StringField("execution_stage", default=None)

    def __init__(self, execution_stage=None, **kwargs):
        if execution_stage in ["build", "union"]:
            output_types = [OutputType.object]
        else:
            output_types = [OutputType.dataframe]
        kwargs["_output_types"] = output_types
        super().__init__(execution_stage=execution_stage, **kwargs)

    def __call__(self, df1: TileableType, df2: TileableType):
        return self.new_tileable([df1, df2], **df1.params)

    @classmethod
    def tile(cls, op: "DataFrameBloomFilter"):
        df1, df2 = op.inputs
        # use df2's chunks to build bloom filter
        chunks = []
        for c in df2.chunks:
            build_op = DataFrameBloomFilter(
                on=op.right_on,
                max_elements=op.max_elements,
                error_rate=op.error_rate,
                execution_stage="build",
            )
            chunks.append(build_op.new_chunk(inputs=[c]))

        # union all chunk filters
        combine_size = options.combine_size
        while len(chunks) > combine_size:
            new_chunks = []
            for i in range(0, len(chunks), combine_size):
                chks = chunks[i : i + combine_size]
                if len(chks) == 1:
                    chk = chks[0]
                else:
                    union_op = DataFrameBloomFilter(execution_stage="union")
                    for j, c in enumerate(chks):
                        c._index = (j, 0)
                    chk = union_op.new_chunk(chks)
                new_chunks.append(chk)
            chunks = new_chunks
        if len(chunks) > 1:
            union_op = DataFrameBloomFilter(execution_stage="union")
            filter_chunk = union_op.new_chunk(chunks)
        else:
            filter_chunk = chunks[0]

        filter_chunk.is_broadcaster = True
        # filter df1
        out_chunks = []
        for chunk in df1.chunks:
            filter_op = DataFrameBloomFilter(on=op.left_on, execution_stage="filter")
            params = chunk.params.copy()
            params["shape"] = (np.nan, chunk.shape[1])
            params["index_value"] = parse_index(pd.RangeIndex(-1))
            out_chunks.append(filter_op.new_chunk([chunk, filter_chunk], **params))

        new_op = op.copy()
        df1_params = df1.params.copy()
        df1_params["chunks"] = out_chunks
        df1_params["nsplits"] = ((np.nan,) * len(out_chunks), df1.nsplits[1])
        return new_op.new_dataframes(op.inputs, **df1_params)

    @classmethod
    def _get_value(cls, value: Any):
        # value could be an element or a series, as BloomFilter
        # doesn't accept series, convert to list here
        if isinstance(value, pd.Series):
            return value.tolist()
        else:
            return value

    @classmethod
    def _filter_on_index(cls, on: Union[str, List, None], data: pd.DataFrame):
        if on is None:
            return True
        elif isinstance(on, str):
            return on not in data.columns
        else:
            assert isinstance(on, list)
            return any(c not in data.columns for c in on)

    @classmethod
    def _build_index_filter(cls, in_data: pd.DataFrame, op: "DataFrameBloomFilter"):
        if isinstance(in_data.index, pd.MultiIndex):
            index = in_data.index.get_level_values(op.on)
        else:
            index = in_data.index
        bloom_filter = BloomFilter(
            max_elements=op.max_elements, error_rate=op.error_rate
        )
        index.map(lambda v: bloom_filter.add(cls._get_value(v)))
        return bloom_filter

    @classmethod
    def _build_series_filter(cls, in_data: pd.Series, op: "DataFrameBloomFilter"):
        try:
            bloom_filter = BloomFilter(
                max_elements=op.max_elements, error_rate=op.error_rate
            )
            in_data[op.on].map(lambda v: bloom_filter.add(cls._get_value(v)))
        except TypeError:
            # has unhashable data, convert to str
            in_data = in_data.astype(str)
            bloom_filter = BloomFilter(
                max_elements=op.max_elements, error_rate=op.error_rate
            )
            in_data[op.on].map(lambda v: bloom_filter.add(cls._get_value(v)))
        return bloom_filter

    @classmethod
    def _build_dataframe_filter(cls, in_data: pd.DataFrame, op: "DataFrameBloomFilter"):
        try:
            bloom_filter = BloomFilter(
                max_elements=op.max_elements, error_rate=op.error_rate
            )
            in_data[op.on].apply(lambda v: bloom_filter.add(cls._get_value(v)), axis=1)
        except TypeError:
            # has unhashable data, convert to str
            in_data = in_data.astype(cls._convert_to_hashable_dtypes(in_data.dtypes))
            bloom_filter = BloomFilter(
                max_elements=op.max_elements, error_rate=op.error_rate
            )
            in_data[op.on].apply(lambda v: bloom_filter.add(cls._get_value(v)), axis=1)
        return bloom_filter

    @classmethod
    def _convert_to_hashable_dtypes(cls, dtypes: pd.Series):
        dtypes = dict(
            (name, dtype) if np.issubdtype(dtype, int) else (name, str)
            for name, dtype in dtypes.iteritems()
        )
        return dtypes

    @classmethod
    def execute(cls, ctx: Union[dict, Context], op: "DataFrameBloomFilter"):
        if op.execution_stage == "build":
            on = op.on
            in_data = ctx[op.inputs[0].key]
            if cls._filter_on_index(on, in_data):
                bloom_filter = cls._build_index_filter(in_data, op)
            elif isinstance(on, str):
                bloom_filter = cls._build_series_filter(in_data, op)
            else:
                bloom_filter = cls._build_dataframe_filter(in_data, op)
            ctx[op.outputs[0].key] = bloom_filter
        elif op.execution_stage == "union":
            # union bloom filters
            filters = [ctx[inp.key] for inp in op.inputs]
            out = filters[0]
            for f in filters[1:]:
                out.union(f)
            ctx[op.outputs[0].key] = out
        elif op.execution_stage == "filter":
            on = op.on
            in_data = ctx[op.inputs[0].key]
            bloom_filter = ctx[op.inputs[1].key]
            if cls._filter_on_index(on, in_data):
                if isinstance(in_data.index, pd.MultiIndex):
                    idx = in_data.index.names.index(on)
                    ctx[op.outputs[0].key] = in_data[
                        in_data.index.map(lambda x: x[idx] in bloom_filter)
                    ]
                else:
                    ctx[op.outputs[0].key] = in_data[
                        in_data.index.map(lambda x: x in bloom_filter)
                    ]
            else:
                row_func = lambda row: cls._get_value(row) in bloom_filter
                if isinstance(on, str):
                    # series
                    try:
                        filtered = in_data[in_data[on].map(row_func)]
                    except TypeError:
                        converted_data = in_data.astype(str)
                        filtered = in_data[converted_data[on].map(row_func)]
                    ctx[op.outputs[0].key] = filtered
                else:
                    # dataframe
                    try:
                        filtered = in_data[in_data[on].apply(row_func, axis=1)]
                    except TypeError:
                        converted_data = in_data.astype(
                            cls._convert_to_hashable_dtypes(in_data.dtypes)
                        )
                        filtered = in_data[converted_data[on].apply(row_func, axis=1)]
                    ctx[op.outputs[0].key] = filtered

        else:  # pragma: no cover
            raise ValueError(f"Unknown execution stage: {op.execution_stage}")


def filter_by_bloom_filter(
    df1: TileableType,
    df2: TileableType,
    left_on: Union[str, List],
    right_on: Union[str, List],
    max_elements: int = 10000,
    error_rate: float = 0.1,
):
    """
    Use bloom filter to filter DataFrame.

    Parameters
    ----------
    df1: DataFrame.
        DataFrame to be filtered.
    df2: DataFrame.
        Dataframe to build filter.
    left_on: str or list.
        Column(s) selected on df1.
    right_on: str or list.
        Column(s) selected on df2.
    max_elements: int
        How many elements you expect the filter to hold.
    error_rate: float
        error_rate defines accuracy.

    Returns
    -------
    DataFrame
        Filtered df1.
    """
    op = DataFrameBloomFilter(
        left_on=left_on,
        right_on=right_on,
        max_elements=max_elements,
        error_rate=error_rate,
    )
    return op(df1, df2)
