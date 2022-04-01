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

from typing import Union

import pandas as pd

from ...core import OutputType
from ...core.context import Context
from ...lib.bloom_filter2 import BloomFilter
from ...serialization.serializables import (
    AnyField,
    Int64Field,
    Float64Field,
    StringField,
)
from ..operands import DataFrameOperandMixin, DataFrameOperand


class DataFrameBloomFilter(DataFrameOperand, DataFrameOperandMixin):
    on = AnyField("on")
    # for build
    max_elements = Int64Field("max_elements")
    error_rate = Float64Field("error_rate")

    execution_stage = StringField("execution_stage")

    def __init__(self, execution_stage=None, **kwargs):
        if execution_stage in ["build", "union"]:
            output_types = [OutputType.object]
        else:
            output_types = [OutputType.dataframe]
        kwargs["_output_types"] = output_types
        super().__init__(execution_stage=execution_stage, **kwargs)

    @classmethod
    def _get_value(cls, value):
        if isinstance(value, pd.Series):
            return value.tolist()
        else:
            return value

    @classmethod
    def _filter_on_index(cls, on, data):
        if on is None:
            return True
        elif isinstance(on, str):
            return on not in data.columns
        elif isinstance(on, list):
            return any(c not in data.columns for c in on)
        else:
            return False

    @classmethod
    def execute(cls, ctx: Union[dict, Context], op: "DataFrameBloomFilter"):
        if op.execution_stage == "build":
            on = op.on
            bloom_filter = BloomFilter(
                max_elements=op.max_elements, error_rate=op.error_rate
            )
            in_data = ctx[op.inputs[0].key]
            if cls._filter_on_index(on, in_data):
                if isinstance(in_data.index, pd.MultiIndex):
                    index = in_data.index.get_level_values(on)
                else:
                    index = in_data.index
                for value in index:
                    bloom_filter.add(cls._get_value(value))
            else:
                value_iter = (
                    in_data[on].iteritems()
                    if isinstance(on, str)
                    else in_data[on].iterrows()
                )
                for _, value in value_iter:
                    bloom_filter.add(cls._get_value(value))
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
                if isinstance(on, str):
                    ctx[op.outputs[0].key] = in_data[
                        in_data[on].map(lambda row: cls._get_value(row) in bloom_filter)
                    ]
                else:
                    ctx[op.outputs[0].key] = in_data[
                        in_data[on].apply(
                            lambda row: cls._get_value(row) in bloom_filter, axis=1
                        )
                    ]
        else:  # pragma: no branch
            raise ValueError(f"Unknown execution stage: {op.execution_stage}")
