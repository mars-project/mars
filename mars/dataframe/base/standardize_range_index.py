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

import pandas as pd

from ... import opcodes as OperandDef
from ...utils import lazy_import
from ...serialization.serializables import Int32Field, ListField, FieldTypes
from ..operands import DataFrameOperandMixin, DataFrameOperand


cudf = lazy_import("cudf")


class ChunkStandardizeRangeIndex(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.STANDARDIZE_RANGE_INDEX

    axis = Int32Field("axis")
    prev_shapes = ListField("prev_shapes", FieldTypes.tuple)

    @classmethod
    def execute(cls, ctx, op: "ChunkStandardizeRangeIndex"):
        xdf = cudf if op.gpu else pd
        in_data = ctx[op.inputs[0].key].copy()
        index_start = sum([shape[op.axis] for shape in op.prev_shapes])
        if op.axis == 0:
            in_data.index = xdf.RangeIndex(index_start, index_start + len(in_data))
        else:
            in_data.columns = xdf.RangeIndex(
                index_start, index_start + in_data.shape[1]
            )
        ctx[op.outputs[0].key] = in_data
