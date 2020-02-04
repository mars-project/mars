#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import pandas as pd

from ... import opcodes as OperandDef
from ...utils import lazy_import
from ..operands import DataFrameOperandMixin, DataFrameOperand


cudf = lazy_import('cudf', globals=globals())


class ChunkStandardizeRangeIndex(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.STANDARDIZE_RANGE_INDEX

    def __init__(self, prepare_inputs=None, object_type=None, **kwargs):
        super().__init__(__prepare_inputs=prepare_inputs, _object_type=object_type, **kwargs)

    @classmethod
    def execute(cls, ctx, op):
        xdf = cudf if op.gpu else pd
        in_data = ctx[op.inputs[-1].key]
        input_keys = [c.key for c in op.inputs[:-1]]
        metas = ctx.get_chunk_metas(input_keys)
        index_start = sum([m.chunk_shape[0] for m in metas])
        ctx[op.outputs[0].key] = in_data.set_index(xdf.RangeIndex(index_start, index_start + len(in_data)))
