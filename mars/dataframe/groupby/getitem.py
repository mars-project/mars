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

import numpy as np
import pandas as pd
from pandas.core import groupby as pd_groupby

from ... import opcodes
from ...operands import OperandStage
from ...serialize import BoolField, Int32Field, AnyField
from ...utils import get_shuffle_input_keys_idxes
from ..utils import build_concatenated_rows_frame, hash_dataframe_on, \
    build_empty_df, build_empty_series, parse_index
from ..operands import DataFrameOperandMixin, DataFrameOperand, ObjectType


class GroupByIndex(DataFrameOperandMixin, DataFrameOperand):
    _op_type_ = opcodes.INDEX
    _op_module_ = 'dataframe.groupby'

    _labels = AnyField('labels')

    def __init__(self, labels=None, object_type=None, **kw):
        super().__init__(_labels=labels, _object_type=object_type, **kw)

    @property
    def labels(self):
        return self._labels

    def __call__(self, groupby):
        indexed = groupby.op.build_mock_groupby()[self.labels]
        if isinstance(indexed, pd_groupby.DataFrameGroupBy):
            self._object_type = ObjectType.dataframe_groupby
        else:
            self._object_type = ObjectType.series_groupby

        return self.new_tileable([groupby])

    @classmethod
    def tile(cls, op):
        in_groupby = op.inputs[0]
