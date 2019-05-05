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

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pass

from ..utils import build_empty_df, parse_index
from ...core import IndexValue
from ....utils import tokenize


def infer_dtypes(left_dtypes, right_dtypes, operator):
    left = build_empty_df(left_dtypes)
    right = build_empty_df(right_dtypes)
    return operator(left, right).dtypes


def infer_index_value(left_index_value, right_index_value, operator):
    if isinstance(left_index_value.value, IndexValue.RangeIndex) and \
            isinstance(right_index_value.value, IndexValue.RangeIndex):
        if left_index_value.value.slice == right_index_value.value.slice:
            return left_index_value
        key = tokenize(left_index_value.key, right_index_value.key,
                       operator.__name__)
        return parse_index(pd.Int64Index([]), key=key)

    # when left index and right index is identical, and both of them are elements unique,
    # we can infer that the out index should be identical also
    if left_index_value.is_unique and right_index_value.is_unique and \
            left_index_value.key == right_index_value.key:
        return left_index_value

    left_index = left_index_value.to_pandas()
    right_index = right_index_value.to_pandas()
    out_index = operator(left_index, right_index)
    key = tokenize(left_index_value.key, right_index_value.key, operator.__name__)
    return parse_index(out_index, key=key)


def filter_dtypes(dtypes, column_min_max):
    l_filter = operator.ge if column_min_max[1] else operator.gt
    l = l_filter(dtypes.index, column_min_max[0])
    r_filter = operator.le if column_min_max[3] else operator.lt
    r = r_filter(dtypes.index, column_min_max[2])
    f = l & r
    return dtypes[f]
