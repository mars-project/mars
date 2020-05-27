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

from ...tensor import tensor as astensor
from ...tensor.core import TENSOR_TYPE, CHUNK_TYPE as TENSOR_CHUNK_TYPE
from ...dataframe import DataFrame, Series
from ...dataframe.core import DATAFRAME_TYPE, SERIES_TYPE, \
    DATAFRAME_CHUNK_TYPE, SERIES_CHUNK_TYPE
from ..operands import OutputType


def convert_to_tensor_or_dataframe(item):
    if isinstance(item, (DATAFRAME_TYPE, pd.DataFrame)):
        item = DataFrame(item)
    elif isinstance(item, (SERIES_TYPE, pd.Series)):
        item = Series(item)
    else:
        item = astensor(item)
    return item


def get_output_types(*objs, unknown_as=None):
    output_types = []
    for obj in objs:
        if obj is None:
            continue
        if isinstance(obj, (TENSOR_TYPE, TENSOR_CHUNK_TYPE)):
            output_types.append(OutputType.tensor)
        elif isinstance(obj, (DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE)):
            output_types.append(OutputType.dataframe)
        elif isinstance(obj, (SERIES_TYPE, SERIES_CHUNK_TYPE)):
            output_types.append(OutputType.series)
        elif unknown_as is not None:
            output_types.append(unknown_as)
        else:  # pragma: no cover
            raise TypeError('Output can only be tensor, dataframe or series')
    return output_types


def concat_chunks(chunks):
    tileable = chunks[0].op.create_tileable_from_chunks(chunks)
    return tileable.op.concat_tileable_chunks(tileable).chunks[0]
