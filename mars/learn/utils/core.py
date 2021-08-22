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

from ...tensor import tensor as astensor
from ...dataframe import DataFrame, Series
from ...dataframe.core import DATAFRAME_TYPE, SERIES_TYPE


def convert_to_tensor_or_dataframe(item):
    if isinstance(item, (DATAFRAME_TYPE, pd.DataFrame)):
        item = DataFrame(item)
    elif isinstance(item, (SERIES_TYPE, pd.Series)):
        item = Series(item)
    else:
        item = astensor(item)
    return item


def concat_chunks(chunks):
    tileable = chunks[0].op.create_tileable_from_chunks(chunks)
    return tileable.op.concat_tileable_chunks(tileable).chunks[0]
