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

import numpy as np


def calc_columns_index(column_name, df):
    """
    Calculate the chunk index on the axis 1 according to the selected column.
    :param column_name: selected column name
    :param df: input tiled DataFrame
    :return: chunk index on the columns axis
    """
    column_nsplits = df.nsplits[1]
    column_loc = df.columns.to_pandas().get_loc(column_name)
    return np.searchsorted(np.cumsum(column_nsplits), column_loc + 1)
