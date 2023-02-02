# Copyright 2022-2023 XProbe Inc.
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

from typing import Any, Set

from .....dataframe.core import BaseDataFrameData


def get_cols_exclude_index(inp: BaseDataFrameData, cols: Any) -> Set[Any]:
    ret = set()
    if isinstance(cols, (list, tuple)):
        for col in cols:
            if col in inp.dtypes.index:
                # exclude index
                ret.add(col)
    else:
        if cols in inp.dtypes.index:
            # exclude index
            ret.add(cols)
    return ret
