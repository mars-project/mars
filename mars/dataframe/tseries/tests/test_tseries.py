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
import pytest

from .... import dataframe as md


def test_to_datetime():
    wrong_args = [pd.DataFrame({"a": [1, 2]}), {"a": [1, 2]}]

    for arg in wrong_args:
        with pytest.raises(ValueError) as cm:
            md.to_datetime(arg)
        assert "[year, month, day]" in str(cm.value)

    with pytest.raises(TypeError):
        md.to_datetime([[1, 2], [3, 4]])
