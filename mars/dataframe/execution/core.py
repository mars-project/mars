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

from ..expressions.core import DataFrameShuffleProxy
from .datasource import register_data_source_handler
from .arithmetic import register_arithmetic_handler
from .merge import register_merge_handler


def register_dataframe_execution_handler():
    from ...executor import register
    register(DataFrameShuffleProxy, lambda *_: None)

    register_data_source_handler()
    register_arithmetic_handler()
    register_merge_handler()
