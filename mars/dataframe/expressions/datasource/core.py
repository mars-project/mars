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


from ....operands.datasource import DataSource
from ....config import options
from ..core import DataFrameOperandMixin


class DataFrameNoInput(DataSource, DataFrameOperandMixin):
    """
    Represents data from pandas DataFrame
    """

    __slots__ = ()

    @classmethod
    def tile(cls, op):
        df = op.outputs[0]

        chunk_size = df.params.raw_chunk_size or options.tensor.chunk_size
