# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

import mars.tensor as mt
import mars.dataframe as md
from mars.core.graph.builder.utils import build_graph


class ChunkGraphBuilderSuite:
    """
    Benchmark that times performance of chunk graph builder
    """

    def setup(self):
        self.df = md.DataFrame(
            mt.random.rand(1000, 10, chunk_size=(1, 10)), columns=list("abcdefghij")
        )

    def time_filter(self):
        df = self.df[self.df["a"] < 0.8]
        build_graph([df], tile=True)
