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

import mars
import mars.tensor as mt
from mars.learn.model_selection import train_test_split


class ModelSelectionSuite:
    """
    Benchmark learn model selection.
    """

    def setup(self):
        self._session = mars.new_session()

    def teardown(self):
        self._session.stop_server()

    def time_train_test_split(self):
        t = mt.random.rand(10_000, 10, chunk_size=200)
        train_test_split(t, test_size=0.3, session=self._session)
