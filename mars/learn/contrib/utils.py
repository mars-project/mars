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

import numpy as np


def pick_workers(workers, size):
    result = np.empty(size, dtype=object)
    rest = size
    while rest > 0:
        start = size - rest
        to_pick_size = min(size - start, len(workers))
        result[start: start + to_pick_size] = \
            np.random.permutation(workers)[:to_pick_size]
        rest = rest - to_pick_size
    return result
