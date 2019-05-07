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

import hashlib
import functools
import operator


def hash_index(index, size):
    def func(x, size):
        return int(hashlib.md5(bytes(x)).hexdigest(), 16) % size

    f = functools.partial(func, size=size)
    grouped = sorted(index.groupby(index.map(f)).items(),
                     key=operator.itemgetter(0))
    return [g[1] for g in grouped]


def hash_dtypes(dtypes, size):
    hashed_indexes = hash_index(dtypes.index, size)
    return [dtypes[index] for index in hashed_indexes]
