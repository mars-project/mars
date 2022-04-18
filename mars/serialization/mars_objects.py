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

from typing import Any, Dict, List

try:
    import scipy.sparse as sps
except ImportError:  # pragma: no cover
    sps = None

from ..lib.sparse import SparseNDArray
from .core import Serializer, serialize, deserialize, buffered


class SparseNDArraySerializer(Serializer):
    @buffered
    def serial(self, obj: Any, context: Dict):
        raw_header, raw_buffers = serialize(obj.raw, context)
        return (raw_header, obj.shape), raw_buffers, True

    def deserial(self, serialized: Dict, context: Dict, subs: List):
        raw_header, obj_shape = serialized
        raw_csr = deserialize(raw_header, subs)
        return SparseNDArray(raw_csr, shape=tuple(obj_shape))


if sps:  # pragma: no branch
    SparseNDArraySerializer.register(SparseNDArray)
