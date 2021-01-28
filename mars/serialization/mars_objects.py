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

from typing import Dict, List

try:
    import scipy.sparse as sps
except ImportError:  # pragma: no cover
    sps = None

from ..lib.sparse import SparseNDArray
from .core import Serializer, serialize, deserialize


class SparseNDArraySerializer(Serializer):
    serializer_name = 'mars.SparseNDArray'

    def serialize(self, obj):
        raw_header, raw_buffers = serialize(obj.raw)
        header = {
            'raw_header': raw_header, 'shape': list(obj.shape),
        }
        return header, raw_buffers

    def deserialize(self, header: Dict, buffers: List):
        raw_csr = deserialize(header['raw_header'], buffers)
        return SparseNDArray(raw_csr, shape=tuple(header['shape']))


if sps:  # pragma: no branch
    SparseNDArraySerializer.register(SparseNDArray)
