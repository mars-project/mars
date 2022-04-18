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

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import scipy.sparse as sps
except ImportError:  # pragma: no cover
    sps = None

from .core import Serializer, buffered, serialize, deserialize


class CsrMatrixSerializer(Serializer):
    @buffered
    def serial(self, obj: Any, context: Dict):
        data_header, data_buffers = serialize(obj.data)
        idx_header, idx_buffers = serialize(obj.indices)
        indptr_header, indptr_buffers = serialize(obj.indptr)
        header = (
            data_header,  # data_header
            len(data_buffers),  # data_buf_num
            idx_header,  # idx_header
            len(idx_buffers),  # idx_buf_num
            indptr_header,  # indptr_header
            obj.shape,  # shape
        )
        return header, data_buffers + idx_buffers + indptr_buffers, True

    def deserial(self, serialized: Tuple, context: Dict, subs: List):
        (
            data_header,
            data_buf_num,
            idx_header,
            idx_buf_num,
            indptr_header,
            shape,
        ) = serialized
        data_buffers = subs[:data_buf_num]
        idx_buffers = subs[data_buf_num : data_buf_num + idx_buf_num]
        indptr_buffers = subs[data_buf_num + idx_buf_num :]

        data = deserialize(data_header, data_buffers)
        indices = deserialize(idx_header, idx_buffers)
        indptr = deserialize(indptr_header, indptr_buffers)
        shape = tuple(shape)

        empty_arr = np.zeros(0, dtype=data.dtype)

        target_csr = sps.coo_matrix(
            (empty_arr, (empty_arr,) * 2), dtype=data.dtype, shape=shape
        ).tocsr()
        target_csr.data, target_csr.indices, target_csr.indptr = data, indices, indptr
        return target_csr


if sps:  # pragma: no branch
    CsrMatrixSerializer.register(sps.csr_matrix)
