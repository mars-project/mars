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

import numpy as np
try:
    import scipy.sparse as sps
except ImportError:  # pragma: no cover
    sps = None

from .core import Serializer, buffered, serialize, deserialize


class CsrMatrixSerializer(Serializer):
    serializer_name = 'sps.csr_matrix'

    @buffered
    def serialize(self, obj: Any, context: Dict):
        data_header, data_buffers = serialize(obj.data)
        idx_header, idx_buffers = serialize(obj.indices)
        indptr_header, indptr_buffers = serialize(obj.indptr)
        header = {
            'data_header': data_header, 'data_buf_num': len(data_buffers),
            'idx_header': idx_header, 'idx_buf_num': len(idx_buffers),
            'indptr_header': indptr_header,
            'shape': list(obj.shape),
        }
        return header, data_buffers + idx_buffers + indptr_buffers

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        data_buf_num = header['data_buf_num']
        idx_buf_num = header['idx_buf_num']
        data_buffers = buffers[:data_buf_num]
        idx_buffers = buffers[data_buf_num:data_buf_num + idx_buf_num]
        indptr_buffers = buffers[data_buf_num + idx_buf_num:]

        data = deserialize(header['data_header'], data_buffers)
        indices = deserialize(header['idx_header'], idx_buffers)
        indptr = deserialize(header['indptr_header'], indptr_buffers)
        shape = tuple(header['shape'])

        empty_arr = np.zeros(0, dtype=data.dtype)

        target_csr = sps.coo_matrix((empty_arr, (empty_arr,) * 2), dtype=data.dtype,
                                    shape=shape).tocsr()
        target_csr.data, target_csr.indices, target_csr.indptr = data, indices, indptr
        return target_csr


if sps:  # pragma: no branch
    CsrMatrixSerializer.register(sps.csr_matrix)
