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

from typing import Dict, List

import numpy as np

from .core import Serializer, buffered, pickle_buffers, unpickle_buffers


class NDArraySerializer(Serializer):
    serializer_name = 'np_ndarray'

    @buffered
    def serialize(self, obj: np.ndarray, context: Dict):
        header = {}
        if obj.dtype.hasobject:
            header['pickle'] = True
            buffers = pickle_buffers(obj)
            return header, buffers

        order = 'C'
        if obj.flags.f_contiguous:
            order = 'F'
        elif not obj.flags.c_contiguous:
            obj = np.ascontiguousarray(obj)
        try:
            desc = np.lib.format.dtype_to_descr(obj.dtype)
            dtype_new_order = None
        except ValueError:
            # for structured dtype, array[[field2, field1]] will create a view,
            # and dtype_to_desc will fail due to the order
            fields = obj.dtype.fields
            new_fields = sorted(fields, key=lambda k: fields[k][1])
            desc = np.lib.format.dtype_to_descr(obj.dtype[new_fields])
            dtype_new_order = list(fields)
        header.update(dict(
            pickle=False,
            descr=desc,
            dtype_new_order=dtype_new_order,
            shape=list(obj.shape),
            strides=list(obj.strides),
            order=order
        ))
        return header, [memoryview(obj.ravel(order=order).view('uint8').data)]

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        if header['pickle']:
            return unpickle_buffers(buffers)

        dtype = np.lib.format.descr_to_dtype(header['descr'])
        dtype_new_order = header['dtype_new_order']
        if dtype_new_order:
            dtype = dtype[dtype_new_order]
        return np.ndarray(
            shape=tuple(header['shape']), dtype=dtype,
            buffer=buffers[0], strides=tuple(header['strides']),
            order=header['order']
        )


NDArraySerializer.register(np.ndarray)
