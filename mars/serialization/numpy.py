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

from .core import Serializer, pickle_buffers, unpickle_buffers


class NDArraySerializer(Serializer):
    serializer_name = 'np_ndarray'

    def serialize(self, obj: np.ndarray):
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
        header.update(dict(
            pickle=False,
            descr=np.lib.format.dtype_to_descr(obj.dtype),
            shape=list(obj.shape),
            strides=list(obj.strides),
            order=order
        ))
        return header, [memoryview(obj.ravel(order=order).view('uint8').data)]

    def deserialize(self, header, buffers):
        if header['pickle']:
            return unpickle_buffers(buffers)

        dtype = np.lib.format.descr_to_dtype(header['descr'])
        return np.ndarray(
            shape=tuple(header['shape']), dtype=dtype,
            buffer=buffers[0], strides=tuple(header['strides']),
            order=header['order']
        )


NDArraySerializer.register(np.ndarray)
