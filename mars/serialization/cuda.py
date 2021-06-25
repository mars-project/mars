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

from typing import Any, List, Dict

from ..utils import lazy_import
from .core import Serializer, buffered

cupy = lazy_import('cupy', globals=globals())
cudf = lazy_import('cudf', globals=globals())


class CupySerializer(Serializer):
    serializer_name = 'cupy'

    @buffered
    def serialize(self, obj: Any, context: Dict):
        if not (obj.flags["C_CONTIGUOUS"] or obj.flags["F_CONTIGUOUS"]):
            obj = cupy.array(obj, copy=True)

        header = obj.__cuda_array_interface__.copy()
        header["strides"] = tuple(obj.strides)
        header["lengths"] = [obj.nbytes]
        buffer = cupy.ndarray(
            shape=(obj.nbytes,), dtype=cupy.dtype("u1"), memptr=obj.data, strides=(1,)
        )
        return header, [buffer]

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        return cupy.ndarray(
            shape=header["shape"],
            dtype=header["typestr"],
            memptr=cupy.asarray(buffers[0]).data,
            strides=header["strides"],
        )


class CudfSerializer(Serializer):
    serializer_name = 'cudf'

    def serialize(self, obj: Any, context: Dict):
        return obj.device_serialize()

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        from cudf.core.abc import Serializable
        return Serializable.device_deserialize(header, buffers)


if cupy is not None:
    CupySerializer.register('cupy.ndarray')
if cudf is not None:
    CudfSerializer.register('cudf.DataFrame')
    CudfSerializer.register('cudf.Series')
    CudfSerializer.register('cudf.Index')
