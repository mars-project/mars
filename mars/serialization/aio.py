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

import asyncio
import struct
from io import BytesIO
from typing import Any

import cloudpickle
import numpy as np

from ..utils import lazy_import
from .core import serialize_with_spawn, deserialize

cupy = lazy_import("cupy", globals=globals())
cudf = lazy_import("cudf", globals=globals())

DEFAULT_SERIALIZATION_VERSION = 1
DEFAULT_SPAWN_THRESHOLD = 100
BUFFER_SIZES_NAME = "buf_sizes"


class AioSerializer:
    def __init__(self, obj: Any, compress=0):
        self._obj = obj
        self._compress = compress

    async def _get_buffers(self):
        headers, buffers = await serialize_with_spawn(
            self._obj, spawn_threshold=DEFAULT_SPAWN_THRESHOLD
        )

        def _is_cuda_buffer(buf):  # pragma: no cover
            if cupy is not None and cudf is not None:
                from cudf.core.buffer import Buffer as CPBuffer
                from cupy import ndarray as cp_ndarray
            else:
                CPBuffer = cp_ndarray = None

            if CPBuffer is not None and isinstance(buf, CPBuffer):
                return True
            elif cp_ndarray is not None and isinstance(buf, cp_ndarray):
                return True
            else:
                return False

        is_cuda_buffers = [_is_cuda_buffer(buf) for buf in buffers]
        headers[0]["is_cuda_buffers"] = np.array(is_cuda_buffers)

        # add buffer lengths into headers
        headers[0][BUFFER_SIZES_NAME] = [
            getattr(buf, "nbytes", len(buf)) for buf in buffers
        ]
        header = cloudpickle.dumps(headers)

        # gen header buffer
        header_bio = BytesIO()
        # write version first
        header_bio.write(struct.pack("B", DEFAULT_SERIALIZATION_VERSION))
        # write header length
        header_bio.write(struct.pack("<Q", len(header)))
        # write compression
        header_bio.write(struct.pack("<H", self._compress))
        # write header
        header_bio.write(header)

        out_buffers = list()
        out_buffers.append(header_bio.getbuffer())
        out_buffers.extend(buffers)

        return out_buffers

    async def run(self):
        return await self._get_buffers()


MALFORMED_MSG = """\
Received malformed data, please check Mars version on both side,
if error occurs when using `mars.new_session('http://web_ip:web_port'),
please check if web port is right."""


class AioDeserializer:
    def __init__(self, file):
        self._file = file

    def _readexactly(self, n: int):
        # asyncio StreamReader may not guarantee to read n bytes
        # for it we need to call `readexactly` instead
        read = (
            self._file.readexactly
            if hasattr(self._file, "readexactly")
            else self._file.read
        )
        return read(n)

    async def _get_obj_header_bytes(self):
        try:
            header_bytes = bytes(await self._file.read(11))
        except ConnectionResetError:
            raise EOFError("Server may be closed")
        if len(header_bytes) == 0:
            raise EOFError("Received empty bytes")
        version = struct.unpack("B", header_bytes[:1])[0]
        # now we only have default version
        assert version == DEFAULT_SERIALIZATION_VERSION, MALFORMED_MSG
        # header length
        header_length = struct.unpack("<Q", header_bytes[1:9])[0]
        # compress
        _ = struct.unpack("<H", header_bytes[9:])[0]
        return await self._readexactly(header_length)

    async def _get_obj(self):
        header = cloudpickle.loads(await self._get_obj_header_bytes())
        # get buffer size
        buffer_sizes = header[0].pop(BUFFER_SIZES_NAME)
        # get buffers
        buffers = [await self._readexactly(size) for size in buffer_sizes]
        # get num of objs
        num_objs = header[0].get("_N", 0)

        if num_objs <= DEFAULT_SPAWN_THRESHOLD:
            return deserialize(header, buffers)
        else:
            return await asyncio.to_thread(deserialize, header, buffers)

    async def run(self):
        return await self._get_obj()

    async def get_size(self):
        # extract header
        header_bytes = await self._get_obj_header_bytes()
        header = cloudpickle.loads(header_bytes)
        # get buffer size
        buffer_sizes = header[0].pop(BUFFER_SIZES_NAME)
        return 11 + len(header_bytes) + sum(buffer_sizes)

    async def get_header(self):
        return cloudpickle.loads(await self._get_obj_header_bytes())
