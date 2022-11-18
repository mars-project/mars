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

from asyncio import StreamReader, StreamWriter
from typing import Dict, List, Union

import numpy as np

from ....serialization.aio import BUFFER_SIZES_NAME
from ....utils import lazy_import

cupy = lazy_import("cupy")
cudf = lazy_import("cudf")
rmm = lazy_import("rmm")

CUDA_CHUNK_SIZE = 16 * 1024**2


def _convert_to_cupy_ndarray(
    cuda_buffer: Union["cupy.ndarray", "rmm.DeviceBuffer"]
) -> "cupy.ndarray":
    if isinstance(cuda_buffer, cupy.ndarray):
        return cuda_buffer

    size = cuda_buffer.nbytes
    data = cuda_buffer.__cuda_array_interface__["data"][0]
    memory = cupy.cuda.UnownedMemory(data, size, cuda_buffer)
    ptr = cupy.cuda.MemoryPointer(memory, 0)
    return cupy.ndarray(shape=size, dtype="u1", memptr=ptr)


def write_buffers(writer: StreamWriter, buffers: List):
    def _write_cuda_buffer(cuda_buffer: Union["cupy.ndarray", "rmm.DeviceBuffer"]):
        # convert cuda buffer to cupy ndarray
        cuda_buffer = _convert_to_cupy_ndarray(cuda_buffer)

        chunk_size = CUDA_CHUNK_SIZE
        offset = 0
        nbytes = buffer.nbytes
        while offset < nbytes:
            size = chunk_size if (offset + chunk_size) < nbytes else nbytes - offset
            # slice on cupy ndarray
            chunk_buffer = cuda_buffer[offset : offset + size]
            # `get` will return numpy ndarray,
            # write its data which is a memoryview into writer
            writer.write(chunk_buffer.get().data)
            offset += size

    for buffer in buffers:
        if hasattr(buffer, "__cuda_array_interface__"):
            # GPU buffer
            _write_cuda_buffer(buffer)
        else:
            writer.write(buffer)


async def read_buffers(header: Dict, reader: StreamReader):
    is_cuda_buffers = header[0].get("is_cuda_buffers")
    buffer_sizes = header[0].pop(BUFFER_SIZES_NAME)

    buffers = []
    for is_cuda_buffer, buf_size in zip(is_cuda_buffers, buffer_sizes):
        if is_cuda_buffer:  # pragma: no cover
            if buf_size == 0:
                content = await reader.readexactly(buf_size)
                buffers.append(content)
            else:
                buffer = rmm.DeviceBuffer(size=buf_size)
                arr = _convert_to_cupy_ndarray(buffer)
                offset = 0
                chunk_size = CUDA_CHUNK_SIZE
                while offset < buf_size:
                    read_size = (
                        chunk_size
                        if (offset + chunk_size) < buf_size
                        else buf_size - offset
                    )
                    content = await reader.readexactly(read_size)
                    chunk_arr = np.frombuffer(content, dtype="u1")
                    arr[offset : offset + len(content)].set(chunk_arr)
                    offset += read_size
                buffers.append(buffer)
        else:
            buffers.append(await reader.readexactly(buf_size))
    return buffers
