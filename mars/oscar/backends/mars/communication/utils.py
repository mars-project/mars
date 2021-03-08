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

import ctypes
from asyncio import StreamReader, StreamWriter
from typing import Dict, List

import numpy as np
try:
    from cudf.core import Buffer
    from cupy import ndarray
    from cupy.cuda.memory import UnownedMemory, MemoryPointer
except ImportError:
    Buffer = None
    ndarray = None
    UnownedMemory, MemoryPointer = None, None


from .....serialization.aio import BUFFER_SIZES_NAME

CUDA_CHUNK_SIZE = 16 * 1024 ** 2


def write_buffers(writer: StreamWriter,
                  buffers: List):
    def _write_cuda_buffer(ptr):
        # copy cuda buffer to host
        chunk_size = CUDA_CHUNK_SIZE
        offset = 0
        nbytes = buffer.nbytes
        while offset < nbytes:
            size = chunk_size if (offset + chunk_size) < nbytes else nbytes - offset
            chunk_buffer = Buffer(ptr + offset, size=size)
            # copy chunk to host memoryview
            writer.write(chunk_buffer.host_serialize()[1][0])
            offset += size

    for buffer in buffers:
        if Buffer is not None and isinstance(buffer, Buffer):  # pragma: no cover
            _write_cuda_buffer(buffer.ptr)
        elif ndarray is not None and isinstance(buffer, ndarray):  # pragma: no cover
            _write_cuda_buffer(buffer.data.ptr)
        else:
            writer.write(buffer)


async def read_buffers(header: Dict,
                       reader: StreamReader):
    serializer = header.get('serializer')
    chunk_size = CUDA_CHUNK_SIZE
    if serializer == 'cudf' or serializer == 'cupy':  # pragma: no cover
        # construct a empty cuda buffer and copy from host
        lengths = header.get('lengths')
        buffers = []
        for length in lengths:
            cuda_buffer = Buffer.empty(length)
            cupy_memory = UnownedMemory(cuda_buffer.ptr, length, cuda_buffer)
            offset = 0
            while offset < length:
                read_size = chunk_size if (offset + chunk_size) < length else length - offset
                content = await reader.read(read_size)
                source_mem = np.frombuffer(content, dtype='uint8').ctypes.data_as(ctypes.c_void_p)
                cupy_pointer = MemoryPointer(cupy_memory, offset)
                cupy_pointer.copy_from(source_mem, len(content))
                offset += read_size
            buffers.append(cuda_buffer)
        return buffers
    else:
        buffer_sizes = header.pop(BUFFER_SIZES_NAME)
        buffers = [await reader.read(size) for size in buffer_sizes]
        return buffers
