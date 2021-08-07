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

import ctypes
from asyncio import StreamReader, StreamWriter
from typing import Dict, List

import numpy as np

from ....serialization.aio import BUFFER_SIZES_NAME
from ....utils import lazy_import

cupy = lazy_import('cupy', globals=globals())
cudf = lazy_import('cudf', globals=globals())

CUDA_CHUNK_SIZE = 16 * 1024 ** 2


def write_buffers(writer: StreamWriter,
                  buffers: List):
    if cupy is not None and cudf is not None:
        from cudf.core import Buffer as CPBuffer
        from cupy import ndarray as cp_ndarray
    else:
        CPBuffer = cp_ndarray = None

    def _write_cuda_buffer(ptr):  # pragma: no cover
        # copy cuda buffer to host
        chunk_size = CUDA_CHUNK_SIZE
        offset = 0
        nbytes = buffer.nbytes
        while offset < nbytes:
            size = chunk_size if (offset + chunk_size) < nbytes else nbytes - offset
            chunk_buffer = CPBuffer(ptr + offset, size=size)
            # copy chunk to host memoryview
            writer.write(chunk_buffer.host_serialize()[1][0])
            offset += size

    for buffer in buffers:
        if CPBuffer is not None and isinstance(buffer, CPBuffer):  # pragma: no cover
            _write_cuda_buffer(buffer.ptr)
        elif cp_ndarray is not None and isinstance(buffer, cp_ndarray):  # pragma: no cover
            _write_cuda_buffer(buffer.data.ptr)
        else:
            writer.write(buffer)


async def read_buffers(header: Dict,
                       reader: StreamReader):
    if cupy is not None and cudf is not None:
        from cudf.core import Buffer as CPBuffer
        from cupy.cuda.memory import UnownedMemory as CPUnownedMemory, \
            MemoryPointer as CPMemoryPointer
    else:
        CPBuffer = CPUnownedMemory = CPMemoryPointer = None

    # construct a empty cuda buffer and copy from host
    is_cuda_buffers = header.get('is_cuda_buffers')
    buffer_sizes = header.pop(BUFFER_SIZES_NAME)

    buffers = []
    for is_cuda_buffer, buf_size in zip(is_cuda_buffers, buffer_sizes):
        if is_cuda_buffer:  # pragma: no cover
            if buf_size == 0:
                content = await reader.readexactly(buf_size)
                buffers.append(content)
            else:
                cuda_buffer = CPBuffer.empty(buf_size)
                cupy_memory = CPUnownedMemory(cuda_buffer.ptr, buf_size, cuda_buffer)
                offset = 0
                chunk_size = CUDA_CHUNK_SIZE
                while offset < buf_size:
                    read_size = chunk_size if (offset + chunk_size) < buf_size else buf_size - offset
                    content = await reader.readexactly(read_size)
                    source_mem = np.frombuffer(content, dtype='uint8').ctypes.data_as(ctypes.c_void_p)
                    cupy_pointer = CPMemoryPointer(cupy_memory, offset)
                    cupy_pointer.copy_from(source_mem, len(content))
                    offset += read_size
                buffers.append(cuda_buffer)
        else:
            buffers.append(await reader.readexactly(buf_size))
    return buffers
