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

import struct
from io import BytesIO
from typing import Any

from .core import pickle, serialize, deserialize


DEFAULT_SERIALIZATION_VERSION = 0
BUFFER_SIZES_NAME = 'buf_sizes'


class AioSerializer:
    def __init__(self,
                 obj: Any,
                 compress=0):
        self._obj = obj
        self._compress = compress

    def _get_buffers(self):
        headers, buffers = serialize(self._obj)

        # add buffer lengths into headers
        headers[BUFFER_SIZES_NAME] = [getattr(buf, 'nbytes', len(buf))
                                      for buf in buffers]
        header = pickle.dumps(headers)

        # gen header buffer
        header_bio = BytesIO()
        # write version first
        header_bio.write(struct.pack('B', DEFAULT_SERIALIZATION_VERSION))
        # write header length
        header_bio.write(struct.pack('<Q', len(header)))
        # write compression
        header_bio.write(struct.pack('<H', self._compress))
        # write header
        header_bio.write(header)

        out_buffers = list()
        out_buffers.append(header_bio.getbuffer())
        out_buffers.extend(buffers)

        return out_buffers

    async def run(self):
        return self._get_buffers()


class AioDeserializer:
    def __init__(self, file):
        self._file = file

    async def _get_obj(self):
        header_bytes = bytes(await self._file.read(11))
        if len(header_bytes) == 0:
            raise EOFError('Received empty bytes')
        version = struct.unpack('B', header_bytes[:1])[0]
        # now we only have default version
        assert version == DEFAULT_SERIALIZATION_VERSION
        # header length
        header_length = struct.unpack('<Q', header_bytes[1:9])[0]
        # compress
        _ = struct.unpack('<H', header_bytes[9:])[0]
        # extract header
        header = pickle.loads(await self._file.read(header_length))
        # get buffer size
        buffer_sizes = header.pop(BUFFER_SIZES_NAME)
        # get buffers
        buffers = [await self._file.read(size) for size in buffer_sizes]

        return deserialize(header, buffers)

    async def run(self):
        return await self._get_obj()

    async def get_size(self):
        header_bytes = bytes(await self._file.read(11))
        # header length
        header_length = struct.unpack('<Q', header_bytes[1:9])[0]
        # extract header
        header = pickle.loads(await self._file.read(header_length))
        # get buffer size
        buffer_sizes = header.pop(BUFFER_SIZES_NAME)
        return 11 + header_length + sum(buffer_sizes)
