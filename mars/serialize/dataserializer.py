# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import functools
import gzip
import struct
import sys
import zlib
from collections import namedtuple

from ..compat import BytesIO, Enum

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import scipy.sparse as sps
except ImportError:  # pragma: no cover
    sps = None
try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

from ..lib.sparse import SparseNDArray

try:
    import pyarrow
except ImportError:  # pragma: no cover
    pyarrow = None


BUFFER_SIZE = 256 * 1024


class DummyCompress(object):
    def __enter__(self):  # pragma: no cover
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # pragma: no cover
        pass

    @staticmethod
    def compress(data):  # pragma: no cover
        return data

    @staticmethod
    def decompress(data):  # pragma: no cover
        return data


try:
    import lz4.frame
    try:
        lz4.frame._compression.BUFFER_SIZE = BUFFER_SIZE
    except AttributeError:  # pragma: no cover
        pass
    lz4_open = functools.partial(lz4.frame.open, block_size=lz4.frame.BLOCKSIZE_MAX1MB)
    lz4_compress = functools.partial(lz4.frame.compress, block_size=lz4.frame.BLOCKSIZE_MAX1MB)
    lz4_compressobj = lz4.frame.LZ4FrameCompressor
    lz4_decompress = lz4.frame.decompress
    lz4_decompressobj = lz4.frame.LZ4FrameDecompressor
except ImportError:  # pragma: no cover
    lz4_open = None
    lz4_compress, lz4_compressobj = None, None
    lz4_decompress, lz4_decompressobj = None, None

if sys.version_info[0] < 3:
    # we don't support gzip in Python 2.7 as
    # there are many limitations in the gzip module
    gz_open = None
    gz_compressobj = gz_decompressobj = None
    gz_compress = gz_decompress = None
else:
    gz_open = gzip.open
    gz_compressobj = functools.partial(
        lambda level=-1: zlib.compressobj(level, zlib.DEFLATED, 16 + zlib.MAX_WBITS, zlib.DEF_MEM_LEVEL, 0)
    )
    gz_decompressobj = functools.partial(lambda: zlib.decompressobj(16 + zlib.MAX_WBITS))
    gz_compress = gzip.compress
    gz_decompress = gzip.decompress

SERIAL_VERSION = 0


class CompressType(Enum):
    NONE = 'none'
    LZ4 = 'lz4'
    GZIP = 'gzip'

    @property
    def tag(self):
        return _compress_tags[self]

    @classmethod
    def from_tag(cls, tag):
        return _tag_to_compress[tag]

    def __gt__(self, other):
        return _compress_tags[self] > _compress_tags[other]


_compress_tags = {
    CompressType.NONE: 0,
    CompressType.LZ4: 1,
    CompressType.GZIP: 2,
}
_tag_to_compress = {
    0: CompressType.NONE,
    1: CompressType.LZ4,
    2: CompressType.GZIP,
}

compressors = {
    CompressType.LZ4: lz4_compress,
    CompressType.GZIP: gz_compress,
}
compressobjs = {
    CompressType.NONE: DummyCompress,
    CompressType.LZ4: lz4_compressobj,
    CompressType.GZIP: gz_compressobj,
}
decompressors = {
    CompressType.LZ4: lz4_decompress,
    CompressType.GZIP: gz_decompress,
}
decompressobjs = {
    CompressType.NONE: DummyCompress,
    CompressType.LZ4: lz4_decompressobj,
    CompressType.GZIP: gz_decompressobj,
}
compress_openers = {
    CompressType.LZ4: lz4_open,
    CompressType.GZIP: gz_open,
}


def get_supported_compressions():
    return set(k for k, v in decompressors.items() if v is not None)


def get_compressobj(compress):
    return compressobjs[compress]()


def get_decompressobj(compress):
    return decompressobjs[compress]()


def open_compression_file(file, compress):
    if compress != CompressType.NONE:
        file = compress_openers[compress](file, 'wb')
    return file


def open_decompression_file(file, compress):
    if compress != CompressType.NONE:
        file = compress_openers[compress](file, 'rb')
    return file


file_header = namedtuple('FileHeader', 'version nbytes compress')
HEADER_LENGTH = 12


def read_file_header(file):
    if hasattr(file, 'read'):
        header_bytes = file.read(HEADER_LENGTH)
    else:
        header_bytes = file[:HEADER_LENGTH]
    version, = struct.unpack('<H', header_bytes[:2])
    nbytes, = struct.unpack('<Q', header_bytes[2:10])
    compress, = struct.unpack('<H', header_bytes[10:12])
    return file_header(version, nbytes, CompressType.from_tag(compress))


def write_file_header(file, header):
    file.write(struct.pack('<H', header.version))
    file.write(struct.pack('<Q', header.nbytes))
    file.write(struct.pack('<H', header.compress.tag))


def peek_file_header(file):
    pos = file.tell()
    try:
        return read_file_header(file)
    finally:
        file.seek(pos)


def load(file, raw=False):
    header = read_file_header(file)
    file = open_decompression_file(file, header.compress)

    try:
        buf = file.read()
    finally:
        if header.compress != CompressType.NONE:
            file.close()

    if raw:
        return buf
    else:
        return pyarrow.deserialize(memoryview(buf), mars_serialize_context())


def loads(buf, raw=False):
    mv = memoryview(buf)
    header = read_file_header(mv)
    compress = header.compress

    if compress == CompressType.NONE:
        data = buf[HEADER_LENGTH:]
    else:
        data = decompressors[compress](mv[HEADER_LENGTH:])
    if raw:
        return data
    else:
        return pyarrow.deserialize(memoryview(data), mars_serialize_context())


def dump(obj, file, compress=CompressType.NONE, raw=False):
    if raw:
        serialized = obj
        data_size = len(serialized)
    else:
        serialized = pyarrow.serialize(obj, mars_serialize_context())
        data_size = serialized.total_bytes

    write_file_header(file, file_header(SERIAL_VERSION, data_size, compress))
    file = open_compression_file(file, compress)
    try:
        if raw:
            file.write(serialized)
        else:
            serialized.write_to(file)
    finally:
        if compress != CompressType.NONE:
            file.close()
    return


def dumps(obj, compress=CompressType.NONE, raw=False):
    sio = BytesIO()
    dump(obj, sio, compress=compress, raw=raw)
    return sio.getvalue()


def peek_serialized_size(file):
    return peek_file_header(file).nbytes


def _serialize_numpy_array_list(obj):
    if obj.dtype.str != '|O':
        # Make the array c_contiguous if necessary so that we can call change
        # the view.
        if not obj.flags.c_contiguous:
            obj = np.ascontiguousarray(obj)
        return obj.view('uint8'), obj.dtype.str
    else:
        return obj.tolist(), obj.dtype.str


def _deserialize_numpy_array_list(data):
    if data[1] != '|O':
        assert data[0].dtype == np.uint8
        return data[0].view(data[1])
    else:
        return np.array(data[0], dtype=np.dtype(data[1]))


def _serialize_sparse_csr_list(obj):
    sizes = []
    outputs = [None, None]
    for item in (obj.data, obj.indices, obj.indptr):
        serialized_items = _serialize_numpy_array_list(item)
        outputs.extend(serialized_items)
        sizes.append(len(serialized_items))
    outputs[0] = struct.pack('<' + 'B' * len(sizes), *sizes)
    outputs[1] = struct.pack('<' + 'L' * len(obj.shape), *obj.shape)
    return tuple(outputs)


def _deserialize_sparse_csr_list(data):
    sizes = struct.unpack('<' + 'B' * len(data[0]), data[0])
    shape = struct.unpack('<' + 'L' * (len(data[1]) // 4), data[1])
    data_parts = []
    pos = 2
    for size in sizes:
        data_parts.append(_deserialize_numpy_array_list(data[pos:pos + size]))
        pos += size

    empty_arr = np.zeros(0, dtype=data_parts[0].dtype)
    if len(shape) == 1:
        sps_shape = (1, shape[0])
    else:
        sps_shape = shape
    target_csr = sps.coo_matrix((empty_arr, (empty_arr,) * 2), shape=sps_shape,
                                dtype=data_parts[0].dtype).tocsr()
    target_csr.data, target_csr.indices, target_csr.indptr = data_parts
    return SparseNDArray(target_csr, shape=shape)


_serialize_context = None


def mars_serialize_context():
    global _serialize_context
    if _serialize_context is None:
        ctx = pyarrow.default_serialization_context()
        ctx.register_type(SparseNDArray, 'mars.SparseNDArray',
                          custom_serializer=_serialize_sparse_csr_list,
                          custom_deserializer=_deserialize_sparse_csr_list)
        _serialize_context = ctx
    return _serialize_context
