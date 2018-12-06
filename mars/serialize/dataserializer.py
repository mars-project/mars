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
import struct

from ..compat import six, BytesIO

try:
    import numpy as np
except ImportError:
    np = None
try:
    import scipy.sparse as sps
except ImportError:
    sps = None
try:
    import cupy as cp
except ImportError:
    cp = None

from ..lib.sparse import SparseNDArray

try:
    import pyarrow
except ImportError:
    pyarrow = None


BUFFER_SIZE = 256 * 1024


class DummyCompress(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @staticmethod
    def decompress(data):
        return data


try:
    import lz4.frame
    try:
        lz4.frame._compression.BUFFER_SIZE = BUFFER_SIZE
    except AttributeError:
        pass
    lz4_open = functools.partial(lz4.frame.open, block_size=lz4.frame.BLOCKSIZE_MAX1MB)
    lz4_compress = functools.partial(lz4.frame.compress, block_size=lz4.frame.BLOCKSIZE_MAX1MB)
    lz4_compressobj = lz4.frame.LZ4FrameCompressor
    lz4_decompress = lz4.frame.decompress
    lz4_decompressobj = lz4.frame.LZ4FrameDecompressor
except ImportError:
    lz4_open = None
    lz4_compress, lz4_compressobj = None, None
    lz4_decompress, lz4_decompressobj = None, None

SERIAL_VERSION = 0


DATA_FLAG_DENSE = 0
DATA_FLAG_CSR = 1

COMPRESS_FLAG_NONE = 0
COMPRESS_FLAG_LZ4 = 1


compressors = {
    COMPRESS_FLAG_LZ4: lz4_compress,
}
compressobjs = {
    COMPRESS_FLAG_NONE: DummyCompress,
    COMPRESS_FLAG_LZ4: lz4_compressobj,
}
decompressors = {
    COMPRESS_FLAG_LZ4: lz4_decompress,
}
decompressobjs = {
    COMPRESS_FLAG_NONE: DummyCompress,
    COMPRESS_FLAG_LZ4: lz4_decompressobj,
}
compress_openers = {
    COMPRESS_FLAG_LZ4: lz4_open,
}


def load(file, raw=False):
    # version bytes
    file.read(2)
    # total size
    file.read(8)

    compress = struct.unpack('<H', file.read(2))[0]

    if compress != COMPRESS_FLAG_NONE:
        file = compress_openers[compress](file, 'rb')

    try:
        buf = file.read()
    finally:
        if compress != COMPRESS_FLAG_NONE:
            file.close()

    if raw:
        return buf
    else:
        return pyarrow.deserialize(memoryview(buf), mars_serialize_context())


class CompressBufferReader(object):
    def __init__(self, buf, compress):
        self._total_bytes = len(buf)
        self._compress_method = compress
        self._compressor = compressobjs[compress]()
        self._pos = 0
        self._mv = memoryview(buf)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def read(self, byte_num):
        if self._pos == self._total_bytes:
            return b''
        bio = BytesIO()
        if self._pos == 0:
            bio.write(struct.pack('<H', SERIAL_VERSION)
                      + struct.pack('<Q', self._total_bytes)
                      + struct.pack('<H', self._compress_method))
            bio.write(self._compressor.begin())
        while self._pos < self._total_bytes and bio.tell() < byte_num:
            end_pos = min(self._pos + byte_num, self._total_bytes)
            bio.write(self._compressor.compress(self._mv[self._pos:end_pos]))
            if end_pos == self._total_bytes:
                bio.write(self._compressor.flush())
            self._pos = end_pos
        return bio.getvalue()

    def close(self):
        self._mv = None
        self._compressor = None


class DecompressBufferWriter(object):
    def __init__(self, buf):
        import pyarrow
        self._buf = buf
        self._writer = pyarrow.FixedSizeBufferWriter(buf)
        self._writer.set_memcopy_threads(6)
        self._decompressor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def write(self, data):
        mv = memoryview(data)
        if self._decompressor is not None:
            self._writer.write(self._decompressor.decompress(mv))
        else:
            if len(data) < 12:
                raise IOError('Block size too small')
            compress = struct.unpack('<H', mv[10:12])[0]
            self._decompressor = decompressobjs[compress]()
            if len(data) > 12:
                self._writer.write(self._decompressor.decompress(mv[12:]))

    def close(self):
        if self._writer is not None:
            self._writer.close()
        self._writer = None
        self._buf = None
        self._decompressor = None


def loads(buf, raw=False):
    mv = memoryview(buf)
    compress = struct.unpack('<H', mv[10:12])[0]

    if compress == COMPRESS_FLAG_NONE:
        data = buf[12:]
    else:
        data = decompressors[compress](mv[12:])
    if raw:
        return data
    else:
        return pyarrow.deserialize(memoryview(data), mars_serialize_context())


def dump(obj, file, compress=COMPRESS_FLAG_NONE, raw=False):
    if raw:
        serialized = obj
        data_size = len(serialized)
    else:
        serialized = pyarrow.serialize(obj, mars_serialize_context())
        data_size = serialized.total_bytes

    # version bytes
    file.write(struct.pack('<H', SERIAL_VERSION))
    file.write(struct.pack('<Q', data_size))
    file.write(struct.pack('<H', compress))

    if compress != COMPRESS_FLAG_NONE:
        file = compress_openers[compress](file, 'wb')

    try:
        if raw:
            file.write(serialized)
        else:
            serialized.write_to(file)
    finally:
        if compress != COMPRESS_FLAG_NONE:
            file.close()
    return


def dumps(obj, compress=COMPRESS_FLAG_NONE, raw=False):
    sio = six.BytesIO()
    dump(obj, sio, compress=compress, raw=raw)
    return sio.getvalue()


def peek_serialized_size(file):
    try:
        file.seek(2)
        return struct.unpack('<Q', file.read(8))[0]
    finally:
        file.seek(0)


class DataTuple(tuple):
    pass


try:
    import pyarrow

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
        target_csr = sps.coo_matrix((empty_arr, (empty_arr,) * 2), shape=shape,
                                    dtype=data_parts[0].dtype).tocsr()
        target_csr.data, target_csr.indices, target_csr.indptr = data_parts
        return SparseNDArray(target_csr)

    def _serialize_data_tuple(obj):
        data_meta = []
        outputs = [None]
        for item in obj:
            if isinstance(item, np.ndarray):
                serialized_items = _serialize_numpy_array_list(item)
                outputs.extend(serialized_items)
                data_meta.append(DATA_FLAG_DENSE)
                data_meta.append(len(serialized_items))
            else:
                serialized_items = _serialize_sparse_csr_list(item)
                outputs.extend(serialized_items)
                data_meta.append(DATA_FLAG_CSR)
                data_meta.append(len(serialized_items))
        outputs[0] = struct.pack('<' + 'B' * len(data_meta), *data_meta)
        return tuple(outputs)

    def _deserialize_data_tuple(data):
        data_meta = struct.unpack('<' + 'B' * len(data[0]), data[0])
        pos = 1
        data_parts = []
        for flag, size in zip(data_meta[0::2], data_meta[1::2]):
            if flag == DATA_FLAG_DENSE:
                data_parts.append(_deserialize_numpy_array_list(data[pos:pos + size]))
            elif flag == DATA_FLAG_CSR:
                data_parts.append(_deserialize_sparse_csr_list(data[pos:pos + size]))
            pos += size
        return DataTuple(data_parts)


    _serialize_context = None


    def mars_serialize_context():
        global _serialize_context
        if _serialize_context is None:
            ctx = pyarrow.default_serialization_context()
            ctx.register_type(SparseNDArray, 'mars.SparseNDArray',
                              custom_serializer=_serialize_sparse_csr_list,
                              custom_deserializer=_deserialize_sparse_csr_list)
            ctx.register_type(DataTuple, 'mars.DataTuple',
                              custom_serializer=_serialize_data_tuple,
                              custom_deserializer=_deserialize_data_tuple)
            _serialize_context = ctx
        return _serialize_context

except ImportError:
    pass
