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

import pyarrow

from ..compat import BytesIO
from ..serialize.dataserializer import CompressType, get_compressobj, get_decompressobj, \
    HEADER_LENGTH, file_header, read_file_header, write_file_header, SERIAL_VERSION


class WorkerBufferIO(object):
    """
    File-like object handling data with file header
    """
    def __init__(self, mode='r', compress_in=None, compress_out=None, block_size=8192):
        """
        :param mode: 'r' indicates read, or 'w' indicates write
        :param compress_in: compression type inside
        :param compress_out: compression type outside
        :param block_size: size of data block when copying
        """
        self._mode = mode
        self._block_size = block_size
        self._compress_type_in = compress_in or CompressType.NONE
        self._compress_type_out = compress_out or CompressType.NONE

        if 'w' in mode:
            self._compressor_in = None
            self._decompressor_out = None
        else:
            self._remain_buf = None
            self._remain_offset = None
            self._block_iterator = self._iter_blocks()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def _read_block(self, size):
        """
        Read a data block from data source with given size.
        """
        raise NotImplementedError

    def _read_header(self):
        """
        Read the data header of data source.
        """
        raise NotImplementedError

    def _iter_blocks(self):
        """
        Returns a generator providing data blocks. The sizes of data blocks can vary.
        """
        bio = BytesIO()
        compressor_out = get_compressobj(self._compress_type_out)

        header = self._read_header()
        self._compress_type_in = header.compress
        decompressor_in = get_decompressobj(header.compress)

        new_header = file_header(header.version, header.nbytes, self._compress_type_out)
        write_file_header(bio, new_header)

        if self._compress_type_in == self._compress_type_out:
            handle_block = lambda v: v

            compressor_out = get_compressobj(CompressType.NONE)
            decompressor_in = get_decompressobj(CompressType.NONE)
        else:
            handle_block = lambda v: compressor_out.compress(decompressor_in.decompress(v))
            if hasattr(compressor_out, 'begin'):
                bio.write(compressor_out.begin())

        # yield file header and compress header
        yield bio.getvalue()

        copy_size = self._block_size
        while True:
            block = self._read_block(copy_size)
            if not block:
                break
            buf = handle_block(block)
            if buf:
                # only yield when some data are produced by the compressor
                yield buf

        if hasattr(compressor_out, 'flush'):
            yield compressor_out.flush()

    def read(self, size=-1):
        bio = BytesIO()
        if self._remain_buf is not None:
            if size < 0 or len(self._remain_buf) <= self._remain_offset + size:
                bio.write(self._remain_buf[self._remain_offset:])
                self._remain_buf = None
            else:
                right = self._remain_offset + size
                bio.write(self._remain_buf[self._remain_offset:right])
                self._remain_offset = right
            if bio.tell() == size:
                return bio.getvalue()
        while True:
            try:
                block = next(self._block_iterator)
            except StopIteration:
                break
            if size < 0 or bio.tell() + len(block) <= size:
                bio.write(block)
                self._remain_buf = None
            else:
                offset = self._remain_offset = size - bio.tell()
                self._remain_buf = block
                bio.write(block[:offset])
            if bio.tell() == size:
                break
        return bio.getvalue()

    def _write_header(self, header):
        """
        Write a data header to the output file.
        """
        raise NotImplementedError

    def _write_block(self, d):
        """
        Write a data block to the output file.
        """
        raise NotImplementedError

    def _write_with_compression(self, mv):
        data_len = len(mv)
        copy_size = self._block_size
        offset = 0

        if self._compressor_in is not None:
            compressor_in = self._compressor_in
        else:
            compressor_in = self._compressor_in = get_compressobj(self._compress_type_in)
            if hasattr(compressor_in, 'begin') and self._compress_type_in != self._compress_type_out:
                self._write_block(compressor_in.begin())

        if self._compress_type_in == self._compress_type_out:
            self._compressor_in = get_compressobj(CompressType.NONE)
            self._decompressor_out = get_decompressobj(CompressType.NONE)

            handle_block = lambda v: v
        else:
            handle_block = lambda v: compressor_in.compress(decompressor_out.decompress(v))

        decompressor_out = self._decompressor_out

        while offset < data_len:
            right = min(data_len, offset + copy_size)
            cblock = handle_block(mv[offset:right])
            if cblock:
                self._write_block(cblock)
            offset = right

    def write(self, d):
        size = len(d)
        mv = memoryview(d)
        if self._decompressor_out is not None:
            # header already processed, we can continue
            # with decompression
            self._write_with_compression(mv)
        else:
            # header not processed, we need to read header
            # to get compression method and build decompressor
            if size < HEADER_LENGTH:
                raise IOError('Block size too small')

            header = read_file_header(mv)
            self._compress_type_out = header.compress
            new_header = file_header(header.version, header.nbytes, self._compress_type_in)
            self._write_header(new_header)

            self._decompressor_out = get_decompressobj(header.compress)
            if size > HEADER_LENGTH:
                self._write_with_compression(mv[HEADER_LENGTH:])

    def close(self):
        if 'w' in self._mode and hasattr(self._compressor_in, 'flush'):
            self._write_block(self._compressor_in.flush())

        self._compressor_in = self._decompressor_out = None
        self._remain_buf = self._block_iterator = None


class ArrowBufferIO(WorkerBufferIO):
    """
    File-like object mocking object stored in shared memory as file with header
    """
    def __init__(self, buf, mode='r', compress_in=None, compress_out=None, block_size=8192):
        super(ArrowBufferIO, self).__init__(
            mode=mode, compress_in=compress_in, compress_out=compress_out, block_size=block_size)

        self._buf = buf
        if 'r' in mode:
            self._mv = memoryview(buf)
            self._nbytes = len(buf)
            self._buf_offset = 0
            self._writer = None
        else:
            self._writer = pyarrow.FixedSizeBufferWriter(buf)
            self._writer.set_memcopy_threads(6)

    def _read_header(self):
        return file_header(SERIAL_VERSION, self._nbytes, self._compress_type_in)

    def _read_block(self, size):
        right = min(self._nbytes, self._buf_offset + size)
        ret = self._mv[self._buf_offset:right]
        self._buf_offset = right
        return ret

    def _write_header(self, header):
        pass

    def _write_block(self, d):
        self._writer.write(d)

    def close(self):
        if self._writer is not None:
            self._writer.close()
        self._writer = self._mv = self._buf = None

        super(ArrowBufferIO, self).close()


class FileBufferIO(WorkerBufferIO):
    """
    File-like object handling input of one compression type and outputs another
    """
    def __init__(self, file, mode='r', compress_in=None, compress_out=None,
                 block_size=8192, managed=True):
        super(FileBufferIO, self).__init__(
            mode=mode, compress_in=compress_in, compress_out=compress_out, block_size=block_size)

        self._managed = managed
        self._file = file

    def _read_header(self):
        return read_file_header(self._file)

    def _read_block(self, size):
        return self._file.read(size)

    def _write_header(self, header):
        return write_file_header(self._file, header)

    def _write_block(self, d):
        self._file.write(d)

    def close(self):
        super(FileBufferIO, self).close()
        if self._file and self._managed:
            self._file.close()
        self._file = None
