# cython: profile=True
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
# cython: annotate = True
from cpython cimport *
from libc.stdint cimport *
from libc.string cimport memcpy
from libcpp cimport bool as c_bool


cdef class Buffer:
    """This class implements the Python 'buffer protocol', which allows
    us to use it for calls into Python libraries without having to
    copy the data."""
    cdef:
        uint8_t *c_buffer
        # hold python buffer reference count
        object data
        Py_ssize_t _size
        Py_ssize_t shape[1]
        Py_ssize_t stride[1]
        public int reader_index, writer_index

    def __init__(self,  data not None, int offset=0, length=None):
        self.data = data
        assert 0 <= offset <= len(data), f'offset {offset} length {len(data)}'
        cdef int length_
        if length is None:
            length_ = len(data) - offset
        else:
            length_ = length
        assert length_ >= 0, f'length should be >= 0 but got {length}'
        self._size = length_
        cdef uint8_t* ptr
        ptr = get_address(data) + offset
        self.c_buffer = ptr
        self.reader_index = 0
        self.writer_index = 0

    @classmethod
    def allocate(cls, int32_t size):
        return Buffer(bytearray(size))

    def reserve(self, new_size):
        assert 0 < new_size < 2 ** 31 - 1
        if new_size > self._size:
            self.data = bytearray(new_size)
            addr = get_address(self.data)
            memcpy(self.c_buffer, addr, self._size)
            self._size = new_size
            self.c_buffer = addr

    def put_bool(self, uint32_t offset, c_bool v):
        self.check_bound(offset, 1)
        memcpy(self.c_buffer + offset, <uint8_t*>(&v), 1)

    def put_int8(self, uint32_t offset, int8_t v):
        self.check_bound(offset, 1)
        memcpy(self.c_buffer + offset, <uint8_t*>(&v), 1)

    def put_int16(self, uint32_t offset, int16_t v):
        self.check_bound(offset, 2)
        memcpy(self.c_buffer + offset, <uint8_t*>(&v), 2)

    def put_int32(self, uint32_t offset, int32_t v):
        self.check_bound(offset, 4)
        memcpy(self.c_buffer + offset, <uint8_t*>(&v), 4)

    def put_int64(self, uint32_t offset, int64_t v):
        self.check_bound(offset, 8)
        memcpy(self.c_buffer + offset, <uint8_t*>(&v), 8)

    def put_float(self, uint32_t offset, float v):
        self.check_bound(offset, 4)
        memcpy(self.c_buffer + offset, <uint8_t*>(&v), 4)

    def put_double(self, uint32_t offset, double v):
        self.check_bound(offset, 8)
        memcpy(self.c_buffer + offset, <uint8_t*>(&v), 8)

    def put_binary(self,
                   uint32_t offset,
                   v not None,
                   int32_t src_offset, int32_t length):
        if length == 0:  # access an emtpy buffer may raise out-of-bound exception.
            return
        self.check_bound(offset, length - src_offset)
        cdef uint8_t* ptr = get_address(v)
        memcpy(self.c_buffer + offset, ptr + src_offset, length)

    def get_bool(self, uint32_t offset):
        self.check_bound(offset, 1)
        return (<c_bool *>(self.c_buffer + offset))[0]

    def get_int8(self, uint32_t offset):
        self.check_bound(offset, 1)
        return (<int8_t *>(self.c_buffer + offset))[0]

    def get_int16(self, uint32_t offset):
        self.check_bound(offset, 2)
        return (<int16_t *>(self.c_buffer + offset))[0]

    def get_int32(self, uint32_t offset):
        self.check_bound(offset, 4)
        return (<int32_t *>(self.c_buffer + offset))[0]

    def get_int64(self, uint32_t offset):
        self.check_bound(offset, 8)
        return (<int64_t *>(self.c_buffer + offset))[0]

    def get_float(self, uint32_t offset):
        self.check_bound(offset, 4)
        return (<float *>(self.c_buffer + offset))[0]

    def get_double(self, uint32_t offset):
        self.check_bound(offset, 8)
        return (<double *>(self.c_buffer + offset))[0]

    def get_binary(self, uint32_t offset, uint32_t nbytes):
        if nbytes == 0:
            return b""
        self.check_bound(offset, nbytes)
        cdef unsigned char* binary_data = self.c_buffer + offset
        # FIXME slice may cause memory leak.
        return binary_data[:nbytes]

    cdef check_bound(self, offset, length):
        size_ = self._size
        assert offset + length <= size_, \
            f"Address range {offset, offset + length} out of bound {0, size_}"

    def write_bool(self, value):
        self.grow(1)
        self.put_bool(self.writer_index, value)
        self.writer_index += 1

    def write_int8(self, value):
        self.grow(1)
        self.put_int8(self.writer_index, value)
        self.writer_index += 1

    def write_int16(self, value):
        self.grow(2)
        self.put_int16(self.writer_index, value)
        self.writer_index += 2

    def write_int32(self, value):
        self.grow(4)
        self.put_int32(self.writer_index, value)
        self.writer_index += 4

    def write_int64(self, value):
        self.grow(8)
        self.put_int64(self.writer_index, value)
        self.writer_index += 8

    def write_float(self, value):
        self.grow(4)
        self.put_float(self.writer_index, value)
        self.writer_index += 4

    def write_double(self, value):
        self.grow(8)
        self.put_double(self.writer_index, value)
        self.writer_index += 8

    cpdef write_varint32(self, int32_t v):
        self.grow(5)
        cdef:
            int64_t value = v
            int64_t offset = self.writer_index
        if value >> 7 == 0:
            self.c_buffer[offset] = <int8_t>value
            self.writer_index += 1
            return 1
        if value >> 14 == 0:
            self.c_buffer[offset] = <int8_t>((value & 0x7F) | 0x80)
            self.c_buffer[offset+1] = <int8_t>(value >> 7)
            self.writer_index += 2
            return 2
        if value >> 21 == 0:
            self.c_buffer[offset] = <int8_t>((value & 0x7F) | 0x80)
            self.c_buffer[offset+1] = <int8_t>(value >> 7 | 0x80)
            self.c_buffer[offset+2] = <int8_t>(value >> 14)
            self.writer_index += 3
            return 3
        if value >> 28 == 0:
            self.c_buffer[offset] = <int8_t>((value & 0x7F) | 0x80)
            self.c_buffer[offset+1] = <int8_t>(value >> 7 | 0x80)
            self.c_buffer[offset+2] = <int8_t>(value >> 14 | 0x80)
            self.c_buffer[offset+3] = <int8_t>(value >> 21)
            self.writer_index += 4
            return 4
        self.c_buffer[offset] = <int8_t>((value & 0x7F) | 0x80)
        self.c_buffer[offset+1] = <int8_t>(value >> 7 | 0x80)
        self.c_buffer[offset+2] = <int8_t>(value >> 14 | 0x80)
        self.c_buffer[offset+3] = <int8_t>(value >> 21 | 0x80)
        self.c_buffer[offset+4] = <int8_t>(value >> 28)
        self.writer_index += 5
        return 5

    def read_varint32(self):
        cdef:
            uint32_t read_bytes_length = 1
            int8_t b
            int32_t result
            uint32_t position = self.reader_index
            int8_t * arr = <int8_t *> (self.c_buffer + position)
        if self.size() - self.reader_index > 5:
            b = arr[0]
            result = b & 0x7F
            if (b & 0x80) != 0:
              read_bytes_length += 1
              b = arr[1]
              result |= (b & 0x7F) << 7
              if (b & 0x80) != 0:
                  read_bytes_length += 1
                  b = arr[2]
                  result |= (b & 0x7F) << 14
                  if (b & 0x80) != 0:
                      read_bytes_length += 1
                      b = arr[3]
                      result |= (b & 0x7F) << 21
                      if (b & 0x80) != 0:
                          read_bytes_length += 1
                          b = arr[4]
                          result |= (b & 0x7F) << 28
            self.reader_index += read_bytes_length
            return result
        else:
            b = self.read_int8()
            result = b & 0x7F
            if (b & 0x80) != 0:
                b = self.read_int8()
                result |= (b & 0x7F) << 7
                if (b & 0x80) != 0:
                    b = self.read_int8()
                    result |= (b & 0x7F) << 14
                    if (b & 0x80) != 0:
                        b = self.read_int8()
                        result |= (b & 0x7F) << 21
                        if (b & 0x80) != 0:
                            b = self.read_int8()
                            result |= (b & 0x7F) << 28
            return result

    def write_varint64(self, int64_t v):
        cdef:
            int64_t value = v
            int64_t offset = self.writer_index
        if value >> 7 == 0:
            self.grow(1)
            self.c_buffer[offset] = <int8_t>value
            self.writer_index += 1
            return 1
        if value >> 14 == 0:
            self.grow(2)
            self.c_buffer[offset] = <int8_t>((value & 0x7F) | 0x80)
            self.c_buffer[offset+1] = <int8_t>(value >> 7)
            self.writer_index += 2
            return 2
        if value >> 21 == 0:
            self.grow(3)
            self.c_buffer[offset] = <int8_t>((value & 0x7F) | 0x80)
            self.c_buffer[offset+1] = <int8_t>(value >> 7 | 0x80)
            self.c_buffer[offset+2] = <int8_t>(value >> 14)
            self.writer_index += 3
            return 3
        if value >> 28 == 0:
            self.grow(4)
            self.c_buffer[offset] = <int8_t>((value & 0x7F) | 0x80)
            self.c_buffer[offset+1] = <int8_t>(value >> 7 | 0x80)
            self.c_buffer[offset+2] = <int8_t>(value >> 14 | 0x80)
            self.c_buffer[offset+3] = <int8_t>(value >> 21)
            self.writer_index += 4
            return 4
        if value >> 35 == 0:
            self.grow(5)
            self.c_buffer[offset] = <int8_t>((value & 0x7F) | 0x80)
            self.c_buffer[offset+1] = <int8_t>(value >> 7 | 0x80)
            self.c_buffer[offset+2] = <int8_t>(value >> 14 | 0x80)
            self.c_buffer[offset+3] = <int8_t>(value >> 21)
            self.c_buffer[offset+4] = <int8_t>(value >> 28)
            self.writer_index += 5
            return 5
        if value >> 42 == 0:
            self.grow(6)
            self.c_buffer[offset] = <int8_t>((value & 0x7F) | 0x80)
            self.c_buffer[offset+1] = <int8_t>(value >> 7 | 0x80)
            self.c_buffer[offset+2] = <int8_t>(value >> 14 | 0x80)
            self.c_buffer[offset+3] = <int8_t>(value >> 21)
            self.c_buffer[offset+4] = <int8_t>(value >> 28)
            self.c_buffer[offset+5] = <int8_t>(value >> 35)
            self.writer_index += 6
            return 6
        if value >> 49 == 0:
            self.grow(7)
            self.c_buffer[offset] = <int8_t>((value & 0x7F) | 0x80)
            self.c_buffer[offset+1] = <int8_t>(value >> 7 | 0x80)
            self.c_buffer[offset+2] = <int8_t>(value >> 14 | 0x80)
            self.c_buffer[offset+3] = <int8_t>(value >> 21)
            self.c_buffer[offset+4] = <int8_t>(value >> 28)
            self.c_buffer[offset+5] = <int8_t>(value >> 35)
            self.c_buffer[offset+6] = <int8_t>(value >> 42)
            self.writer_index += 7
            return 7
        if value >> 56 == 0:
            self.grow(8)
            self.c_buffer[offset] = <int8_t>((value & 0x7F) | 0x80)
            self.c_buffer[offset+1] = <int8_t>(value >> 7 | 0x80)
            self.c_buffer[offset+2] = <int8_t>(value >> 14 | 0x80)
            self.c_buffer[offset+3] = <int8_t>(value >> 21)
            self.c_buffer[offset+4] = <int8_t>(value >> 28)
            self.c_buffer[offset+5] = <int8_t>(value >> 35)
            self.c_buffer[offset+6] = <int8_t>(value >> 42)
            self.c_buffer[offset+7] = <int8_t>(value >> 49)
            self.writer_index += 8
            return 8
        self.grow(9)
        self.c_buffer[offset] = <int8_t> ((value & 0x7F) | 0x80)
        self.c_buffer[offset + 1] = <int8_t> (value >> 7 | 0x80)
        self.c_buffer[offset + 2] = <int8_t> (value >> 14 | 0x80)
        self.c_buffer[offset + 3] = <int8_t> (value >> 21)
        self.c_buffer[offset + 4] = <int8_t> (value >> 28)
        self.c_buffer[offset + 5] = <int8_t> (value >> 35)
        self.c_buffer[offset + 6] = <int8_t> (value >> 42)
        self.c_buffer[offset + 7] = <int8_t> (value >> 49)
        self.c_buffer[offset + 8] = <int8_t> (value >> 56)
        self.writer_index += 9
        return 9

    def write_binary(self, value, src_offset=0, length=None):
        if length is None:
            length = len(value) - src_offset
        self.grow(length)
        self.write_varint32(length)
        self.put_binary(self.writer_index, value, src_offset, length)
        self.writer_index += length

    def read_binary(self):
        cdef int32_t length = self.read_varint32()
        value = self.get_binary(self.reader_index, length)
        self.reader_index += length
        return value

    cdef write_string(self, value):
        if PyUnicode_Check(value):
            encoded = PyUnicode_AsEncodedString(value, "UTF-8", "encode to utf-8 error")
            self.write_binary(encoded)
        else:
            raise TypeError("value should be unicode, but get type of {}"
                            .format(type(value)))

    def read_string(self):
        return self.read_binary().decode("UTF-8")



    def grow(self, needed_size):
        self.ensure(self.writer_index + needed_size)

    def ensure(self, length):
        if length > self._size:
            self.reserve(length * 2)

    def read_bool(self):
        value = self.get_bool(self.reader_index)
        self.reader_index += 1
        return value

    def read_int8(self):
        value = self.get_int8(self.reader_index)
        self.reader_index += 1
        return value

    def read_int16(self):
        value = self.get_int16(self.reader_index)
        self.reader_index += 2
        return value

    def read_int32(self):
        value = self.get_int32(self.reader_index)
        self.reader_index += 4
        return value

    def read_int64(self):
        value = self.get_int64(self.reader_index)
        self.reader_index += 8
        return value

    def read_float(self):
        value = self.get_float(self.reader_index)
        self.reader_index += 4
        return value

    def read_double(self):
        value = self.get_double(self.reader_index)
        self.reader_index += 8
        return value

    def readline(self, size=None):
        raise NotImplementedError

    def __len__(self):
        return self._size

    def size(self):
        return self._size

    def to_bytes(self, int32_t offset=0, int32_t length=0) -> bytes:
        if length != 0:
            assert 0 < length <= self._size,\
                f"length {length} size {self._size}"
        else:
            length = self._size
        return (self.c_buffer + offset)[:length]

    def to_pybytes(self):
        return self.to_bytes()

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef Py_ssize_t itemsize = 1
        self.shape[0] = self._size
        self.stride[0] = itemsize
        buffer.buf = <char *> self.c_buffer
        buffer.format = 'B'
        buffer.internal = NULL                  # see References
        buffer.itemsize = itemsize
        buffer.len = self._size  # product(shape) * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.stride
        buffer.suboffsets = NULL                # for pointer arrays only

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __str__(self):
        return "Buffer(reader_index={}, writer_index={}, size={})".format(
            self.reader_index, self.writer_index, self._size
        )


cdef uint8_t* get_address(v):
    view = memoryview(v)
    dtype = view.format
    cdef:
        const char[:] signed_char_data
        const unsigned char[:] unsigned_data
        const int16_t[:] signed_short_data
        const int32_t[:] signed_int_data
        const int64_t[:] signed_long_data
        const float[:] signed_float_data
        const double[:] signed_double_data
        uint8_t* ptr
    if dtype == "b":
        signed_char_data = v
        ptr = <uint8_t*>(&signed_char_data[0])
    elif dtype == "B":
        unsigned_data = v
        ptr = <uint8_t*>(&unsigned_data[0])
    elif dtype == "h":
        signed_short_data = v
        ptr = <uint8_t*>(&signed_short_data[0])
    elif dtype == "i":
        signed_int_data = v
        ptr = <uint8_t*>(&signed_int_data[0])
    elif dtype == "l":
        signed_long_data = v
        ptr = <uint8_t*>(&signed_long_data[0])
    elif dtype == "f":
        signed_float_data = v
        ptr = <uint8_t*>(&signed_float_data[0])
    elif dtype == "d":
        signed_double_data = v
        ptr = <uint8_t*>(&signed_double_data[0])
    else:
        raise Exception(f"Unsupported buffer of type {type(v)} and format {dtype}")
    return ptr
