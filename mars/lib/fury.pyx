# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
# cython: annotate = True

from cpython cimport *
from libc.stdint cimport *
from libc.string cimport memcpy
from libcpp cimport bool as c_bool

cimport cython


cdef extern from "Python.h":
    const char* PyUnicode_AsUTF8AndSize(object obj, Py_ssize_t *l) except NULL


cdef int32_t _max_buffer_size = 2 ** 31 - 1

@cython.final
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
        char *errors

    def __init__(self, unsigned char[:] data not None, int32_t offset, int32_t length):
        self.data = data
        if offset < 0 or offset > data.nbytes:
            raise ValueError(f"offset {offset} length {data.nbytes}")
        if length < 0:
            raise ValueError(f"length should be >= 0 but got {length}")
        self._size = length
        cdef uint8_t* ptr
        ptr = &data[0] + offset
        self.c_buffer = ptr
        self.reader_index = 0
        self.writer_index = 0

    @staticmethod
    cdef from_buffer(unsigned char[:] data):
        cdef Buffer buffer = Buffer.__new__(Buffer)
        buffer.reader_index = 0
        buffer.writer_index = 0
        buffer.c_buffer = <uint8_t *>(&data[0])
        buffer.data = data
        buffer._size = data.nbytes

    @classmethod
    def allocate(cls, int32_t size):
        return Buffer(bytearray(size), 0, size)

    def point_to_bytes(self, bytes_buf: bytes):
        cdef const unsigned char[:] data = bytes_buf
        self.reader_index = 0
        self.writer_index = 0
        self.c_buffer = <uint8_t *>(&data[0])
        self.data = data
        self._size = data.nbytes

    cpdef inline grow(self, int32_t needed_size):
        cdef int32_t length = self.writer_index + needed_size
        if length > self._size:
            assert 0 < length < _max_buffer_size
            old_data = self.data  # hold reference to avoid gc
            self.data = bytearray(length + length)
            addr = _get_address(self.data)
            memcpy(addr, self.c_buffer, self._size)
            self._size = length + length
            self.c_buffer = addr

    cdef inline check_bound(self, uint32_t offset, uint32_t length):
        if offset + length > self._size:
            raise Exception(f"Address range {offset, offset + length} out of bound {0, self._size}")

    cpdef inline unsafe_put_bool(self, uint32_t offset, c_bool v):
        if v:
            (self.c_buffer + offset)[0] = 1
        else:
            (self.c_buffer + offset)[0] = 0

    cpdef inline c_bool get_bool(self, uint32_t offset):
        self.check_bound(offset, 1)
        cdef uint8_t v = (self.c_buffer + offset)[0]
        if v == 0:
            return False
        else:
            return True

    cpdef inline write_bool(self, c_bool value):
        self.grow(1)
        self.unsafe_put_bool(self.writer_index, value)
        self.writer_index += 1

    cpdef inline c_bool read_bool(self):
        value = self.get_bool(self.reader_index)
        self.reader_index += 1
        return value

    cpdef inline unsafe_put_int8(self, uint32_t offset, int8_t v):
        (<int8_t *>(self.c_buffer + offset))[0] = v

    cpdef inline int8_t get_int8(self, uint32_t offset):
        self.check_bound(offset, 1)
        return (<int8_t *>(self.c_buffer + offset))[0]

    cpdef inline uint8_t get_uint8(self, uint32_t offset):
        self.check_bound(offset, 1)
        return (self.c_buffer + offset)[0]

    cpdef inline unsafe_put_int16(self, uint32_t offset, int16_t v):
        cdef uint8_t* arr = self.c_buffer + offset
        arr[0] = <uint8_t>v
        arr[1] = <uint8_t>(v >> 8)

    cpdef inline unsafe_put_int32(self, uint32_t offset, int32_t v):
        cdef uint8_t* arr = self.c_buffer + offset
        arr[0] = <uint8_t>v
        arr[1] = <uint8_t>(v >> 8)
        arr[2] = <uint8_t>(v >> 16)
        arr[3] = <uint8_t>(v >> 24)

    cpdef inline unsafe_put_int64(self, uint32_t offset, int64_t v):
        cdef uint8_t* arr = self.c_buffer + offset
        arr[0] = <uint8_t>v
        arr[1] = <uint8_t>(v >> 8)
        arr[2] = <uint8_t>(v >> 16)
        arr[3] = <uint8_t>(v >> 24)
        arr[4] = <uint8_t>(v >> 32)
        arr[5] = <uint8_t>(v >> 40)
        arr[6] = <uint8_t>(v >> 48)
        arr[7] = <uint8_t>(v >> 56)

    cpdef inline unsafe_put_float32(self, uint32_t offset, float v):
        cdef int32_t x = (<int32_t *>(&v))[0]
        self.unsafe_put_int32(offset, x)

    cpdef inline unsafe_put_float64(self, uint32_t offset, double v):
        cdef int64_t x = (<int64_t *> (&v))[0]
        self.unsafe_put_int64(offset, x)

    cpdef inline int16_t get_int16(self, uint32_t offset):
        self.check_bound(offset, 2)
        cdef uint8_t* arr = self.c_buffer + offset
        cdef int16_t result = arr[0]
        return (result & 0xFF) | (((<int16_t>arr[1]) & 0xFF) << 8)

    cpdef inline int32_t get_int32(self, uint32_t offset):
        self.check_bound(offset, 4)
        cdef uint8_t* arr = self.c_buffer + offset
        cdef int32_t result = arr[0]
        return (result & 0xFF) | (((<int32_t> arr[1]) & 0xFF) << 8) |\
               (((<int32_t> arr[2]) & 0xFF) << 16) | (((<int32_t> arr[3]) & 0xFF) << 24)

    cpdef inline int64_t get_int64(self, uint32_t offset):
        self.check_bound(offset, 8)
        cdef uint8_t* arr = self.c_buffer + offset
        cdef int64_t result = arr[0]
        return (result & 0xFF) | (((<int64_t> arr[1]) & 0xFF) << 8) | \
               (((<int64_t> arr[2]) & 0xFF) << 16) | (((<int64_t> arr[3]) & 0xFF) << 24) | \
               (((<int64_t> arr[4]) & 0xFF) << 32) |(((<int64_t> arr[5]) & 0xFF) << 40) | \
               (((<int64_t> arr[6]) & 0xFF) << 48) |(((<int64_t> arr[7]) & 0xFF) << 56)

    cpdef inline float get_float32(self, uint32_t offset):
        cdef int32_t v = self.get_int32(offset)
        return (<float *>(&v))[0]

    cpdef inline double get_float64(self, uint32_t offset):
        cdef int64_t v = self.get_int64(offset)
        return (<double *> (&v))[0]

    cpdef inline write_int8(self, value):
        self.grow(1)
        self.unsafe_put_int8(self.writer_index, value)
        self.writer_index += 1

    cpdef inline write_int16(self, value):
        self.grow(2)
        self.unsafe_put_int16(self.writer_index, value)
        self.writer_index += 2

    cpdef inline write_int32(self, value):
        self.grow(4)
        self.unsafe_put_int32(self.writer_index, value)
        self.writer_index += 4

    cpdef inline write_int64(self, value):
        self.grow(8)
        self.unsafe_put_int64(self.writer_index, value)
        self.writer_index += 8

    cpdef inline write_float32(self, value):
        self.grow(4)
        self.unsafe_put_float32(self.writer_index, value)
        self.writer_index += 4

    cpdef inline write_float64(self, value):
        self.grow(8)
        self.unsafe_put_float64(self.writer_index, value)
        self.writer_index += 8

    cpdef inline write_varint32(self, int32_t v):
        cdef:
            int32_t value = v
            int32_t offset = self.writer_index
        self.grow(5)
        if value >> 7 == 0:
            self.c_buffer[offset] = <int8_t> value
            self.writer_index += 1
            return 1
        self.c_buffer[offset] = <int8_t> ((value & 0x7F) | 0x80)
        if value >> 14 == 0:
            self.c_buffer[offset + 1] = <int8_t> (value >> 7)
            self.writer_index += 2
            return 2
        self.c_buffer[offset + 1] = <int8_t> (value >> 7 | 0x80)
        if value >> 21 == 0:
            self.c_buffer[offset + 2] = <int8_t> (value >> 14)
            self.writer_index += 3
            return 3
        self.c_buffer[offset + 2] = <int8_t> (value >> 14 | 0x80)
        if value >> 28 == 0:
            self.c_buffer[offset + 3] = <int8_t> (value >> 21)
            self.writer_index += 4
            return 4
        self.c_buffer[offset + 3] = <int8_t> (value >> 21 | 0x80)
        self.c_buffer[offset + 4] = <int8_t> (value >> 28)
        self.writer_index += 5
        return 5

    cpdef inline int32_t read_varint32(self):
        cdef:
            uint32_t read_bytes_length = 1
            int32_t b
            int32_t result
            uint32_t position = self.reader_index
            int8_t * arr = <int8_t *> (self.c_buffer + position)
        if self._size - self.reader_index > 5:
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

    cpdef inline write_flagged_varint32(self, c_bool flag, int32_t v):
        self.grow(5)
        cdef:
            int32_t value = v
            int32_t offset = self.writer_index
            int8_t first
        if flag:
            # Mask first 6 bits, bit 8 is the flag.
            first = (value & 0x3F) | 0x80
        if value >> 6 == 0:
            self.c_buffer[offset] = first
            self.writer_index += 1
            return 1
        if value >> 13 == 0:
            self.c_buffer[offset] = first | 0x40  # Set bit 7.
            self.c_buffer[offset + 1] = <int8_t> (value >> 6)
            self.writer_index += 2
            return 2
        if value >> 20 == 0:
            self.c_buffer[offset] = first | 0x40  # Set bit 7.
            self.c_buffer[offset + 1] = <int8_t> (value >> 6 | 0x80)
            self.c_buffer[offset + 2] = <int8_t> (value >> 13)
            self.writer_index += 3
            return 3
        if value >> 27 == 0:
            self.c_buffer[offset] = first | 0x40  # Set bit 7.
            self.c_buffer[offset + 1] = <int8_t> (value >> 6 | 0x80)
            self.c_buffer[offset + 2] = <int8_t> (value >> 13 | 0x80)
            self.c_buffer[offset + 3] = <int8_t> (value >> 20)
            self.writer_index += 4
            return 4
        self.c_buffer[offset] = first | 0x40  # Set bit 7.
        self.c_buffer[offset + 1] = <int8_t> (value >> 6 | 0x80)
        self.c_buffer[offset + 2] = <int8_t> (value >> 3 | 0x80)
        self.c_buffer[offset + 3] = <int8_t> (value >> 20 | 0x80)
        self.c_buffer[offset + 4] = <int8_t> (value >> 27)
        self.writer_index += 5
        return 5

    cpdef inline c_bool read_varint32_flag(self):
        cdef int8_t head = self.read_int8()
        return (head & 0x80) != 0

    cpdef inline int32_t read_flagged_varint(self):
        cdef:
            uint32_t read_bytes_length = 1
            int32_t b
            int32_t result
            uint32_t position = self.reader_index
            int8_t * arr = <int8_t *> (self.c_buffer + position)
        if self._size - self.reader_index > 5:
            b = arr[0]
            result = b & 0x3F  # Mask first 6 bits.
            if (b & 0x40) != 0:  # Bit 7 means another byte, bit 8 is flag bit.
                read_bytes_length += 1
                b = arr[1]
                result |= (b & 0x7F) << 6
                if (b & 0x80) != 0:
                    read_bytes_length += 1
                    b = arr[2]
                    result |= (b & 0x7F) << 13
                    if (b & 0x80) != 0:
                        read_bytes_length += 1
                        b = arr[3]
                        result |= (b & 0x7F) << 20
                        if (b & 0x80) != 0:
                            read_bytes_length += 1
                            b = arr[4]
                            result |= (b & 0x7F) << 27
            self.reader_index += read_bytes_length
            return result
        else:
            b = self.read_int8()
            result = b & 0x3F  # Mask first 6 bits.
            if (b & 0x40) != 0:
                b = self.read_int8()
                result |= (b & 0x7F) << 6
                if (b & 0x80) != 0:
                    b = self.read_int8()
                    result |= (b & 0x7F) << 13
                    if (b & 0x80) != 0:
                        b = self.read_int8()
                        result |= (b & 0x7F) << 20
                        if (b & 0x80) != 0:
                            b = self.read_int8()
                            result |= (b & 0x7F) << 28
            return result

    cpdef inline write_varint64(self, int64_t v):
        cdef:
            uint64_t value = v
            int64_t offset = self.writer_index
        self.grow(9)
        if value >> 7 == 0:
            self.c_buffer[offset] = <int8_t>value
            self.writer_index += 1
            return 1
        self.c_buffer[offset] = <int8_t> ((value & 0x7F) | 0x80)
        if value >> 14 == 0:
            self.c_buffer[offset+1] = <int8_t>(value >> 7)
            self.writer_index += 2
            return 2
        self.c_buffer[offset + 1] = <int8_t> (value >> 7 | 0x80)
        if value >> 21 == 0:
            self.c_buffer[offset+2] = <int8_t>(value >> 14)
            self.writer_index += 3
            return 3
        self.c_buffer[offset + 2] = <int8_t> (value >> 14 | 0x80)
        if value >> 28 == 0:
            self.c_buffer[offset+3] = <int8_t>(value >> 21)
            self.writer_index += 4
            return 4
        self.c_buffer[offset + 3] = <int8_t> (value >> 21 | 0x80)
        if value >> 35 == 0:
            self.c_buffer[offset+4] = <int8_t>(value >> 28)
            self.writer_index += 5
            return 5
        self.c_buffer[offset + 4] = <int8_t> (value >> 28 | 0x80)
        if value >> 42 == 0:
            self.c_buffer[offset+5] = <int8_t>(value >> 35)
            self.writer_index += 6
            return 6
        self.c_buffer[offset + 5] = <int8_t> (value >> 35 | 0x80)
        if value >> 49 == 0:
            self.c_buffer[offset+6] = <int8_t>(value >> 42)
            self.writer_index += 7
            return 7
        self.c_buffer[offset + 6] = <int8_t> (value >> 42 | 0x80)
        if value >> 56 == 0:
            self.c_buffer[offset+7] = <int8_t>(value >> 49)
            self.writer_index += 8
            return 8
        self.c_buffer[offset + 7] = <int8_t> (value >> 49 | 0x80)
        self.c_buffer[offset + 8] = <int8_t> (value >> 56)
        self.writer_index += 9
        return 9

    cpdef inline int64_t read_varint64(self):
        cdef:
            uint32_t read_bytes_length = 1
            int64_t b
            int64_t result
            uint32_t position = self.reader_index
            int8_t * arr = <int8_t *> (self.c_buffer + position)
        if self._size - self.reader_index > 9:
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
                            if (b & 0x80) != 0:
                                read_bytes_length += 1
                                b = arr[5]
                                result |= (b & 0x7F) << 35
                                if (b & 0x80) != 0:
                                    read_bytes_length += 1
                                    b = arr[6]
                                    result |= (b & 0x7F) << 42
                                    if (b & 0x80) != 0:
                                        read_bytes_length += 1
                                        b = arr[7]
                                        result |= (b & 0x7F) << 49
                                        if (b & 0x80) != 0:
                                            read_bytes_length += 1
                                            b = arr[8]
                                            # highest bit in last byte is symbols bit
                                            result |= b << 56
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
                            if (b & 0x80) != 0:
                                b = self.read_int8()
                                result |= (b & 0x7F) << 35
                                if (b & 0x80) != 0:
                                    b = self.read_int8()
                                    result |= (b & 0x7F) << 42
                                    if (b & 0x80) != 0:
                                        b = self.read_int8()
                                        result |= (b & 0x7F) << 49
                                        if (b & 0x80) != 0:
                                            b = self.read_int8()
                                            # highest bit in last byte is symbols bit
                                            result |= b << 56
            return result

    cpdef inline write_bytes(self, bytes value):
        cdef const unsigned char[:] data = value
        cdef int32_t length = data.nbytes
        self.write_varint32(length)
        if length > 0:
            self.grow(length)
            memcpy(self.c_buffer + self.writer_index, &data[0], length)
            self.writer_index += length

    cpdef inline bytes read_bytes(self):
        cdef int32_t length = self.read_varint32()
        value = self.get_bytes(self.reader_index, length)
        self.reader_index += length
        return value

    cpdef inline bytes get_bytes(self, uint32_t offset, uint32_t nbytes):
        if nbytes == 0:
            return b""
        self.check_bound(offset, nbytes)
        cdef unsigned char* binary_data = self.c_buffer + offset
        return binary_data[:nbytes]

    cdef inline write_binary(self, const uint8_t *data, int32_t length):
        self.write_varint32(length)
        self.grow(length)
        memcpy(self.c_buffer + self.writer_index, data, length)
        self.writer_index += length

    cdef inline int32_t read_binary(self, uint8_t** buf):
        cdef int32_t length = self.read_varint32()
        buf[0] = self.c_buffer + self.reader_index
        self.reader_index += length
        return length

    cpdef inline write_string(self, value):
        cdef Py_ssize_t length
        cdef const char * buf = PyUnicode_AsUTF8AndSize(value, &length)
        self.write_binary(<const uint8_t *>buf, length)

    cpdef inline str read_string(self):
        cdef uint8_t* buf
        cdef int32_t length = self.read_binary(&buf)
        str_obj = PyUnicode_DecodeUTF8(<const char *>buf, length, self.errors)
        return str_obj

    cpdef inline int8_t read_int8(self):
        value = self.get_int8(self.reader_index)
        self.reader_index += 1
        return value

    cpdef inline int16_t read_int16(self):
        value = self.get_int16(self.reader_index)
        self.reader_index += 2
        return value

    cpdef inline int32_t read_int32(self):
        value = self.get_int32(self.reader_index)
        self.reader_index += 4
        return value

    cpdef inline int64_t read_int64(self):
        value = self.get_int64(self.reader_index)
        self.reader_index += 8
        return value

    cpdef inline float read_float32(self):
        value = self.get_float32(self.reader_index)
        self.reader_index += 4
        return value

    cpdef inline double read_float64(self):
        value = self.get_float64(self.reader_index)
        self.reader_index += 8
        return value

    cpdef readline(self, size=None):
        raise NotImplementedError

    # Fast path for common composite types to avoid dynamic methods invoke cost.
    cpdef inline write_int32_list(self, list value):
        self.write_varint32(len(value))
        for item in value:
            self.write_varint32(item)

    cpdef inline list read_int32_list(self):
        length = self.read_varint32()
        cdef list value = []
        for _ in range(length):
            value.append(self.read_varint32())
        return value

    cpdef inline write_int64_list(self, list value):
        self.write_varint32(len(value))
        for item in value:
            self.write_varint64(item)

    cpdef inline list read_int64_list(self):
        length = self.read_varint32()
        value = list()
        for _ in range(length):
            value.append(self.read_varint64())
        return value

    cpdef inline write_float64_list(self, list value):
        self.write_varint32(len(value))
        for item in value:
            self.write_float64(item)

    cpdef inline list read_float64_list(self):
        length = self.read_varint32()
        value = list()
        for _ in range(length):
            value.append(self.read_float64())
        return value

    cpdef inline write_string_list(self, list value):
        self.write_varint32(len(value))
        for item in value:
            self.write_string(item)

    cpdef inline list read_string_list(self):
        length = self.read_varint32()
        value = list()
        for _ in range(length):
            value.append(self.read_string())
        return value

    cpdef inline write_int32_tuple(self, tuple value):
        self.write_varint32(len(value))
        for item in value:
            self.write_varint32(item)

    cpdef inline tuple read_int32_tuple(self):
        return tuple(self.read_int32_list())

    cpdef inline write_string_tuple(self, tuple value):
        self.write_varint32(len(value))
        for item in value:
            self.write_string(item)

    cpdef inline tuple read_string_tuple(self):
        return tuple(self.read_string_list())

    cpdef inline write_string_string_dict(self, dict value):
        self.write_varint32(len(value))
        for k, v in value.items():
            self.write_string(k)
            self.write_string(v)

    cpdef inline dict read_string_string_dict(self):
        length = self.read_varint32()
        cdef dict value = {}
        for _ in range(length):
            k = self.read_string()
            v = self.read_string()
            value[k] = v
        return value

    cpdef inline write_nullable_bool(self, value):
        self.grow(1)
        if value is None:
            (self.c_buffer + self.writer_index)[0] = -1
        else:
            if value:
                (self.c_buffer + self.writer_index)[0] = 1
            else:
                (self.c_buffer + self.writer_index)[0] = 0
        self.writer_index += 1

    cpdef inline read_nullable_bool(self):
        value = self.get_int8(self.reader_index)
        if value == -1:
            return None
        elif value == 0:
            return False
        elif value == 1:
            return True
        else:
            raise Exception(f"Unexpected value: {value}, should be in {{-1,0,1}}")

    cpdef inline write_nullable_varint32(self, value):
        if value is None:
            return self.write_flagged_varint32(False, 0)
        else:
            return self.write_flagged_varint32(True, value)

    cpdef inline read_nullable_varint32(self):
        if self.read_varint32_flag():
            return self.read_flagged_varint()
        else:
            return None

    # Inline nullable type writing to reduce python function invoke cost.
    cpdef inline write_nullable_varint64(self, value):
        if value is None:
            self.write_int8(0)
        else:
            self.write_int8(1)
            return self.write_varint64(value)

    cpdef inline read_nullable_varint64(self):
        if self.read_int8() == 0:
            return None
        else:
            return self.read_varint64()

    cpdef inline write_nullable_bytes(self, value):
        if value is None:
            self.write_int8(0)
        else:
            self.write_int8(1)
            return self.write_bytes(value)

    cpdef inline read_nullable_bytes(self):
        if self.read_int8() == 0:
            return None
        else:
            return self.read_bytes()

    cpdef inline write_nullable_string(self, value):
        if value is None:
            self.write_int8(0)
        else:
            self.write_int8(1)
            return self.write_string(value)

    cpdef inline read_nullable_string(self):
        if self.read_int8() == 0:
            return None
        else:
            return self.read_string()

    cpdef inline write_nullable_float32(self, value):
        if value is None:
            self.write_int8(0)
        else:
            self.write_int8(1)
            return self.write_float32(value)

    cpdef inline read_nullable_float32(self):
        if self.read_int8() == 0:
            return None
        else:
            return self.read_float32()

    cpdef inline write_nullable_float64(self, value):
        if value is None:
            self.write_int8(0)
        else:
            self.write_int8(1)
            return self.write_float64(value)

    cpdef inline read_nullable_float64(self):
        if self.read_int8() == 0:
            return None
        else:
            return self.read_float64()

    def __len__(self):
        return self._size

    cpdef inline size(self):
        return self._size

    def slice(self, offset=0, length=None):
        return type(self)(self, offset, length)

    cpdef bytes to_bytes(self, int32_t offset=0, int32_t length=0):
        if length != 0:
            assert 0 < length <= self._size,\
                f"length {length} size {self._size}"
        else:
            length = self._size
        return (self.c_buffer + offset)[:length]

    cpdef to_pybytes(self):
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


# functools.partial needs to pack and unpack arguments, which is costly.
cdef class Serializer:
    cdef readonly c_bool nullable

    def __init__(self,  c_bool nullable = False):
        self.nullable = nullable

    cpdef write(self, Buffer buffer, value):
        pass

    cpdef read(self, Buffer buffer):
        pass


cdef class PrimitiveSerializer(Serializer):
    cdef:
        readonly object writer
        readonly object reader

    def __init__(self,  writer not None, reader not None):
        super().__init__(False)
        self.writer = writer
        self.reader = reader

    cpdef write(self, Buffer buffer, value):
        self.writer(buffer, value)

    cpdef read(self, Buffer buffer):
        return self.reader(buffer)


cdef class NullableSerializer(Serializer):
    cdef:
        object writer
        object reader

    def __init__(self,  writer not None, reader not None):
        super().__init__(True)
        self.writer = writer
        self.reader = reader

    cpdef write(self, Buffer buffer, value):
        if value is None:
            buffer.write_int8(0)
        else:
            buffer.write_int8(1)
            self.writer(buffer, value)

    cpdef read(self, Buffer buffer):
        if buffer.read_int8() == 0:
            return None
        else:
            return self.reader(buffer)


cdef class StringSerializer(PrimitiveSerializer):
    cpdef write(self, Buffer buffer, value):
        buffer.write_nullable_string(value)

    cpdef read(self, Buffer buffer):
        return buffer.read_nullable_string()


cdef class CollectionSerializer(Serializer):
    cdef:
        readonly Serializer element_serializer
        readonly c_bool element_nullable

    def __init__(self, c_bool nullable, c_bool element_nullable, Serializer element_serializer = None):
        super().__init__(nullable)
        self.nullable = nullable
        self.element_nullable = element_nullable
        self.element_serializer = element_serializer

    cdef inline c_bool write_header(self, Buffer buffer, value):
        if self.nullable:
            if value is None:
                buffer.write_int8(0)
                return True
            else:
                buffer.write_int8(1)
        buffer.write_varint32(len(value))
        return False


cdef class ListSerializer(CollectionSerializer):
    cpdef write(self, Buffer buffer, value):
        if self.write_header(buffer, value):
            return
        element_serializer = self.element_serializer
        for v in value:
            element_serializer.write(buffer, value)

    cpdef read(self, Buffer buffer):
        if self.nullable:
            if buffer.read_int8() == 0:
                return None
        length = buffer.read_varint32()
        cdef list value = []
        element_serializer = self.element_serializer
        for _ in range(length):
            value.append(element_serializer.read(buffer))
        return value


cdef class Int32CollectionSerializer(CollectionSerializer):
    cpdef read(self, Buffer buffer):
        if self.nullable:
            if buffer.read_int8() == 0:
                return None
        length = buffer.read_varint32()
        cdef list value = []
        if self.element_nullable:
            for _ in range(length):
                value.append(buffer.read_nullable_varint32())
        else:
            for _ in range(length):
                value.append(buffer.read_varint32())
        return value

cdef class Int64CollectionSerializer(CollectionSerializer):
    cpdef read(self, Buffer buffer):
        if self.nullable:
            if buffer.read_int8() == 0:
                return None
        length = buffer.read_varint32()
        cdef list value = []
        if self.element_nullable:
            for _ in range(length):
                value.append(buffer.read_nullable_varint64())
        else:
            for _ in range(length):
                value.append(buffer.read_varint64())
        return value


cdef class Float64CollectionSerializer(CollectionSerializer):
    cpdef read(self, Buffer buffer):
        if self.nullable:
            if buffer.read_int8() == 0:
                return None
        length = buffer.read_varint32()
        cdef list value = []
        if self.element_nullable:
            for _ in range(length):
                value.append(buffer.read_nullable_float64())
        else:
            for _ in range(length):
                value.append(buffer.read_float64())
        return value


cdef class StringCollectionSerializer(CollectionSerializer):
    cpdef read(self, Buffer buffer):
        if self.nullable:
            if buffer.read_int8() == 0:
                return None
        length = buffer.read_varint32()
        cdef list value = []
        if self.element_nullable:
            for _ in range(length):
                value.append(buffer.read_nullable_string())
        else:
            for _ in range(length):
                value.append(buffer.read_string())
        return value


cdef class Int32ListSerializer(Int32CollectionSerializer):
    cpdef write(self, Buffer buffer, v):
        if self.write_header(buffer, v):
            return
        cdef list value = v
        if self.element_nullable:
            for item in value:
                buffer.write_nullable_varint32(item)
        else:
            for item in value:
                buffer.write_varint32(item)


cdef class Int64ListSerializer(Int64CollectionSerializer):
    cpdef write(self, Buffer buffer, v):
        if self.write_header(buffer, v):
            return
        cdef list value = v
        if self.element_nullable:
            for item in value:
                buffer.write_nullable_varint64(item)
        else:
            for item in value:
                buffer.write_varint64(item)


cdef class Float64ListSerializer(Float64CollectionSerializer):
    cpdef write(self, Buffer buffer, v):
        if self.write_header(buffer, v):
            return
        cdef list value = v
        if self.element_nullable:
            for item in value:
                buffer.write_nullable_float64(item)
        else:
            for item in value:
                buffer.write_float64(item)


cdef class StringListSerializer(StringCollectionSerializer):
    cpdef write(self, Buffer buffer, v):
        if self.write_header(buffer, v):
            return
        cdef list value = v
        if self.element_nullable:
            for item in value:
                buffer.write_nullable_string(item)
        else:
            for item in value:
                buffer.write_string(item)


cdef class Int32TupleSerializer(Int32CollectionSerializer):
    cpdef write(self, Buffer buffer, v):
        if self.write_header(buffer, v):
            return
        cdef tuple value = v
        if self.element_nullable:
            for item in value:
                buffer.write_nullable_varint32(item)
        else:
            for item in value:
                buffer.write_varint32(item)

    cpdef read(self, Buffer buffer):
        return tuple(super().read(buffer))


cdef class Int64TupleSerializer(Int64CollectionSerializer):
    cpdef write(self, Buffer buffer, v):
        if self.write_header(buffer, v):
            return
        cdef tuple value = v
        if self.element_nullable:
            for item in value:
                buffer.write_nullable_varint64(item)
        else:
            for item in value:
                buffer.write_varint64(item)

    cpdef read(self, Buffer buffer):
        return tuple(super().read(buffer))


cdef class Float64TupleSerializer(Float64CollectionSerializer):
    cpdef write(self, Buffer buffer, v):
        if self.write_header(buffer, v):
            return
        cdef tuple value = v
        if self.element_nullable:
            for item in value:
                buffer.write_nullable_float64(item)
        else:
            for item in value:
                buffer.write_float64(item)

    cpdef read(self, Buffer buffer):
        return tuple(super().read(buffer))


cdef class StringTupleSerializer(StringListSerializer):
    cpdef write(self, Buffer buffer, v):
        if self.write_header(buffer, v):
            return
        cdef tuple value = v
        if self.element_nullable:
            for item in value:
                buffer.write_nullable_string(item)
        else:
            for item in value:
                buffer.write_string(item)

    cpdef read(self, Buffer buffer):
        return tuple(super().read(buffer))


cdef class DictSerializer(Serializer):
    cdef:
        readonly Serializer key_serializer
        readonly Serializer value_serializer
        readonly c_bool key_nullable
        readonly c_bool value_nullable

    def __init__(self, c_bool nullable, c_bool key_nullable, c_bool value_nullable,
                 Serializer key_serializer, Serializer value_serializer):
        super().__init__(nullable)
        self.key_serializer = key_serializer
        self.value_serializer = value_serializer
        self.key_nullable = key_nullable
        self.value_nullable = value_nullable

    cdef inline c_bool write_header(self, Buffer buffer, value):
        if self.nullable:
            if value is None:
                buffer.write_int8(0)
                return True
            else:
                buffer.write_int8(1)
        buffer.write_varint32(len(value))
        return False

    cpdef write(self, Buffer buffer, v):
        if self.write_header(buffer, v):
            return
        cdef dict value = v
        key_serializer = self.key_serializer
        value_serializer = self.value_serializer
        for k, v in value.items():
            key_serializer.write(buffer, k,)
            value_serializer.write(buffer, v)

    cpdef read(self, Buffer buffer):
        if self.nullable:
            if buffer.read_int8() == 0:
                return None
        length = buffer.read_varint32()
        cdef dict value = {}
        key_serializer = self.key_serializer
        value_serializer = self.value_serializer
        for _ in range(length):
            k = key_serializer.read(buffer)
            v = value_serializer.read(buffer)
            value[k] = v
        return value


cdef class StringKVDictSerializer(DictSerializer):
    def __init__(self, c_bool nullable, c_bool key_nullable, c_bool value_nullable):
        super().__init__(nullable, key_nullable, value_nullable,
                         StringSerializer(key_nullable), StringSerializer(value_nullable))

    cpdef write(self, Buffer buffer, v):
        if self.write_header(buffer, v):
            return
        cdef dict value = v
        if not self.key_nullable and not self.value_nullable:
            for k, v in value.items():
                buffer.write_string(k)
                buffer.write_string(v)
        elif not self.key_nullable:
            for k, v in value.items():
                buffer.write_string(k)
                buffer.write_nullable_string(v)
        else:
            for k, v in value.items():
                buffer.write_nullable_string(k)
                buffer.write_nullable_string(v)

    cpdef read(self, Buffer buffer):
        if self.nullable:
            if buffer.read_int8() == 0:
                return None
        length = buffer.read_varint32()
        cdef dict value = {}
        if not self.key_nullable and not self.value_nullable:
            for _ in range(length):
                k = buffer.read_string()
                v = buffer.read_string()
                value[k] = v
        elif not self.key_nullable:
            for k, v in value.items():
                k = buffer.read_string()
                v = buffer.read_nullable_string()
                value[k] = v
        else:
            for k, v in value.items():
                k = buffer.read_nullable_string()
                v = buffer.read_nullable_string()
                value[k] = v
        return value


cdef inline uint8_t* _get_address(v):
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
