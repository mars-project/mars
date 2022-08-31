import pickle
import time

import msgpack
from mars.lib.fury import Buffer


def test_write_primitives():
    buffer = Buffer.allocate(8)
    buffer.write_bool(True)
    print(buffer.get_int8(0))
    buffer.write_int8(-1)
    buffer.write_int8(2**7 - 1)
    buffer.write_int8(-(2**7))
    buffer.write_int16(2**15 - 1)
    buffer.write_int16(-(2**15))
    buffer.write_int32(2**31 - 1)
    buffer.write_int32(-(2**31))
    buffer.write_int64(2**63 - 1)
    buffer.write_int64(-(2**63))
    buffer.write_float32(1.0)
    buffer.write_float32(-1.0)
    buffer.write_float64(1.0)
    buffer.write_float64(-1.0)
    buffer.write_bytes(b"")  # write empty buffer
    binary = b"b" * 100
    buffer.write_bytes(binary)
    new_buffer = Buffer(buffer.get_bytes(0, buffer.writer_index))
    print(buffer.get_int8(0))
    assert new_buffer.read_bool() is True
    assert new_buffer.read_int8() == -1
    assert new_buffer.read_int8() == 2**7 - 1
    assert new_buffer.read_int8() == -(2**7)
    assert new_buffer.read_int16() == 2**15 - 1
    assert new_buffer.read_int16() == -(2**15)
    assert new_buffer.read_int32() == 2**31 - 1
    assert new_buffer.read_int32() == -(2**31)
    assert new_buffer.read_int64() == 2**63 - 1
    assert new_buffer.read_int64() == -(2**63)
    assert new_buffer.read_float32() == 1.0
    assert new_buffer.read_float32() == -1.0
    assert new_buffer.read_float64() == 1.0
    assert new_buffer.read_float64() == -1.0
    assert new_buffer.read_bytes() == b""
    assert new_buffer.read_bytes() == binary
    assert new_buffer.slice(0, 10).to_pybytes() == new_buffer.to_pybytes()[:10]
    assert new_buffer.slice(5, 25).to_pybytes() == new_buffer.to_pybytes()[5:30]
    for i in range(len(new_buffer)):
        v1 = new_buffer.get_uint8(i)
        v2 = new_buffer.to_pybytes()[i]
        assert v1 == v2, f"index {i} {v1} {v2}"


def test_write_varint32():
    buf = Buffer.allocate(32)
    for i in range(32):
        for j in range(i):
            buf.write_int8(1)
            buf.read_int8()
        check_varint32(buf, 1, 1)
        check_varint32(buf, -1, 5)
        check_varint32(buf, 1 << 6, 1)
        check_varint32(buf, 1 << 7, 2)
        check_varint32(buf, -(2**6), 5)
        check_varint32(buf, -(2**7), 5)
        check_varint32(buf, 1 << 13, 2)
        check_varint32(buf, 1 << 14, 3)
        check_varint32(buf, -(2**13), 5)
        check_varint32(buf, -(2**14), 5)
        check_varint32(buf, 1 << 20, 3)
        check_varint32(buf, 1 << 21, 4)
        check_varint32(buf, -(2**20), 5)
        check_varint32(buf, -(2**21), 5)
        check_varint32(buf, 1 << 27, 4)
        check_varint32(buf, 1 << 28, 5)
        check_varint32(buf, -(2**27), 5)
        check_varint32(buf, -(2**28), 5)
        check_varint32(buf, 1 << 30, 5)
        check_varint32(buf, -(2**30), 5)


def check_varint32(buf: Buffer, value: int, bytes_written: int):
    reader_index = buf.reader_index
    assert buf.writer_index == buf.reader_index
    actual_bytes_written = buf.write_varint32(value)
    assert actual_bytes_written == bytes_written
    varint = buf.read_varint32()
    assert buf.writer_index == buf.reader_index
    assert value == varint
    # test slow read branch in `read_varint32`
    assert (
        buf.slice(reader_index, buf.reader_index - reader_index).read_varint32()
        == value
    )


def test_write_varint64():
    buf = Buffer.allocate(32)
    check_varint64(buf, -1, 9)
    for i in range(32):
        for j in range(i):
            buf.write_int8(1)
            buf.read_int8()
        check_varint64(buf, -1, 9)
        check_varint64(buf, 1, 1)
        check_varint64(buf, 1 << 6, 1)
        check_varint64(buf, 1 << 7, 2)
        check_varint64(buf, -(2**6), 9)
        check_varint64(buf, -(2**7), 9)
        check_varint64(buf, 1 << 13, 2)
        check_varint64(buf, 1 << 14, 3)
        check_varint64(buf, -(2**13), 9)
        check_varint64(buf, -(2**14), 9)
        check_varint64(buf, 1 << 20, 3)
        check_varint64(buf, 1 << 21, 4)
        check_varint64(buf, -(2**20), 9)
        check_varint64(buf, -(2**21), 9)
        check_varint64(buf, 1 << 27, 4)
        check_varint64(buf, 1 << 28, 5)
        check_varint64(buf, -(2**27), 9)
        check_varint64(buf, -(2**28), 9)
        check_varint64(buf, 1 << 30, 5)
        check_varint64(buf, -(2**30), 9)
        check_varint64(buf, 1 << 31, 5)
        check_varint64(buf, -(2**31), 9)
        check_varint64(buf, 1 << 32, 5)
        check_varint64(buf, -(2**32), 9)
        check_varint64(buf, 1 << 34, 5)
        check_varint64(buf, -(2**34), 9)
        check_varint64(buf, 1 << 35, 6)
        check_varint64(buf, -(2**35), 9)
        check_varint64(buf, 1 << 41, 6)
        check_varint64(buf, -(2**41), 9)
        check_varint64(buf, 1 << 42, 7)
        check_varint64(buf, -(2**42), 9)
        check_varint64(buf, 1 << 48, 7)
        check_varint64(buf, -(2**48), 9)
        check_varint64(buf, 1 << 49, 8)
        check_varint64(buf, -(2**49), 9)
        check_varint64(buf, 1 << 55, 8)
        check_varint64(buf, -(2**55), 9)
        check_varint64(buf, 1 << 56, 9)
        check_varint64(buf, -(2**56), 9)
        check_varint64(buf, 1 << 62, 9)
        check_varint64(buf, -(2**62), 9)
        check_varint64(buf, 1 << 63 - 1, 9)
        check_varint64(buf, -(2**63), 9)


def check_varint64(buf: Buffer, value: int, bytes_written: int):
    reader_index = buf.reader_index
    assert buf.writer_index == buf.reader_index
    actual_bytes_written = buf.write_varint64(value)
    assert actual_bytes_written == bytes_written
    varint = buf.read_varint64()
    assert buf.writer_index == buf.reader_index
    assert value == varint
    # test slow read branch in `read_varint64`
    assert (
        buf.slice(reader_index, buf.reader_index - reader_index).read_varint64()
        == value
    )


def test_write_str():
    buffer = Buffer.allocate(8)
    buffer.write_string("")
    buffer.write_string("abc")
    buffer.write_string("abc你好")
    assert buffer.read_string() == ""
    assert buffer.read_string() == "abc"
    assert buffer.read_string() == "abc你好"


def test_write_collection():
    buffer = Buffer.allocate(8)
    list1 = [
        1,
        -1,
        2**7 - 1,
        -(2**7),
        2**15 - 1,
        -(2**15),
        2**31 - 1,
        -(2**31),
    ]
    buffer.write_int32_list(list1)
    list2 = [1, -1, 2**31 - 1, -(2**31), 2**63 - 1, -(2**63)]
    buffer.write_int64_list(list2)
    list3 = [1.0, -1.0, 1 / 3, -1 / 3]
    buffer.write_float64_list(list3)
    list4 = ["", "abc", "abc你好"]
    buffer.write_string_list(list4)
    assert buffer.read_int32_list() == list1
    assert buffer.read_int64_list() == list2
    assert buffer.read_float64_list() == list3
    assert buffer.read_string_list() == list4

    buffer.write_int32_tuple(tuple(list1))
    assert buffer.read_int32_tuple() == tuple(list1)
    buffer.write_string_tuple(tuple(list4))
    assert buffer.read_string_tuple() == tuple(list4)


def test_write_dict():
    buffer = Buffer.allocate(8)
    dict1 = {}
    buffer.write_string_string_dict(dict1)
    assert buffer.read_string_string_dict() == dict1
    dict2 = {"": "", "a": "bc", "abc你好": "abc"}
    buffer.write_string_string_dict(dict2)
    assert buffer.read_string_string_dict() == dict2


def test_binary():
    buffer = Buffer.allocate(8)
    buffer.write_bytes(b"")
    buffer.write_bytes(b"123")
    binary = buffer.to_bytes(length=buffer.writer_index)
    assert binary == b"\x00\x03123"
    buffer.point_to_bytes(binary)
    assert buffer.read_bytes() == b""
    assert buffer.read_bytes() == b"123"


def benchmark_write():
    buf = Buffer.allocate(3100000)
    start = time.time_ns()
    values = [
        # (2 ** 7 - 1,  Buffer.write_varint32),
        (
            [
                1,
                -1,
                2**7 - 1,
                -(2**7),
                2**15 - 1,
                -(2**15),
                2**31 - 1,
                -(2**31),
            ],
            Buffer.write_int32_list,
        ),
        # (list(range(100)), Buffer.write_int64_list),
        # (["", "abc", "abc你好"], Buffer.write_string_list)
    ]
    for v, func in values:
        fury_size = None
        for i in range(1000):
            for _ in range(10000):
                buf.writer_index = 0
                func(buf, v)
            fury_size = buf.writer_index
        print(
            f"Fury Serialize {v} cost {(time.time_ns() - start)/1000_000} fury_size {fury_size}"
        )
        import io

        start = time.time_ns()
        pickle_size = None
        for i in range(1000):
            packer = msgpack.Packer(autoreset=False)
            for _ in range(10000):
                packer.reset()
                packer.pack(v)
                pickle_size = len(packer.bytes())
        print(
            f"msgpack Serialize {v} cost {(time.time_ns() - start)/1000_000} {pickle_size}"
        )

        start = time.time_ns()
        pickle_size = None
        for i in range(1000):
            file = io.BytesIO()
            pickler = pickle.Pickler(file)
            for _ in range(10000):
                pickler.clear_memo()
                pickler.dump(v)
                pickle_size = file.tell()
        print(
            f"Pickle Serialize {v} cost {(time.time_ns() - start)/1000_000} {pickle_size}"
        )


if __name__ == "__main__":
    benchmark_write()
