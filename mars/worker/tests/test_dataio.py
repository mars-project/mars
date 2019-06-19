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

import unittest

import numpy as np
try:
    import pyarrow
except ImportError:
    pyarrow = None

from mars.compat import BytesIO
from mars.serialize import dataserializer
from mars.worker.dataio import ArrowBufferIO, FileBufferIO


class Test(unittest.TestCase):
    @unittest.skipIf(pyarrow is None, 'PyArrow is not installed.')
    def testArrowBufferIO(self):
        if not np:
            return
        import pyarrow
        from numpy.testing import assert_array_equal

        for compress in [dataserializer.CompressType.LZ4, dataserializer.CompressType.GZIP]:
            if compress not in dataserializer.get_supported_compressions():
                continue

            data = np.random.random((1000, 100))
            serialized = pyarrow.serialize(data).to_buffer()

            # test complete read
            reader = ArrowBufferIO(
                pyarrow.py_buffer(serialized), 'r', compress_out=compress)
            assert_array_equal(data, dataserializer.loads(reader.read()))

            # test partial read
            reader = ArrowBufferIO(
                pyarrow.py_buffer(serialized), 'r', compress_out=compress)
            block = reader.read(128)
            data_left = reader.read()
            assert_array_equal(data, dataserializer.loads(block + data_left))

            # test read by chunks
            bio = BytesIO()
            reader = ArrowBufferIO(
                pyarrow.py_buffer(serialized), 'r', compress_out=compress)
            while True:
                block = reader.read(128)
                if not block:
                    break
                bio.write(block)

            compressed = bio.getvalue()
            assert_array_equal(data, dataserializer.loads(compressed))

            # test write by chunks
            data_sink = bytearray(len(serialized))
            compressed_mv = memoryview(compressed)
            writer = ArrowBufferIO(pyarrow.py_buffer(data_sink), 'w')
            pos = 0
            while pos < len(compressed):
                endpos = min(pos + 128, len(compressed))
                writer.write(compressed_mv[pos:endpos])
                pos = endpos

            assert_array_equal(data, pyarrow.deserialize(data_sink))

    def testFileBufferIO(self):
        if not np:
            return
        from numpy.testing import assert_array_equal

        compressions = [dataserializer.CompressType.NONE] + \
            list(dataserializer.get_supported_compressions())

        for c1 in compressions:
            for c2 in compressions:
                data = np.random.random((1000, 100))

                # test complete read
                compressed_read_file = BytesIO(dataserializer.dumps(data, compress=c1))
                reader = FileBufferIO(compressed_read_file, 'r', compress_out=c2)
                compressed = reader.read()
                self.assertEqual(c2, dataserializer.read_file_header(compressed).compress)
                assert_array_equal(data, dataserializer.loads(compressed))

                # test partial read
                compressed_read_file = BytesIO(dataserializer.dumps(data, compress=c1))
                reader = FileBufferIO(compressed_read_file, 'r', compress_out=c2)
                block = reader.read(128)
                data_left = reader.read()
                assert_array_equal(data, dataserializer.loads(block + data_left))

                # test read by chunks
                bio = BytesIO()
                compressed_read_file = BytesIO(dataserializer.dumps(data, compress=c1))
                reader = FileBufferIO(compressed_read_file, 'r', compress_out=c2)
                while True:
                    block = reader.read(128)
                    if not block:
                        break
                    bio.write(block)

                compressed = bio.getvalue()
                self.assertEqual(c2, dataserializer.read_file_header(compressed).compress)
                assert_array_equal(data, dataserializer.loads(compressed))

                # test write by chunks
                compressed_read_file.seek(0)
                compressed_write_file = BytesIO()
                writer = FileBufferIO(compressed_write_file, 'w', compress_in=c2,
                                      managed=False)
                while True:
                    block = compressed_read_file.read(128)
                    if not block:
                        break
                    writer.write(block)
                writer.close()

                compressed = compressed_write_file.getvalue()
                self.assertEqual(c2, dataserializer.read_file_header(compressed).compress)
                assert_array_equal(data, dataserializer.loads(compressed))
