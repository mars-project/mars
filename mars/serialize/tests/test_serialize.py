#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import json
import os
import tempfile

import numpy as np

try:
    import pyarrow
except ImportError:
    pyarrow = None

from mars.compat import unittest, six, OrderedDict, BytesIO
from mars.lib import sparse
from mars.serialize.core import Serializable, IdentityField, StringField, Int32Field, BytesField, \
    KeyField, ReferenceField, OneOfField, ListField, NDArrayField, DictField, TupleField, \
    ValueType, serializes, deserializes, ProviderType, AttributeAsDict
from mars.serialize import dataserializer
from mars.serialize.pbserializer import ProtobufSerializeProvider
from mars.serialize.jsonserializer import JsonSerializeProvider
from mars.core import BaseWithKey
from mars.utils import to_binary, to_text


class Node1(Serializable):
    a = IdentityField('a', ValueType.string)
    b = Int32Field('b')
    c = KeyField('c')
    d = ReferenceField('d', 'Node2')
    e = ListField('e')
    f = ListField('f', ValueType.reference('self'))

    def __new__(cls, *args, **kwargs):
        if 'a' in kwargs and kwargs['a'] == 'test1':
            return object.__new__(Node8)
        return object.__new__(cls)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from mars.serialize.tests.testser_pb2 import Node1Def
            return Node1Def
        return super(Node1, cls).cls(provider)


class Node8(Node1):
    pass


class Node2(Serializable, BaseWithKey):
    a = ListField('a', ValueType.list(ValueType.string))
    _key = StringField('key')
    _id = StringField('id')
    data = ListField('data', ValueType.int32)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from mars.serialize.tests.testser_pb2 import Node2Def
            return Node2Def
        return super(Node2, cls).cls(provider)


class Node3(Serializable):
    value = OneOfField('value', n1='Node1', n2='Node2')

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from mars.serialize.tests.testser_pb2 import Node3Def
            return Node3Def
        return super(Node3, cls).cls(provider)


class Node5(AttributeAsDict):
    a = StringField('a')


class Node6(AttributeAsDict):
    nid = IdentityField('id', ValueType.int64)
    b = Int32Field('b')

    def __new__(cls, *args, **kwargs):
        if 'nid' in kwargs and kwargs['nid'] != 0:
            return object.__new__(Node7)
        return object.__new__(cls)


class Node7(Node6):
    pass


class Node4(AttributeAsDict):
    attr_tag = 'attr'

    a = BytesField('b')
    b = NDArrayField('c')
    c = DictField('d', ValueType.string, ValueType.list(ValueType.bool))
    d = DictField('e')
    e = TupleField('f', ValueType.int64, ValueType.unicode, ValueType.string, ValueType.float32,
                   ValueType.datetime64, ValueType.timedelta64, ValueType.dtype)
    f = TupleField('g', ValueType.slice)
    g = ReferenceField('h', Node5)
    h = ListField('i', ValueType.reference('Node5'))
    i = OneOfField('j', n5=Node5, n6=Node6)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from mars.serialize.tests.testser_pb2 import Node4Def
            return Node4Def
        return super(Node4, cls).cls(provider)


class Test(unittest.TestCase):
    def testPBSerialize(self):
        provider = ProtobufSerializeProvider()

        node2 = Node2(a=[['ss'], ['dd']], data=[3, 7, 212])
        node1 = Node1(a='test1', b=2, d=Node2(a=[['1', '2'], ['3', '4']]),
                      c=node2,
                      e=[[2, 3], node2, True, {1: node2}, np.datetime64('1066-10-13'), np.timedelta64(1, 'D')],
                      f=[Node1(b=111), Node1(b=222)])
        node3 = Node3(value=node1)

        serials = serializes(provider, [node2, node3])
        d_node2, d_node3 = deserializes(provider, [Node2, Node3], serials)

        self.assertIsNot(node2, d_node2)
        self.assertEqual(node2.a, d_node2.a)
        self.assertEqual(node2.data, d_node2.data)

        self.assertIsNot(node3, d_node3)
        self.assertIsInstance(d_node3.value, Node8)
        self.assertIsNot(node3.value, d_node3.value)
        self.assertEqual(node3.value.a, d_node3.value.a)
        self.assertEqual(node3.value.b, d_node3.value.b)
        self.assertIsNot(node3.value.c, d_node3.value.c)
        self.assertEqual(node3.value.c.a, d_node3.value.c.a)
        self.assertIsNot(node3.value.d, d_node3.value.d)
        self.assertEqual(node3.value.d.a, d_node3.value.d.a)
        self.assertEqual(node3.value.e[0], d_node3.value.e[0])
        self.assertNotIsInstance(d_node3.value.e[1], six.string_types)
        self.assertIs(d_node3.value.e[1], d_node3.value.c)
        self.assertEqual(node3.value.e[2], True)
        self.assertEqual([n.b for n in node3.value.f], [n.b for n in d_node3.value.f])
        self.assertNotIsInstance(node3.value.f[0], Node8)

    def testJSONSerialize(self):
        provider = JsonSerializeProvider()

        node2 = Node2(a=[['ss'], ['dd']], data=[3, 7, 212])
        node1 = Node1(a='test1', b=2, d=Node2(a=[['1', '2'], ['3', '4']]),
                      c=node2,
                      e=[[2, 3], node2, True, {1: node2}, np.datetime64('1066-10-13'), np.timedelta64(1, 'D')],
                      f=[Node1(b=111), Node1(b=222)])
        node3 = Node3(value=node1)

        serials = serializes(provider, [node2, node3])
        serials = [json.loads(json.dumps(s), object_hook=OrderedDict) for s in serials]
        d_node2, d_node3 = deserializes(provider, [Node2, Node3], serials)

        self.assertIsNot(node2, d_node2)
        self.assertEqual(node2.a, d_node2.a)
        self.assertEqual(node2.data, d_node2.data)

        self.assertIsNot(node3, d_node3)
        self.assertIsInstance(d_node3.value, Node8)
        self.assertIsNot(node3.value, d_node3.value)
        self.assertEqual(node3.value.a, d_node3.value.a)
        self.assertEqual(node3.value.b, d_node3.value.b)
        self.assertIsNot(node3.value.c, d_node3.value.c)
        self.assertEqual(node3.value.c.a, d_node3.value.c.a)
        self.assertIsNot(node3.value.d, d_node3.value.d)
        self.assertEqual(node3.value.d.a, d_node3.value.d.a)
        self.assertEqual(node3.value.e[0], d_node3.value.e[0])
        self.assertNotIsInstance(d_node3.value.e[1], six.string_types)
        self.assertIs(d_node3.value.e[1], d_node3.value.c)
        self.assertEqual(node3.value.e[2], True)
        self.assertEqual([n.b for n in node3.value.f], [n.b for n in d_node3.value.f])
        self.assertNotIsInstance(node3.value.f[0], Node8)

    def testAttributeAsDict(self):
        node4 = Node4(a=to_binary('中文'), b=np.random.randint(4, size=(3, 4)),
                      c={'a': [True, False, False], 'd': [False, None]},
                      e=(1234, to_text('测试'), '属性', None, np.datetime64('1066-10-13'),
                         np.timedelta64(1, 'D'), np.dtype([('x', 'i4'), ('y', 'f4')])),
                      f=(slice(10), slice(0, 2), None, slice(2, 0, -1)),
                      g=Node5(a='aa'),
                      h=[Node5(a='bb'), None],
                      i=Node6(b=3, nid=1))

        pbs = ProtobufSerializeProvider()

        serial = node4.serialize(pbs)
        d_node4 = Node4.deserialize(pbs, serial)

        self.assertEqual(node4.a, d_node4.a)
        self.assertTrue(np.array_equal(node4.b, d_node4.b))
        self.assertEqual(node4.c, d_node4.c)
        self.assertFalse(hasattr(d_node4, 'd'))
        self.assertEqual(node4.e, d_node4.e)
        self.assertEqual(node4.f, d_node4.f)
        self.assertEqual(node4.g.a, d_node4.g.a)
        self.assertEqual(node4.h[0].a, d_node4.h[0].a)
        self.assertIsNone(d_node4.h[1])
        self.assertIsInstance(d_node4.i, Node7)
        self.assertEqual(d_node4.i.b, 3)

        jss = JsonSerializeProvider()

        serial = node4.serialize(jss)
        serial = json.loads(json.dumps(serial), object_hook=OrderedDict)
        d_node4 = Node4.deserialize(jss, serial)

        self.assertEqual(node4.a, d_node4.a)
        self.assertTrue(np.array_equal(node4.b, d_node4.b))
        self.assertEqual(node4.c, d_node4.c)
        self.assertFalse(hasattr(d_node4, 'd'))
        self.assertEqual(node4.e, d_node4.e)
        self.assertEqual(node4.f, d_node4.f)
        self.assertEqual(node4.g.a, d_node4.g.a)
        self.assertEqual(node4.h[0].a, d_node4.h[0].a)
        self.assertIsNone(d_node4.h[1])
        self.assertIsInstance(d_node4.i, Node7)
        self.assertEqual(d_node4.i.b, 3)

    def testException(self):
        node1 = Node1(e=[object()])

        pbs = ProtobufSerializeProvider()

        with self.assertRaises(TypeError):
            node1.serialize(pbs)

        jss = JsonSerializeProvider()

        with self.assertRaises(TypeError):
            node1.serialize(jss)

    def testDataSerialize(self):
        try:
            import numpy as np
            from numpy.testing import assert_array_equal
        except ImportError:
            np = None

        try:
            import scipy.sparse as sps
        except ImportError:
            sps = None

        if np:
            array = np.random.rand(1000, 100)
            assert_array_equal(array, dataserializer.loads(dataserializer.dumps(array)))
            assert_array_equal(array, dataserializer.loads(dataserializer.dumps(
                array, compress=dataserializer.COMPRESS_FLAG_LZ4)))

            array = np.random.rand(1000, 100)
            assert_array_equal(array, dataserializer.load(BytesIO(dataserializer.dumps(array))))
            assert_array_equal(array, dataserializer.load(BytesIO(dataserializer.dumps(
                array, compress=dataserializer.COMPRESS_FLAG_LZ4))))

            array = np.random.rand(1000, 100).T  # test non c-contiguous
            assert_array_equal(array, dataserializer.loads(dataserializer.dumps(array)))
            assert_array_equal(array, dataserializer.loads(dataserializer.dumps(
                array, compress=dataserializer.COMPRESS_FLAG_LZ4)))

            array = np.float64(0.2345)
            assert_array_equal(array, dataserializer.loads(dataserializer.dumps(array)))
            assert_array_equal(array, dataserializer.loads(dataserializer.dumps(
                array, compress=dataserializer.COMPRESS_FLAG_LZ4)))

            fn = os.path.join(tempfile.gettempdir(), 'test_dump_file_%d.bin' % id(self))
            try:
                array = np.random.rand(1000, 100).T  # test non c-contiguous
                with open(fn, 'wb') as dump_file:
                    dataserializer.dump(array, dump_file)
                with open(fn, 'rb') as dump_file:
                    assert_array_equal(array, dataserializer.load(dump_file))
                with open(fn, 'wb') as dump_file:
                    dataserializer.dump(array, dump_file,
                                        compress=dataserializer.COMPRESS_FLAG_LZ4)
                with open(fn, 'rb') as dump_file:
                    assert_array_equal(array, dataserializer.load(dump_file))
            finally:
                if os.path.exists(fn):
                    os.unlink(fn)

        if sps:
            mat = sparse.SparseMatrix(sps.random(100, 100, 0.1, format='csr'))
            des_mat = dataserializer.loads(dataserializer.dumps(mat))
            self.assertTrue((mat.spmatrix != des_mat.spmatrix).nnz == 0)

            des_mat = dataserializer.loads(dataserializer.dumps(
                mat, compress=dataserializer.COMPRESS_FLAG_LZ4))
            self.assertTrue((mat.spmatrix != des_mat.spmatrix).nnz == 0)

    @unittest.skipIf(pyarrow is None, 'PyArrow is not installed.')
    def testArrowSerialize(self):
        try:
            import numpy as np
            from numpy.testing import assert_array_equal
        except ImportError:
            np = None

        try:
            import scipy.sparse as sps
        except ImportError:
            sps = None

        from mars.serialize.dataserializer import DataTuple, mars_serialize_context
        context = mars_serialize_context()

        if np:
            array = np.random.rand(1000, 100)
            assert_array_equal(array, pyarrow.deserialize(pyarrow.serialize(array, context).to_buffer(), context))

        if sps:
            mat = sparse.SparseMatrix(sps.random(100, 100, 0.1, format='csr'))
            des_mat = pyarrow.deserialize(pyarrow.serialize(mat, context).to_buffer(), context)
            self.assertTrue((mat.spmatrix != des_mat.spmatrix).nnz == 0)

        if np and sps:
            array = np.random.rand(1000, 100)
            mat = sparse.SparseMatrix(sps.random(100, 100, 0.1, format='csr'))
            tp = DataTuple((array, mat))
            des_tp = pyarrow.deserialize(pyarrow.serialize(tp, context).to_buffer(), context)
            assert_array_equal(tp[0], des_tp[0])
            self.assertTrue((tp[1].spmatrix != des_tp[1].spmatrix).nnz == 0)

    @unittest.skipIf(pyarrow is None, 'PyArrow is not installed.')
    def testCompressIO(self):
        if not np:
            return
        import pyarrow
        from numpy.testing import assert_array_equal

        data = np.random.random((1000, 100))
        serialized = pyarrow.serialize(data).to_buffer()

        bio = BytesIO()
        reader = dataserializer.CompressBufferReader(pyarrow.py_buffer(serialized),
                                                     dataserializer.COMPRESS_FLAG_LZ4)
        while True:
            block = reader.read(128)
            if not block:
                break
            bio.write(block)

        compressed = bio.getvalue()
        assert_array_equal(data, dataserializer.loads(compressed))

        data_sink = bytearray(len(serialized))
        compressed_mv = memoryview(compressed)
        writer = dataserializer.DecompressBufferWriter(pyarrow.py_buffer(data_sink))
        pos = 0
        while pos < len(compressed):
            endpos = min(pos + 128, len(compressed))
            writer.write(compressed_mv[pos:endpos])
            pos = endpos

        assert_array_equal(data, pyarrow.deserialize(data_sink))
