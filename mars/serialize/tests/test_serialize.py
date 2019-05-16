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

import datetime
import json
import os
import tempfile
import unittest

import numpy as np
try:
    import pyarrow
except ImportError:
    pyarrow = None
try:
    import pandas as pd
except ImportError:
    pd = None

from mars.compat import six, OrderedDict, BytesIO
from mars.lib import sparse
from mars.serialize.core import Serializable, IdentityField, StringField, UnicodeField, \
    BytesField, Int8Field, Int16Field, Int32Field, Int64Field, UInt8Field, UInt16Field, \
    UInt32Field, UInt64Field, Float16Field, Float32Field, Float64Field, BoolField, \
    Datetime64Field, Timedelta64Field, DataTypeField, KeyField, ReferenceField, OneOfField, \
    ListField, NDArrayField, DictField, TupleField, ValueType, serializes, deserializes, \
    IndexField, SeriesField, DataFrameField, SliceField, ProviderType, AttributeAsDict
from mars.serialize import dataserializer
from mars.serialize.pbserializer import ProtobufSerializeProvider
from mars.serialize.jsonserializer import JsonSerializeProvider
from mars.core import Base
from mars.utils import to_binary, to_text


class Node1(Serializable):
    a = IdentityField('a', ValueType.string)
    b1 = Int8Field('b1')
    b2 = Int16Field('b2')
    b3 = Int32Field('b3')
    b4 = Int64Field('b4')
    c1 = UInt8Field('c1')
    c2 = UInt16Field('c2')
    c3 = UInt32Field('c3')
    c4 = UInt64Field('c4')
    d1 = Float16Field('d1')
    d2 = Float32Field('d2')
    d3 = Float64Field('d3')
    e = BoolField('e')
    f = KeyField('f')
    g = ReferenceField('g', 'Node2')
    h = ListField('h')
    i = ListField('i', ValueType.reference('self'))

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


class Node2(Base, Serializable):
    a = ListField('a', ValueType.list(ValueType.string))
    _key = StringField('key')
    _id = StringField('id')
    _name = UnicodeField('name')
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
    b = SliceField('b')


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
    c = Datetime64Field('d')
    d = Timedelta64Field('e')
    e = DataTypeField('f')
    f = DictField('g', ValueType.string, ValueType.list(ValueType.bool))
    g = DictField('h')
    h = TupleField('i', ValueType.int64, ValueType.unicode, ValueType.string, ValueType.float32,
                   ValueType.datetime64, ValueType.timedelta64, ValueType.dtype)
    i = TupleField('j', ValueType.slice)
    j = ReferenceField('k', Node5)
    k = ListField('l', ValueType.reference('Node5'))
    l = OneOfField('m', n5=Node5, n6=Node6)
    m = IndexField('n')
    mm = IndexField('mn')
    n = SeriesField('o')
    o = DataFrameField('p')
    p = ListField('q')

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
        node1 = Node1(a='test1',
                      b1=-2, b2=2000, b3=-5000, b4=500000,
                      c1=2, c2=2000, c3=5000, c4=500000,
                      d1=2.5, d2=7.37, d3=5.976321,
                      e=False,
                      f=node2,
                      g=Node2(a=[['1', '2'], ['3', '4']]),
                      h=[[2, 3], node2, True, {1: node2}, np.datetime64('1066-10-13'), np.timedelta64(1, 'D')],
                      i=[Node1(b1=111), Node1(b1=222)])
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
        self.assertEqual(node3.value.b1, d_node3.value.b1)
        self.assertEqual(node3.value.b2, d_node3.value.b2)
        self.assertEqual(node3.value.b3, d_node3.value.b3)
        self.assertEqual(node3.value.b4, d_node3.value.b4)
        self.assertEqual(node3.value.c1, d_node3.value.c1)
        self.assertEqual(node3.value.c2, d_node3.value.c2)
        self.assertEqual(node3.value.c3, d_node3.value.c3)
        self.assertEqual(node3.value.c4, d_node3.value.c4)
        self.assertAlmostEqual(node3.value.d1, d_node3.value.d1, places=2)
        self.assertAlmostEqual(node3.value.d2, d_node3.value.d2, places=4)
        self.assertAlmostEqual(node3.value.d3, d_node3.value.d3)
        self.assertEqual(node3.value.e, d_node3.value.e)
        self.assertIsNot(node3.value.f, d_node3.value.f)
        self.assertEqual(node3.value.f.a, d_node3.value.f.a)
        self.assertIsNot(node3.value.g, d_node3.value.g)
        self.assertEqual(node3.value.g.a, d_node3.value.g.a)
        self.assertEqual(node3.value.h[0], d_node3.value.h[0])
        self.assertNotIsInstance(d_node3.value.h[1], six.string_types)
        self.assertIs(d_node3.value.h[1], d_node3.value.f)
        self.assertEqual(node3.value.h[2], True)
        self.assertEqual([n.b1 for n in node3.value.i], [n.b1 for n in d_node3.value.i])
        self.assertNotIsInstance(node3.value.i[0], Node8)

    def testJSONSerialize(self):
        provider = JsonSerializeProvider()

        node2 = Node2(a=[['ss'], ['dd']], data=[3, 7, 212])
        node1 = Node1(a='test1',
                      b1=2, b2=2000, b3=5000, b4=500000,
                      c1=2, c2=2000, c3=5000, c4=500000,
                      d1=2.5, d2=7.37, d3=5.976321,
                      e=False,
                      f=node2,
                      g=Node2(a=[['1', '2'], ['3', '4']]),
                      h=[[2, 3], node2, True, {1: node2}, np.datetime64('1066-10-13'), np.timedelta64(1, 'D')],
                      i=[Node1(b1=111), Node1(b1=222)])
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
        self.assertEqual(node3.value.b1, d_node3.value.b1)
        self.assertEqual(node3.value.b2, d_node3.value.b2)
        self.assertEqual(node3.value.b3, d_node3.value.b3)
        self.assertEqual(node3.value.b4, d_node3.value.b4)
        self.assertEqual(node3.value.c1, d_node3.value.c1)
        self.assertEqual(node3.value.c2, d_node3.value.c2)
        self.assertEqual(node3.value.c3, d_node3.value.c3)
        self.assertEqual(node3.value.c4, d_node3.value.c4)
        self.assertAlmostEqual(node3.value.d1, d_node3.value.d1, places=2)
        self.assertAlmostEqual(node3.value.d2, d_node3.value.d2, places=4)
        self.assertAlmostEqual(node3.value.d3, d_node3.value.d3)
        self.assertEqual(node3.value.e, d_node3.value.e)
        self.assertIsNot(node3.value.f, d_node3.value.f)
        self.assertEqual(node3.value.f.a, d_node3.value.f.a)
        self.assertIsNot(node3.value.g, d_node3.value.g)
        self.assertEqual(node3.value.g.a, d_node3.value.g.a)
        self.assertEqual(node3.value.h[0], d_node3.value.h[0])
        self.assertNotIsInstance(d_node3.value.h[1], six.string_types)
        self.assertIs(d_node3.value.h[1], d_node3.value.f)
        self.assertEqual(node3.value.h[2], True)
        self.assertEqual([n.b1 for n in node3.value.i], [n.b1 for n in d_node3.value.i])
        self.assertNotIsInstance(node3.value.i[0], Node8)

    def testAttributeAsDict(self):
        other_data = {}
        if pd:
            df = pd.DataFrame({'a': [1, 2, 3], 'b': [to_text('测试'), to_binary('属性'), 'c']},
                              index=[[0, 0, 1], ['测试', '属性', '测试']])
            other_data['m'] = df.columns
            other_data['mm'] = df.index
            other_data['n'] = df['b']
            other_data['o'] = df
            other_data['p'] = [df.columns, df.index, df['a'], df]
        node4 = Node4(a=to_binary('中文'),
                      b=np.random.randint(4, size=(3, 4)),
                      c=np.datetime64(datetime.datetime.now()),
                      d=np.timedelta64(datetime.timedelta(seconds=1234)),
                      e=np.dtype('int'),
                      f={'a': [True, False, False], 'd': [False, None]},
                      h=(1234, to_text('测试'), '属性', None, np.datetime64('1066-10-13'),
                         np.timedelta64(1, 'D'), np.dtype([('x', 'i4'), ('y', 'f4')])),
                      i=(slice(10), slice(0, 2), None, slice(2, 0, -1)),
                      j=Node5(a='aa', b=slice(1, 100, 3)),
                      k=[Node5(a='bb', b=slice(200, -1, -4)), None],
                      l=Node6(b=3, nid=1), **other_data)

        pbs = ProtobufSerializeProvider()

        serial = node4.serialize(pbs)
        d_node4 = Node4.deserialize(pbs, serial)

        self.assertEqual(node4.a, d_node4.a)
        np.testing.assert_array_equal(node4.b, d_node4.b)
        self.assertEqual(node4.c, d_node4.c)
        self.assertEqual(node4.d, d_node4.d)
        self.assertEqual(node4.e, d_node4.e)
        self.assertEqual(node4.f, d_node4.f)
        self.assertFalse(hasattr(d_node4, 'g'))
        self.assertEqual(node4.h, d_node4.h)
        self.assertEqual(node4.i, d_node4.i)
        self.assertEqual(node4.j.a, d_node4.j.a)
        self.assertEqual(node4.j.b, d_node4.j.b)
        self.assertEqual(node4.k[0].a, d_node4.k[0].a)
        self.assertEqual(node4.k[0].b, d_node4.k[0].b)
        self.assertIsNone(d_node4.k[1])
        self.assertIsInstance(d_node4.l, Node7)
        self.assertEqual(d_node4.l.b, 3)
        if pd:
            pd.testing.assert_index_equal(node4.m, d_node4.m)
            pd.testing.assert_index_equal(node4.mm, d_node4.mm)
            pd.testing.assert_series_equal(node4.n, d_node4.n)
            pd.testing.assert_frame_equal(node4.o, d_node4.o)
            pd.testing.assert_index_equal(node4.p[0], d_node4.p[0])
            pd.testing.assert_index_equal(node4.p[1], d_node4.p[1])
            pd.testing.assert_series_equal(node4.p[2], d_node4.p[2])
            pd.testing.assert_frame_equal(node4.p[3], d_node4.p[3])

        jss = JsonSerializeProvider()

        serial = node4.serialize(jss)
        serial = json.loads(json.dumps(serial), object_hook=OrderedDict)
        d_node4 = Node4.deserialize(jss, serial)

        self.assertEqual(node4.a, d_node4.a)
        np.testing.assert_array_equal(node4.b, d_node4.b)
        self.assertEqual(node4.c, d_node4.c)
        self.assertEqual(node4.d, d_node4.d)
        self.assertEqual(node4.e, d_node4.e)
        self.assertEqual(node4.f, d_node4.f)
        self.assertFalse(hasattr(d_node4, 'g'))
        self.assertEqual(node4.h, d_node4.h)
        self.assertEqual(node4.i, d_node4.i)
        self.assertEqual(node4.j.a, d_node4.j.a)
        self.assertEqual(node4.k[0].a, d_node4.k[0].a)
        self.assertIsNone(d_node4.k[1])
        self.assertIsInstance(d_node4.l, Node7)
        self.assertEqual(d_node4.l.b, 3)
        if pd:
            pd.testing.assert_index_equal(node4.m, d_node4.m)
            pd.testing.assert_index_equal(node4.mm, d_node4.mm)
            pd.testing.assert_series_equal(node4.n, d_node4.n)
            pd.testing.assert_frame_equal(node4.o, d_node4.o)
            pd.testing.assert_index_equal(node4.p[0], d_node4.p[0])
            pd.testing.assert_index_equal(node4.p[1], d_node4.p[1])
            pd.testing.assert_series_equal(node4.p[2], d_node4.p[2])
            pd.testing.assert_frame_equal(node4.p[3], d_node4.p[3])

    def testException(self):
        node1 = Node1(h=[object()])

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
            if not six.PY2:
                assert_array_equal(array, dataserializer.loads(dataserializer.dumps(
                    array, compress=dataserializer.COMPRESS_FLAG_GZIP)))

            array = np.random.rand(1000, 100)
            assert_array_equal(array, dataserializer.load(BytesIO(dataserializer.dumps(array))))
            assert_array_equal(array, dataserializer.load(BytesIO(dataserializer.dumps(
                array, compress=dataserializer.COMPRESS_FLAG_LZ4))))
            if not six.PY2:
                assert_array_equal(array, dataserializer.load(BytesIO(dataserializer.dumps(
                    array, compress=dataserializer.COMPRESS_FLAG_GZIP))))

            array = np.random.rand(1000, 100).T  # test non c-contiguous
            assert_array_equal(array, dataserializer.loads(dataserializer.dumps(array)))
            assert_array_equal(array, dataserializer.loads(dataserializer.dumps(
                array, compress=dataserializer.COMPRESS_FLAG_LZ4)))
            if not six.PY2:
                assert_array_equal(array, dataserializer.loads(dataserializer.dumps(
                    array, compress=dataserializer.COMPRESS_FLAG_GZIP)))

            array = np.float64(0.2345)
            assert_array_equal(array, dataserializer.loads(dataserializer.dumps(array)))
            assert_array_equal(array, dataserializer.loads(dataserializer.dumps(
                array, compress=dataserializer.COMPRESS_FLAG_LZ4)))
            if not six.PY2:
                assert_array_equal(array, dataserializer.loads(dataserializer.dumps(
                    array, compress=dataserializer.COMPRESS_FLAG_GZIP)))

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

                if not six.PY2:
                    with open(fn, 'wb') as dump_file:
                        dataserializer.dump(array, dump_file,
                                            compress=dataserializer.COMPRESS_FLAG_GZIP)
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

            if not six.PY2:
                des_mat = dataserializer.loads(dataserializer.dumps(
                    mat, compress=dataserializer.COMPRESS_FLAG_GZIP))
                self.assertTrue((mat.spmatrix != des_mat.spmatrix).nnz == 0)

            vector = sparse.SparseVector(sps.csr_matrix(np.random.rand(2)), shape=(2,))
            des_vector = dataserializer.loads(dataserializer.dumps(vector))
            self.assertTrue((vector.spmatrix != des_vector.spmatrix).nnz == 0)

            des_vector = dataserializer.loads(dataserializer.dumps(
                vector, compress=dataserializer.COMPRESS_FLAG_LZ4))
            self.assertTrue((vector.spmatrix != des_vector.spmatrix).nnz == 0)

            if not six.PY2:
                des_vector = dataserializer.loads(dataserializer.dumps(
                    vector, compress=dataserializer.COMPRESS_FLAG_GZIP))
                self.assertTrue((vector.spmatrix != des_vector.spmatrix).nnz == 0)

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

        from mars.serialize.dataserializer import mars_serialize_context
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
            tp = (array, mat)
            des_tp = pyarrow.deserialize(pyarrow.serialize(tp, context).to_buffer(), context)
            assert_array_equal(tp[0], des_tp[0])
            self.assertTrue((tp[1].spmatrix != des_tp[1].spmatrix).nnz == 0)

    @unittest.skipIf(pyarrow is None, 'PyArrow is not installed.')
    def testCompressIO(self):
        if not np:
            return
        import pyarrow
        from numpy.testing import assert_array_equal

        for compress in [dataserializer.COMPRESS_FLAG_LZ4, dataserializer.COMPRESS_FLAG_GZIP]:
            if compress not in dataserializer.get_supported_compressions():
                continue

            data = np.random.random((1000, 100))
            serialized = pyarrow.serialize(data).to_buffer()

            bio = BytesIO()
            reader = dataserializer.CompressBufferReader(pyarrow.py_buffer(serialized), compress)
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
