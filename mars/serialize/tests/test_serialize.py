#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import datetime
import itertools
import json
import os
import tempfile
import unittest
from collections import OrderedDict
from io import BytesIO

import numpy as np
import pytz
import pandas as pd
from numpy.testing import assert_array_equal

from mars.lib import sparse
from mars.lib.groupby_wrapper import wrapped_groupby
from mars.serialize.core import Serializable, IdentityField, StringField, UnicodeField, \
    BytesField, Int8Field, Int16Field, Int32Field, Int64Field, UInt8Field, UInt16Field, \
    UInt32Field, UInt64Field, Float16Field, Float32Field, Float64Field, BoolField, \
    Datetime64Field, Timedelta64Field, DataTypeField, KeyField, ReferenceField, OneOfField, \
    ListField, NDArrayField, DictField, TupleField, ValueType, serializes, deserializes, \
    IndexField, SeriesField, DataFrameField, SliceField, Complex64Field, Complex128Field, \
    AnyField, FunctionField, TZInfoField, IntervalArrayField, ProviderType, AttributeAsDict
from mars.serialize import dataserializer
from mars.serialize.pbserializer import ProtobufSerializeProvider
from mars.serialize.jsonserializer import JsonSerializeProvider
from mars.core import Base, Entity
from mars.errors import SerializationFailed
from mars.tests.core import assert_groupby_equal
from mars.utils import to_binary, to_text

try:
    import pyarrow
except ImportError:
    pyarrow = None
try:
    import scipy.sparse as sps
except ImportError:
    sps = None


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
    cl1 = Complex64Field('cl1')
    cl2 = Complex128Field('cl2')
    e = BoolField('e')
    f1 = KeyField('f1')
    f2 = AnyField('f2')
    g = ReferenceField('g', 'Node2')
    h = ListField('h')
    i = ListField('i', ValueType.reference('self'))
    j = ReferenceField('j', None)
    k = ListField('k', ValueType.reference(None))
    l = FunctionField('l')  # noqa: E741
    m = TZInfoField('m')
    n = IntervalArrayField('n')

    def __new__(cls, *args, **kwargs):
        if 'a' in kwargs and kwargs['a'] == 'test1':
            return object.__new__(Node8)
        return object.__new__(cls)

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from mars.serialize.tests.testser_pb2 import Node1Def
            return Node1Def
        return super().cls(provider)


class Node8(Node1):
    pass


class Node9(Node1):
    f1 = AnyField('f1')


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
        return super().cls(provider)


class Node2Entity(Entity):
    __slots__ = ()
    _allow_data_type_ = (Node2,)


class Node3(Serializable):
    value = OneOfField('value', n1='Node1', n2='Node2')

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from mars.serialize.tests.testser_pb2 import Node3Def
            return Node3Def
        return super().cls(provider)


class Node5(AttributeAsDict):
    a = StringField('a')
    b = SliceField('b')


class Node6(AttributeAsDict):
    nid = IdentityField('id', ValueType.int64)
    b = Int32Field('b')
    r = ReferenceField('r', 'self')
    rl = ListField('rl', ValueType.reference('self'))

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
    l = OneOfField('m', n5=Node5, n6=Node6)  # noqa: E741
    m = ReferenceField('n', None)
    n = ListField('o', ValueType.reference(None))
    w = IndexField('v')
    ww = IndexField('vw')
    x = SeriesField('w')
    y = DataFrameField('x')
    z = ListField('y')
    o = FunctionField('p')

    @classmethod
    def cls(cls, provider):
        if provider.type == ProviderType.protobuf:
            from mars.serialize.tests.testser_pb2 import Node4Def
            return Node4Def
        return super().cls(provider)


class Test(unittest.TestCase):
    def testPBSerialize(self):
        provider = ProtobufSerializeProvider()

        node2 = Node2(a=[['ss'], ['dd']], data=[3, 7, 212])
        node1 = Node1(a='test1',
                      b1=-2, b2=2000, b3=-5000, b4=500000,
                      c1=2, c2=2000, c3=5000, c4=500000,
                      d1=2.5, d2=7.37, d3=5.976321,
                      cl1=1+2j, cl2=2.5+3.1j,
                      e=False,
                      f1=Node2Entity(node2),
                      f2=Node2Entity(node2),
                      g=Node2(a=[['1', '2'], ['3', '4']]),
                      h=[[2, 3], node2, True, {1: node2}, np.datetime64('1066-10-13'),
                         np.timedelta64(1, 'D'), np.complex64(1+2j), np.complex128(2+3j),
                         lambda x: x + 2, pytz.timezone('Asia/Shanghai'),
                         pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])],
                      i=[Node8(b1=111), Node8(b1=222)],
                      j=Node2(a=[['u'], ['v']]),
                      k=[Node5(a='uvw'), Node8(b1=222, j=Node5(a='xyz')), None],
                      l=lambda x: x + 1,
                      m=pytz.timezone('Asia/Shanghai'),
                      n=pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)]))
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
        self.assertAlmostEqual(node3.value.cl1, d_node3.value.cl1)
        self.assertAlmostEqual(node3.value.cl2, d_node3.value.cl2)
        self.assertEqual(node3.value.e, d_node3.value.e)
        self.assertIsNot(node3.value.f1, d_node3.value.f1)
        self.assertEqual(node3.value.f1.a, d_node3.value.f1.a)
        self.assertIsNot(node3.value.f2, d_node3.value.f2)
        self.assertEqual(node3.value.f2.a, d_node3.value.f2.a)
        self.assertIsNot(node3.value.g, d_node3.value.g)
        self.assertEqual(node3.value.g.a, d_node3.value.g.a)
        self.assertEqual(node3.value.h[0], d_node3.value.h[0])
        self.assertNotIsInstance(d_node3.value.h[1], str)
        self.assertIs(d_node3.value.h[1], d_node3.value.f1)
        self.assertEqual(node3.value.h[2], True)
        self.assertAlmostEqual(node3.value.h[6], d_node3.value.h[6])
        self.assertAlmostEqual(node3.value.h[7], d_node3.value.h[7])
        self.assertEqual(node3.value.h[8](2), 4)
        self.assertEqual(node3.value.h[9], d_node3.value.h[9])
        np.testing.assert_array_equal(node3.value.h[10], d_node3.value.h[10])
        self.assertEqual([n.b1 for n in node3.value.i], [n.b1 for n in d_node3.value.i])
        self.assertIsInstance(d_node3.value.i[0], Node8)
        self.assertIsInstance(d_node3.value.j, Node2)
        self.assertEqual(node3.value.j.a, d_node3.value.j.a)
        self.assertIsInstance(d_node3.value.k[0], Node5)
        self.assertEqual(node3.value.k[0].a, d_node3.value.k[0].a)
        self.assertIsInstance(d_node3.value.k[1], Node8)
        self.assertEqual(node3.value.k[1].b1, d_node3.value.k[1].b1)
        self.assertIsInstance(d_node3.value.k[1].j, Node5)
        self.assertEqual(node3.value.k[1].j.a, d_node3.value.k[1].j.a)
        self.assertIsNone(node3.value.k[2])
        self.assertEqual(d_node3.value.l(1), 2)
        self.assertEqual(d_node3.value.m, node3.value.m)
        np.testing.assert_array_equal(d_node3.value.n, node3.value.n)

        with self.assertRaises(ValueError):
            serializes(provider, [Node3(value='sth else')])

    def testJSONSerialize(self):
        provider = JsonSerializeProvider()

        node2 = Node2(a=[['ss'], ['dd']], data=[3, 7, 212])
        node1 = Node1(a='test1',
                      b1=2, b2=2000, b3=5000, b4=500000,
                      c1=2, c2=2000, c3=5000, c4=500000,
                      d1=2.5, d2=7.37, d3=5.976321,
                      cl1=1+2j, cl2=2.5+3.1j,
                      e=False,
                      f1=Node2Entity(node2),
                      f2=Node2Entity(node2),
                      g=Node2(a=[['1', '2'], ['3', '4']]),
                      h=[[2, 3], node2, True, {1: node2}, np.datetime64('1066-10-13'),
                         np.timedelta64(1, 'D'), np.complex64(1+2j), np.complex128(2+3j),
                         lambda x: x + 2, pytz.timezone('Asia/Shanghai'),
                         pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])],
                      i=[Node8(b1=111), Node8(b1=222)],
                      j=Node2(a=[['u'], ['v']]),
                      k=[Node5(a='uvw'), Node8(b1=222, j=Node5(a='xyz')), None],
                      l=lambda x: x + 1,
                      m=pytz.timezone('Asia/Shanghai'),
                      n=pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)]))
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
        self.assertAlmostEqual(node3.value.cl1, d_node3.value.cl1)
        self.assertAlmostEqual(node3.value.cl2, d_node3.value.cl2)
        self.assertEqual(node3.value.e, d_node3.value.e)
        self.assertIsNot(node3.value.f1, d_node3.value.f1)
        self.assertEqual(node3.value.f1.a, d_node3.value.f1.a)
        self.assertIsNot(node3.value.f2, d_node3.value.f2)
        self.assertEqual(node3.value.f2.a, d_node3.value.f2.a)
        self.assertIsNot(node3.value.g, d_node3.value.g)
        self.assertEqual(node3.value.g.a, d_node3.value.g.a)
        self.assertEqual(node3.value.h[0], d_node3.value.h[0])
        self.assertNotIsInstance(d_node3.value.h[1], str)
        self.assertIs(d_node3.value.h[1], d_node3.value.f1)
        self.assertEqual(node3.value.h[2], True)
        self.assertAlmostEqual(node3.value.h[6], d_node3.value.h[6])
        self.assertAlmostEqual(node3.value.h[7], d_node3.value.h[7])
        self.assertEqual(node3.value.h[8](2), 4)
        self.assertEqual(node3.value.h[9], d_node3.value.h[9])
        np.testing.assert_array_equal(node3.value.h[10], d_node3.value.h[10])
        self.assertEqual([n.b1 for n in node3.value.i], [n.b1 for n in d_node3.value.i])
        self.assertIsInstance(d_node3.value.i[0], Node8)
        self.assertIsInstance(d_node3.value.j, Node2)
        self.assertEqual(node3.value.j.a, d_node3.value.j.a)
        self.assertIsInstance(d_node3.value.k[0], Node5)
        self.assertEqual(node3.value.k[0].a, d_node3.value.k[0].a)
        self.assertIsInstance(d_node3.value.k[1], Node8)
        self.assertEqual(node3.value.k[1].b1, d_node3.value.k[1].b1)
        self.assertIsInstance(d_node3.value.k[1].j, Node5)
        self.assertEqual(node3.value.k[1].j.a, d_node3.value.k[1].j.a)
        self.assertIsNone(node3.value.k[2])
        self.assertEqual(d_node3.value.l(1), 2)
        self.assertEqual(d_node3.value.m, node3.value.m)
        np.testing.assert_array_equal(d_node3.value.n, node3.value.n)

        with self.assertRaises(ValueError):
            serializes(provider, [Node3(value='sth else')])

    def testNumpyDtypePBSerialize(self):
        provider = ProtobufSerializeProvider()

        node9 = Node9(b1=np.int8(-2), b2=np.int16(2000), b3=np.int32(-5000), b4=np.int64(500000),
                      c1=np.uint8(2), c2=np.uint16(2000), c3=np.uint32(5000), c4=np.uint64(500000),
                      d1=np.float16(2.5), d2=np.float32(7.37), d3=np.float64(5.976321),
                      f1=np.int8(3))

        serials = serializes(provider, [node9])
        d_node9, = deserializes(provider, [Node9], serials)

        self.assertIsNot(node9, d_node9)
        self.assertEqual(node9.b1, d_node9.b1)
        self.assertEqual(node9.b2, d_node9.b2)
        self.assertEqual(node9.b3, d_node9.b3)
        self.assertEqual(node9.b4, d_node9.b4)
        self.assertEqual(node9.c1, d_node9.c1)
        self.assertEqual(node9.c2, d_node9.c2)
        self.assertEqual(node9.c3, d_node9.c3)
        self.assertEqual(node9.c4, d_node9.c4)
        self.assertAlmostEqual(node9.d1, d_node9.d1, places=2)
        self.assertAlmostEqual(node9.d2, d_node9.d2, places=4)
        self.assertAlmostEqual(node9.d3, d_node9.d3)
        self.assertEqual(node9.f1, d_node9.f1)

        node_rec1 = Node9(f1=np.dtype([('label', 'int32'),
                                       ('s0', '<U16'), ('s1', 'int32'), ('s2', 'int32'),
                                       ('d0', '<U16'), ('d1', 'int32'), ('d2', 'int32'), ('d3', '<U256')]))
        node_rec2 = Node9(f1=np.dtype([('label', 'int32'),
                                       ('s0', '<U16'), ('s1', 'int32'), ('s2', 'int32'), ('s3', '<U256'),
                                       ('d0', '<U16'), ('d1', 'int32'), ('d2', 'int32'), ('d3', '<U256')]))

        serials = serializes(provider, [node_rec1])
        d_node_rec1, = deserializes(provider, [Node9], serials)

        self.assertIsNot(node_rec1, d_node_rec1)
        self.assertEqual(node_rec1.f1, d_node_rec1.f1)

        serials = serializes(provider, [node_rec2])
        d_node_rec2, = deserializes(provider, [Node9], serials)

        self.assertIsNot(node_rec2, d_node_rec2)
        self.assertEqual(node_rec2.f1, d_node_rec2.f1)

    def testNumpyDtypeJSONSerialize(self):
        provider = JsonSerializeProvider()

        node9 = Node9(b1=np.int8(-2), b2=np.int16(2000), b3=np.int32(-5000), b4=np.int64(500000),
                      c1=np.uint8(2), c2=np.uint16(2000), c3=np.uint32(5000), c4=np.uint64(500000),
                      d1=np.float16(2.5), d2=np.float32(7.37), d3=np.float64(5.976321),
                      f1=np.int8(3))

        serials = serializes(provider, [node9])
        d_node9, = deserializes(provider, [Node9], serials)

        self.assertIsNot(node9, d_node9)
        self.assertEqual(node9.b1, d_node9.b1)
        self.assertEqual(node9.b2, d_node9.b2)
        self.assertEqual(node9.b3, d_node9.b3)
        self.assertEqual(node9.b4, d_node9.b4)
        self.assertEqual(node9.c1, d_node9.c1)
        self.assertEqual(node9.c2, d_node9.c2)
        self.assertEqual(node9.c3, d_node9.c3)
        self.assertEqual(node9.c4, d_node9.c4)
        self.assertAlmostEqual(node9.d1, d_node9.d1, places=2)
        self.assertAlmostEqual(node9.d2, d_node9.d2, places=4)
        self.assertAlmostEqual(node9.d3, d_node9.d3)
        self.assertEqual(node9.f1, d_node9.f1)

        node_rec1 = Node9(f1=np.dtype([('label', 'int32'),
                                       ('s0', '<U16'), ('s1', 'int32'), ('s2', 'int32'),
                                       ('d0', '<U16'), ('d1', 'int32'), ('d2', 'int32'), ('d3', '<U256')]))
        node_rec2 = Node9(f1=np.dtype([('label', 'int32'),
                                       ('s0', '<U16'), ('s1', 'int32'), ('s2', 'int32'), ('s3', '<U256'),
                                       ('d0', '<U16'), ('d1', 'int32'), ('d2', 'int32'), ('d3', '<U256')]))

        serials = serializes(provider, [node_rec1])
        d_node_rec1, = deserializes(provider, [Node9], serials)

        self.assertIsNot(node_rec1, d_node_rec1)
        self.assertEqual(node_rec1.f1, d_node_rec1.f1)

        serials = serializes(provider, [node_rec2])
        d_node_rec2, = deserializes(provider, [Node9], serials)

        self.assertIsNot(node_rec2, d_node_rec2)
        self.assertEqual(node_rec2.f1, d_node_rec2.f1)

    def testAttributeAsDict(self):
        other_data = {}
        if pd:
            df = pd.DataFrame({'a': [1, 2, 3], 'b': [to_text('测试'), to_binary('属性'), 'c']},
                              index=[[0, 0, 1], ['测试', '属性', '测试']])
            other_data['w'] = df.columns
            other_data['ww'] = df.index
            other_data['x'] = df['b']
            other_data['y'] = df
            other_data['z'] = [df.columns, df.index, df['a'], df]
        node4 = Node4(a=to_binary('中文'),
                      b=np.random.randint(4, size=(3, 4)),
                      c=np.datetime64(datetime.datetime.now()),
                      d=np.timedelta64(datetime.timedelta(seconds=1234)),
                      e=np.dtype('int'),
                      f={'a': [True, False, False], 'd': [False, None]},
                      h=(1234, to_text('测试'), '属性', None, np.datetime64('1066-10-13'),
                         np.timedelta64(1, 'D'), np.dtype([('x', 'i4'), ('y', 'f4')])),
                      i=(slice(10), slice(0, 2), None, slice(2, 0, -1),
                         slice('a', 'b'), slice(datetime.datetime.now(), datetime.datetime.now())),
                      j=Node5(a='aa', b=slice(1, 100, 3)),
                      k=[Node5(a='bb', b=slice(200, -1, -4)), None],
                      l=Node6(b=3, nid=1),
                      m=Node6(b=4, nid=2),
                      n=[Node5(a='cc', b=slice(100, -2, -5)), None],
                      **other_data)

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
        self.assertEqual(node4.l.b, d_node4.l.b)
        self.assertIsInstance(d_node4.m, Node7)
        self.assertEqual(node4.m.b, d_node4.m.b)
        self.assertIsInstance(d_node4.n[0], Node5)
        self.assertEqual(node4.n[0].a, d_node4.n[0].a)
        self.assertEqual(node4.n[0].b, d_node4.n[0].b)
        self.assertIsNone(d_node4.n[1])
        if pd:
            pd.testing.assert_index_equal(node4.w, d_node4.w)
            pd.testing.assert_index_equal(node4.ww, d_node4.ww)
            pd.testing.assert_series_equal(node4.x, d_node4.x)
            pd.testing.assert_frame_equal(node4.y, d_node4.y)
            pd.testing.assert_index_equal(node4.z[0], d_node4.z[0])
            pd.testing.assert_index_equal(node4.z[1], d_node4.z[1])
            pd.testing.assert_series_equal(node4.z[2], d_node4.z[2])
            pd.testing.assert_frame_equal(node4.z[3], d_node4.z[3])

        with self.assertRaises(TypeError):
            node42 = Node4(j=Node6())
            node42.serialize(pbs)

        with self.assertRaises(TypeError):
            node6 = Node6(nid=0)
            node7 = Node7(nid=1, r=node6)
            node7.serialize(pbs)

        with self.assertRaises(TypeError):
            node6 = Node6(nid=0)
            node7 = Node7(nid=1, rl=[node6])
            node7.serialize(pbs)

        node61 = Node6(nid=0)
        node62 = Node6(nid=0, r=node61)
        serial = node62.serialize(pbs)
        d_node62 = Node6.deserialize(pbs, serial)
        self.assertIsInstance(d_node62.r, Node6)

        node61 = Node6(nid=0)
        node62 = Node6(nid=0, rl=[node61])
        serial = node62.serialize(pbs)
        d_node62 = Node6.deserialize(pbs, serial)
        self.assertIsInstance(d_node62.rl[0], Node6)

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
        self.assertEqual(node4.l.b, d_node4.l.b)
        self.assertIsInstance(d_node4.m, Node7)
        self.assertEqual(node4.m.b, d_node4.m.b)
        self.assertIsInstance(d_node4.n[0], Node5)
        self.assertEqual(node4.n[0].a, d_node4.n[0].a)
        self.assertEqual(node4.n[0].b, d_node4.n[0].b)
        self.assertIsNone(d_node4.n[1])
        if pd:
            pd.testing.assert_index_equal(node4.w, d_node4.w)
            pd.testing.assert_index_equal(node4.ww, d_node4.ww)
            pd.testing.assert_series_equal(node4.x, d_node4.x)
            pd.testing.assert_frame_equal(node4.y, d_node4.y)
            pd.testing.assert_index_equal(node4.z[0], d_node4.z[0])
            pd.testing.assert_index_equal(node4.z[1], d_node4.z[1])
            pd.testing.assert_series_equal(node4.z[2], d_node4.z[2])
            pd.testing.assert_frame_equal(node4.z[3], d_node4.z[3])

        with self.assertRaises(TypeError):
            node42 = Node4(j=Node6())
            node42.serialize(jss)

        with self.assertRaises(TypeError):
            node6 = Node6()
            node7 = Node7(r=node6)
            node7.serialize(jss)

        with self.assertRaises(TypeError):
            node6 = Node6(nid=0)
            node7 = Node7(nid=1, rl=[node6])
            node7.serialize(jss)

        node61 = Node6()
        node62 = Node6(r=node61)
        serial = node62.serialize(jss)
        d_node62 = Node6.deserialize(jss, serial)
        self.assertIsInstance(d_node62.r, Node6)

        node61 = Node6(nid=0)
        node62 = Node6(nid=0, rl=[node61])
        serial = node62.serialize(jss)
        d_node62 = Node6.deserialize(jss, serial)
        self.assertIsInstance(d_node62.rl[0], Node6)

    def testException(self):
        node1 = Node1(h=[object()])

        pbs = ProtobufSerializeProvider()

        with self.assertRaises(TypeError):
            node1.serialize(pbs)

        jss = JsonSerializeProvider()

        with self.assertRaises(TypeError):
            node1.serialize(jss)

    def testDataSerialize(self):
        for type_, compress in itertools.product(
                (None,) + tuple(dataserializer.SerialType.__members__.values()),
                (None,) + tuple(dataserializer.CompressType.__members__.values())):
            array = np.random.rand(1000, 100)
            assert_array_equal(array, dataserializer.loads(
                dataserializer.dumps(array, serial_type=type_, compress=compress)))

            array = np.random.rand(1000, 100)
            assert_array_equal(array, dataserializer.load(
                BytesIO(dataserializer.dumps(array, serial_type=type_, compress=compress))))

            array = np.random.rand(1000, 100).T  # test non c-contiguous
            assert_array_equal(array, dataserializer.loads(
                dataserializer.dumps(array, serial_type=type_, compress=compress)))

            array = np.float64(0.2345)
            assert_array_equal(array, dataserializer.loads(
                dataserializer.dumps(array, serial_type=type_, compress=compress)))

        # test non-serializable object
        if pyarrow:
            non_serial = type('non_serial', (object,), dict(nbytes=10))
            with self.assertRaises(SerializationFailed):
                dataserializer.dumps(non_serial())

        # test structured arrays.
        rec_dtype = np.dtype([('a', 'int64'), ('b', 'double'), ('c', '<U8')])
        array = np.ones((100,), dtype=rec_dtype)
        array_loaded = dataserializer.loads(dataserializer.dumps(array))
        self.assertEqual(array.dtype, array_loaded.dtype)
        assert_array_equal(array, array_loaded)

        fn = os.path.join(tempfile.gettempdir(), f'test_dump_file_{id(self)}.bin')
        try:
            array = np.random.rand(1000, 100).T  # test non c-contiguous
            with open(fn, 'wb') as dump_file:
                dataserializer.dump(array, dump_file)
            with open(fn, 'rb') as dump_file:
                assert_array_equal(array, dataserializer.load(dump_file))

            with open(fn, 'wb') as dump_file:
                dataserializer.dump(array, dump_file,
                                    compress=dataserializer.CompressType.LZ4)
            with open(fn, 'rb') as dump_file:
                assert_array_equal(array, dataserializer.load(dump_file))

            with open(fn, 'wb') as dump_file:
                dataserializer.dump(array, dump_file,
                                    compress=dataserializer.CompressType.GZIP)
            with open(fn, 'rb') as dump_file:
                assert_array_equal(array, dataserializer.load(dump_file))
        finally:
            if os.path.exists(fn):
                os.unlink(fn)

        # test sparse
        if sps:
            mat = sparse.SparseMatrix(sps.random(100, 100, 0.1, format='csr'))
            des_mat = dataserializer.loads(dataserializer.dumps(mat))
            self.assertTrue((mat.spmatrix != des_mat.spmatrix).nnz == 0)

            des_mat = dataserializer.loads(dataserializer.dumps(
                mat, compress=dataserializer.CompressType.LZ4))
            self.assertTrue((mat.spmatrix != des_mat.spmatrix).nnz == 0)

            des_mat = dataserializer.loads(dataserializer.dumps(
                mat, compress=dataserializer.CompressType.GZIP))
            self.assertTrue((mat.spmatrix != des_mat.spmatrix).nnz == 0)

            vector = sparse.SparseVector(sps.csr_matrix(np.random.rand(2)), shape=(2,))
            des_vector = dataserializer.loads(dataserializer.dumps(vector))
            self.assertTrue((vector.spmatrix != des_vector.spmatrix).nnz == 0)

            des_vector = dataserializer.loads(dataserializer.dumps(
                vector, compress=dataserializer.CompressType.LZ4))
            self.assertTrue((vector.spmatrix != des_vector.spmatrix).nnz == 0)

            des_vector = dataserializer.loads(dataserializer.dumps(
                vector, compress=dataserializer.CompressType.GZIP))
            self.assertTrue((vector.spmatrix != des_vector.spmatrix).nnz == 0)

        # test groupby
        df1 = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                            'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                            'c': list('aabaaddce')})
        grouped = wrapped_groupby(df1, 'b')
        restored = dataserializer.loads(dataserializer.dumps(grouped))
        assert_groupby_equal(grouped, restored.groupby_obj)

        grouped = wrapped_groupby(df1, 'b').c
        restored = dataserializer.loads(dataserializer.dumps(grouped))
        assert_groupby_equal(grouped, restored.groupby_obj)

        grouped = wrapped_groupby(df1, 'b')
        getattr(grouped, 'indices')
        restored = dataserializer.loads(dataserializer.dumps(grouped))
        assert_groupby_equal(grouped, restored.groupby_obj)

        grouped = wrapped_groupby(df1.b, lambda x: x % 2)
        restored = dataserializer.loads(dataserializer.dumps(grouped))
        assert_groupby_equal(grouped, restored.groupby_obj)

        grouped = wrapped_groupby(df1.b, lambda x: x % 2)
        getattr(grouped, 'indices')
        restored = dataserializer.loads(dataserializer.dumps(grouped))
        assert_groupby_equal(grouped, restored.groupby_obj)

        # test categorical
        s = np.random.RandomState(0).random(10)
        cat = pd.cut(s, [0.3, 0.5, 0.8])
        self.assertIsInstance(cat, pd.Categorical)
        des_cat = dataserializer.loads(dataserializer.dumps(cat))
        self.assertEqual(len(cat), len(des_cat))
        for c, dc in zip(cat, des_cat):
            np.testing.assert_equal(c, dc)

        # test IntervalIndex
        s = pd.interval_range(10, 100, 3)
        dest_s = dataserializer.loads((dataserializer.dumps(s)))
        pd.testing.assert_index_equal(s, dest_s)

        # test complex
        s = complex(10 + 5j)
        dest_s = dataserializer.loads((dataserializer.dumps(s)))
        self.assertIs(type(s), type(dest_s))
        self.assertEqual(s, dest_s)

        s = np.complex64(10 + 5j)
        dest_s = dataserializer.loads((dataserializer.dumps(s)))
        self.assertIs(type(s), type(dest_s))
        self.assertEqual(s, dest_s)

        # test DataFrame with SparseDtype
        s = pd.Series([1, 2, np.nan, np.nan, 3]).astype(
            pd.SparseDtype(np.dtype(np.float64), np.nan))
        dest_s = dataserializer.loads((dataserializer.dumps(s)))
        pd.testing.assert_series_equal(s, dest_s)
        df = pd.DataFrame({'s': s})
        dest_df = dataserializer.loads((dataserializer.dumps(df)))
        pd.testing.assert_frame_equal(df, dest_df)

    @unittest.skipIf(pyarrow is None, 'PyArrow is not installed.')
    def testArrowSerialize(self):
        array = np.random.rand(1000, 100)
        assert_array_equal(array, dataserializer.deserialize(dataserializer.serialize(array).to_buffer()))

        if sps:
            mat = sparse.SparseMatrix(sps.random(100, 100, 0.1, format='csr'))
            des_mat = dataserializer.deserialize(dataserializer.serialize(mat).to_buffer())
            self.assertTrue((mat.spmatrix != des_mat.spmatrix).nnz == 0)

            array = np.random.rand(1000, 100)
            mat = sparse.SparseMatrix(sps.random(100, 100, 0.1, format='csr'))
            tp = (array, mat)
            des_tp = dataserializer.deserialize(dataserializer.serialize(tp).to_buffer())
            assert_array_equal(tp[0], des_tp[0])
            self.assertTrue((tp[1].spmatrix != des_tp[1].spmatrix).nnz == 0)
