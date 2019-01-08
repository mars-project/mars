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


from .core import ValueType, Serializable, SerializableMetaclass, AttributeAsDict, \
    SerializableWithKey, AttributeAsDictKey, \
    serializes, deserializes, ProviderType, Provider, \
    AnyField, IdentityField, BoolField, Int8Field, Int16Field, Int32Field, Int64Field, \
    UInt8Field, UInt16Field, UInt32Field, UInt64Field, Float16Field, Float32Field, Float64Field, \
    StringField, BytesField, UnicodeField, KeyField, NDArrayField, DataTypeField, \
    ListField, TupleField, DictField, ReferenceField, OneOfField
from .jsonserializer import JsonSerializeProvider
try:
    from .pbserializer import ProtobufSerializeProvider
except ImportError:  # pragma: no cover
    # ProtobufSerializeProvider used in distributed environment
    pass
