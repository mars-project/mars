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

import base64
import pickle
import weakref

import numpy as np
cimport numpy as np
from cpython.version cimport PY_MAJOR_VERSION
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from ..compat import six, OrderedDict, izip
from .._utils cimport to_str
from .core cimport Provider, ValueType, ProviderType, \
    Field, List, Tuple, Dict, Identity, Reference, KeyPlaceholder, \
    ReferenceField, OneOfField, ListField
from .core import HasKey
from .dataserializer import dumps as datadumps, loads as dataloads


cdef dict PRIMITIVE_TYPE_TO_NAME = {
    ValueType.bool: 'bool',
    ValueType.int8: 'int8',
    ValueType.int16: 'int16',
    ValueType.int32: 'int32',
    ValueType.int64: 'int64',
    ValueType.uint8: 'uint8',
    ValueType.uint16: 'uint16',
    ValueType.uint32: 'uint32',
    ValueType.uint64: 'uint64',
    ValueType.float16: 'float16',
    ValueType.float32: 'float32',
    ValueType.float64: 'float64',
    ValueType.bytes: 'bytes',
    ValueType.unicode: 'unicode',
}

cdef dict EXTEND_TYPE_TO_NAME = {
    ValueType.slice: 'slice',
    ValueType.arr: 'arr',
    ValueType.dtype: 'dtype',
    ValueType.index: 'index',
    ValueType.series: 'series',
    ValueType.dataframe: 'dataframe',
    ValueType.key: 'key',
    ValueType.datetime64: 'datetime64',
    ValueType.timedelta64: 'timedelta64',
    ValueType.complex64: 'complex64',
    ValueType.complex128: 'complex128',
}


cdef inline str _get_name(value_type):
    if value_type in PRIMITIVE_TYPE_TO_NAME:
        return PRIMITIVE_TYPE_TO_NAME[value_type]
    if value_type in EXTEND_TYPE_TO_NAME:
        return EXTEND_TYPE_TO_NAME[value_type]
    return value_type.name


cdef inline object _get_type(str value_type_name):
    return getattr(ValueType, value_type_name)


cdef class JsonSerializeProvider(Provider):
    def __init__(self):
        self.type = ProviderType.json

    cdef inline str _to_str(self, val):
        assert isinstance(val, (bytes, unicode))
        if unicode is str and isinstance(val, bytes):
            return val.decode('utf-8')
        return val

    cdef inline bytes _to_bytes(self, val):
        assert isinstance(val, (bytes, unicode))
        if isinstance(val, unicode):
            return val.encode('utf-8')
        return val

    cdef inline dict _serialize_slice(self, slice value):
        return {
            'type': _get_name(ValueType.slice),
            'value': {
                'start': value.start,
                'stop': value.stop,
                'step': value.step
            }
        }

    cdef inline slice _deserialize_slice(self, object obj, list callbacks):
        value = obj['value']
        return slice(value['start'], value['stop'], value['step'])

    cdef inline dict _serialize_arr(self, np.ndarray value):
        return {
            'type': _get_name(ValueType.arr),
            'value': self._to_str(base64.b64encode(datadumps(value)))
        }

    cdef inline np.ndarray _deserialize_arr(self, object obj, list callbacks):
        cdef bytes bt

        value = obj['value']

        bt = self._to_bytes(base64.b64decode(value))

        if bt is not None:
            return dataloads(bt)

        return None

    cdef inline dict _serialize_dtype(self, np.dtype value):
        cdef str v

        if 'V' not in value.str:
            v = value.str
        else:
            v = self._to_str(base64.b64encode(pickle.dumps(value)))
        return {
            'type': _get_name(ValueType.dtype),
            'value': v
        }

    cdef inline np.dtype _deserialize_dtype(self, object value, list callbacks):
        try:
            return np.dtype(value['value'])
        except TypeError:
            val = value['value']
            return np.dtype(pickle.loads(base64.b64decode(val)))

    cdef inline dict _serialize_index(self, value):
        return {
            'type': _get_name(ValueType.index),
            'value': self._to_str(base64.b64encode(datadumps(value)))
        }

    cdef inline object _deserialize_pd_entity(self, object obj, list callbacks):
        cdef bytes bt

        value = obj['value']

        bt = self._to_bytes(base64.b64decode(value))

        if bt is not None:
            return dataloads(bt)

        return None

    cdef inline dict _serialize_series(self, value):
        return {
            'type': _get_name(ValueType.series),
            'value': self._to_str(base64.b64encode(datadumps(value)))
        }

    cdef inline dict _serialize_dataframe(self, value):
        return {
            'type': _get_name(ValueType.dataframe),
            'value': self._to_str(base64.b64encode(datadumps(value)))
        }

    cdef inline dict _serialize_key(self, value):
        return {
            'type': _get_name(ValueType.key),
            'value': (value.key, value.id)
        }

    cdef inline KeyPlaceholder _deserialize_key(self, object obj, list callbacks):
        if PY_MAJOR_VERSION >= 3:
            return KeyPlaceholder(*obj['value'])
        else:
            return KeyPlaceholder(*(to_str(v) for v in obj['value']))

    cdef inline dict _serialize_list(self, list value, tp=None, bint weak_ref=False):
        return {
            'type': _get_name(ValueType.list),
            'value': [self._serialize_value(val if not weak_ref else val(),
                                            tp.type if tp is not None else None)
                      for val in value]
        }

    cdef inline list _deserialize_list(self, object obj, list callbacks, bint weak_ref):
        cdef list res
        cdef int i
        cdef int j

        res = []
        for i, it_val in enumerate(obj['value']):
            val = self._deserialize_value(it_val, callbacks, weak_ref)
            res.append(val)
            if isinstance(val, KeyPlaceholder):
                def cb(j, v):
                    def inner(subs):
                        o = subs[v.key, v.id]
                        if weak_ref:
                            o = weakref.ref(o)
                        res[j] = o
                    return inner
                callbacks.append(cb(i, val))

        return res

    cdef inline dict _serialize_tuple(self, tuple value, tp=None, bint weak_ref=False):
        if tp is None or not isinstance(tp.type, tuple):
            return {
                'type': _get_name(ValueType.tuple),
                'value': [self._serialize_value(val if not weak_ref else val(),
                                                tp.type if tp is not None else None)
                          for val in value]
            }
        else:
            return {
                'type': _get_name(ValueType.tuple),
                'value': [self._serialize_value(val if not weak_ref else val(),
                                                it_type)
                          for val, it_type in izip(value, tp.type)]
            }

    cdef inline tuple _deserialize_tuple(self, object obj, list callbacks, bint weak_ref):
        return tuple(self._deserialize_value(it_val, callbacks, weak_ref)
                     for it_val in obj['value'])

    cdef inline dict _serialize_dict(self, object value, tp=None, bint weak_ref=False):
        return {
            'type': _get_name(ValueType.dict),
            'value': [(self._serialize_value(k if not weak_ref else k(),
                                             tp.key_type if tp is not None else tp),
                       self._serialize_value(v if not weak_ref else v(),
                                             tp.value_type if tp is not None else tp))
                      for k, v in value.items()]
        }

    cdef inline object _deserialize_dict(self, object obj, list callbacks, bint weak_ref):
        res = OrderedDict()
        for k, v in obj['value']:
            key = self._deserialize_value(k, callbacks, weak_ref)
            value = self._deserialize_value(v, callbacks, weak_ref)
            res[key] = value
            if isinstance(value, KeyPlaceholder):
                def cb(ko, vo):
                    def inner(subs):
                        o = subs[vo.key, vo.id]
                        if weak_ref:
                            o = weakref.ref(o)
                        res[ko] = o
                    return inner
                callbacks.append(cb(key, value))

        return res

    cdef inline dict _serialize_datetime64_timedelta64(self, value, tp):
        bio = six.BytesIO()
        np.save(bio, value)
        return {
            'type': _get_name(tp),
            'value': self._to_str(base64.b64encode(bio.getvalue()))
        }

    cdef inline _deserialize_datetime64_timedelta64(self, obj, list callbacks):
        cdef bytes v

        value = obj['value']
        v = base64.b64decode(value)

        if v is not None:
            return np.load(six.BytesIO(v))
        return None

    cdef inline dict _serialize_datetime64(self, value):
        return self._serialize_datetime64_timedelta64(value, ValueType.datetime64)

    cdef inline _deserialize_datetime64(self, obj, list callbacks):
        return self._deserialize_datetime64_timedelta64(obj, callbacks)

    cdef inline dict _serialize_timedelta64(self, value):
        return self._serialize_datetime64_timedelta64(value, ValueType.timedelta64)

    cdef inline _deserialize_timedelta64(self, obj, list callbacks):
        return self._deserialize_datetime64_timedelta64(obj, callbacks)

    cdef inline dict _serialize_complex(self, value, tp):
        return {
            'type': _get_name(tp),
            'value': (value.real, value.imag)
        }

    cdef inline _deserialize_complex(self, obj, list callbacks):
        cdef list v

        v = obj['value']
        return complex(*v)

    cdef inline object _serialize_typed_value(self, value, tp, bint weak_ref=False):
        if type(tp) not in (List, Tuple, Dict) and weak_ref:
            # not iterable, and is weak ref
            value = value()
        if value is None:
            return
        if tp == ValueType.bytes:
            return {
                'type': _get_name(tp),
                'value': self._to_str(base64.b64encode(value))
            }
        elif tp in PRIMITIVE_TYPE_TO_NAME:
            # primitive type, we do not do any type check here
            return value
        elif type(tp) is Identity:
            return self._serialize_typed_value(value, tp.type, weak_ref=weak_ref)
        elif tp in {ValueType.complex64, ValueType.complex128}:
            return self._serialize_complex(value, tp)
        elif tp is ValueType.slice:
            return self._serialize_slice(value)
        elif tp is ValueType.arr:
            return self._serialize_arr(value)
        elif tp is ValueType.dtype:
            return self._serialize_dtype(value)
        elif tp is ValueType.index:
            return self._serialize_index(value)
        elif tp is ValueType.series:
            return self._serialize_series(value)
        elif tp is ValueType.dataframe:
            return self._serialize_dataframe(value)
        elif tp is ValueType.key:
            return self._serialize_key(value)
        elif tp == ValueType.datetime64:
            return self._serialize_datetime64(value)
        elif tp == ValueType.timedelta64:
            return self._serialize_timedelta64(value)
        elif isinstance(tp, List):
            if not isinstance(value, list):
                value = list(value)
            return self._serialize_list(value, tp, weak_ref=weak_ref)
        elif isinstance(tp, Tuple):
            return self._serialize_tuple(value, tp, weak_ref=weak_ref)
        elif isinstance(tp, Dict):
            return self._serialize_dict(value, tp, weak_ref=weak_ref)
        else:
            raise TypeError('Unknown type to serialize: {0}'.format(tp))

    cdef inline object _serialize_untyped_value(self, value, bint weak_ref=False):
        if not isinstance(value, (list, tuple, dict)) and weak_ref:
            # not iterable, and is weak ref
            value = value()
        if value is None:
            return

        if isinstance(value, bool):
            return value
        elif isinstance(value, bytes):
            return {
                'type': _get_name(ValueType.bytes),
                'value': self._to_str(base64.b64encode(value)),
            }
        elif isinstance(value, unicode):
            return value
        elif isinstance(value, (int, long)):
            return value
        elif isinstance(value, float):
            return value
        elif isinstance(value, complex):
            return self._serialize_complex(value, ValueType.complex128)
        elif isinstance(value, slice):
            return self._serialize_slice(value)
        elif isinstance(value, np.ndarray):
            return self._serialize_arr(value)
        elif isinstance(value, np.dtype):
            return self._serialize_dtype(value)
        elif pd is not None and isinstance(value, pd.Index):
            return self._serialize_index(value)
        elif pd is not None and isinstance(value, pd.Series):
            return self._serialize_series(value)
        elif pd is not None and isinstance(value, pd.DataFrame):
            return self._serialize_dataframe(value)
        elif isinstance(value, HasKey):
            return self._serialize_key(value)
        elif isinstance(value, list):
            return self._serialize_list(value, tp=None, weak_ref=weak_ref)
        elif isinstance(value, tuple):
            return self._serialize_tuple(value, tp=None, weak_ref=weak_ref)
        elif isinstance(value, dict):
            return self._serialize_dict(value, tp=None, weak_ref=weak_ref)
        elif isinstance(value, np.datetime64):
            return self._serialize_datetime64(value)
        elif isinstance(value, np.timedelta64):
            return self._serialize_timedelta64(value)
        elif isinstance(value, np.number):
            return self._serialize_untyped_value(value.item())
        else:
            raise TypeError('Unknown type to serialize: {0}'.format(type(value)))

    cdef inline object _serialize_value(self, value, tp=None, bint weak_ref=False):
        if tp is None:
            return self._serialize_untyped_value(value, weak_ref=weak_ref)
        else:
            return self._serialize_typed_value(value, tp, weak_ref=weak_ref)

    cdef inline _on_serial(self, Field field, x):
        x = x if x is not None else field.default
        return field.on_serialize(x) if field.on_serialize is not None else x

    cpdef serialize_field(self, Field field, model_instance, obj):
        cdef str tag
        cdef object new_obj
        cdef bint has_val
        cdef object value
        cdef OneOfField oneoffield

        if isinstance(field, ReferenceField):
            tag = field.tag_name(self)
            new_obj = obj[tag] = dict()
            if hasattr(model_instance, field.attr):
                field_val = getattr(model_instance, field.attr)
                if field.weak_ref:
                    field_val = field_val()
                if field_val is not None:
                    if not isinstance(field_val, field.type.model):
                        raise TypeError('Does not match type for reference field {0}: '
                                        'expect {1}, got {2}'.format(tag, field.type.model, type(field_val)))
                    value = self._on_serial(field, field_val)
                    value.serialize(self, new_obj)
        elif isinstance(field, OneOfField):
            has_val = False
            field_val = getattr(model_instance, field.attr, None)
            if field.weak_ref and field_val is not None:
                field_val = field_val()
            value = self._on_serial(field, field_val)
            oneoffield = <OneOfField>field
            for f in oneoffield.fields:
                tag = f.tag_name(self)
                if isinstance(value, f.type.model):
                    if not has_val:
                        has_val = True
                    else:
                        raise ValueError('Only one of attributes({0}) can be specified'.format(field.attrs))

                    new_obj = obj[tag] = dict()
                    value.serialize(self, new_obj)
                    return
            if not has_val and value is not None:
                raise ValueError('Value {0} cannot match any type for OneOfField `{1}`'.format(
                    value, field.tag_name(self)))
        elif isinstance(field, ListField) and type(field.type.type) == Reference:
            tag = field.tag_name(self)
            value = self._on_serial(field, getattr(model_instance, field.attr, None))
            if value is None:
                return
            new_obj = obj[tag] = list()
            for val in value:
                if field.weak_ref:
                    val = val()
                if val is not None:
                    if isinstance(val, field.type.type.model):
                        new_obj.append(val.serialize(self, dict()))
                    else:
                        raise TypeError('Does not match type for reference in list field {0}: '
                                        'expect {1}, got {2}'.format(tag, field.type.type.model, type(val)))
                else:
                    new_obj.append(None)
        else:
            tag = field.tag_name(self)
            val = self._on_serial(field, getattr(model_instance, field.attr, None))
            if val is None:
                return
            obj[tag] = self._serialize_value(val, field.type, weak_ref=field.weak_ref)

    cdef inline _deserialize_value(self, obj, list callbacks, bint weak_ref):
        if not isinstance(obj, dict):
            return obj

        ref = lambda x: weakref.ref(x) if weak_ref else x

        if PY_MAJOR_VERSION >= 3:
            tp = _get_type(obj['type'])
        else:
            tp = _get_type(to_str(obj['type']))

        if tp is ValueType.bytes:
            return ref(base64.b64decode(obj['value']))
        elif tp in {ValueType.complex64, ValueType.complex128}:
            return ref(self._deserialize_complex(obj, callbacks))
        elif tp is ValueType.slice:
            return ref(self._deserialize_slice(obj, callbacks))
        elif tp is ValueType.arr:
            return ref(self._deserialize_arr(obj, callbacks))
        elif tp is ValueType.dtype:
            return ref(self._deserialize_dtype(obj, callbacks))
        elif tp in (ValueType.index, ValueType.series, ValueType.dataframe):
            return ref(self._deserialize_pd_entity(obj, callbacks))
        elif tp is ValueType.key:
            return self._deserialize_key(obj, callbacks)  # the weakref will do in the callback, so skip
        elif tp is ValueType.datetime64:
            return ref(self._deserialize_datetime64(obj, callbacks))
        elif tp is ValueType.timedelta64:
            return ref(self._deserialize_timedelta64(obj, callbacks))
        elif tp is ValueType.list:
            return self._deserialize_list(obj, callbacks, weak_ref)
        elif tp is ValueType.tuple:
            return self._deserialize_tuple(obj, callbacks, weak_ref)
        elif tp is ValueType.dict:
            return self._deserialize_dict(obj, callbacks, weak_ref)
        else:
            raise TypeError('Unknown type to deserialize {0}'.format(obj['type']))

    cdef inline _on_deserial(self, Field field, x):
        x = x if x is not None else field.default
        return field.on_deserialize(x) if field.on_deserialize is not None else x

    def deserialize_field(self, Field field, model_instance, obj, list callbacks, dict key_to_instance):
        cdef str tag
        cdef object val
        cdef bint has_val
        cdef OneOfField oneoffield

        if isinstance(field, ReferenceField):
            tag = field.tag_name(self)
            val = obj.get(tag)
            if val is None:
                return
            setattr(model_instance, field.attr,
                    self._on_deserial(field, field.type.model.deserialize(self, val, callbacks, key_to_instance)))
        elif isinstance(field, OneOfField):
            oneoffield = <OneOfField>field
            has_val = False
            for f in oneoffield.fields:
                tag = f.tag_name(self)
                val = obj.get(tag)
                if val:
                    if not has_val:
                        has_val = True
                    else:
                        raise ValueError(
                            'Only one of attributes({0}) can be specified'.format(field.attrs))

                    setattr(model_instance, f.attr,
                            self._on_deserial(field, f.type.model.deserialize(self, obj[tag],
                                                                              callbacks, key_to_instance)))
        elif isinstance(field, ListField) and type(field.type.type) == Reference:
            tag = field.tag_name(self)
            if tag not in obj:
                return
            setattr(model_instance, field.attr,
                    self._on_deserial(
                        field, [field.type.type.model.deserialize(self, it_obj, callbacks, key_to_instance)
                                if it_obj is not None else None
                                for it_obj in obj[tag]]))
        else:
            tag = field.tag_name(self)
            val = self._deserialize_value(obj.get(tag), callbacks, field.weak_ref)
            if val is None:
                return
            setattr(model_instance, field.attr, self._on_deserial(field, val))
            if isinstance(val, KeyPlaceholder):
                def cb(subs):
                    o = subs[val.key, val.id]
                    if field.weak_ref:
                        o = weakref.ref(o)
                    setattr(model_instance, field.attr, self._on_deserial(field, o))
                callbacks.append(cb)

