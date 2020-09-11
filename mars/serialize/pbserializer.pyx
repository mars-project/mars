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

import pickle
import sys
import weakref
from collections import OrderedDict
from datetime import tzinfo
from io import BytesIO

import numpy as np
cimport numpy as np
cimport cython
import cloudpickle
import pandas as pd
from pandas.api.extensions import ExtensionDtype
from pandas.arrays import IntervalArray

from ..utils import serialize_function
from .core cimport ProviderType, ValueType, Identity, List, Tuple, Dict, \
    Reference, KeyPlaceholder, AttrWrapper, Provider, Field, OneOfField, \
    ReferenceField, IdentityField, ListField, TupleField, \
    get_serializable_by_index
from .core import HasKey, HasData
from .protos.value_pb2 import Value
from .dataserializer import dumps as datadumps, loads as dataloads

try:
    from pandas.tseries.offsets import Tick as PDTick
    from pandas.tseries.frequencies import to_offset
except ImportError:
    PDTick = to_offset = None


cdef dict PRIMITIVE_TYPE_TO_VALUE_FIELD = {
    ValueType.bool: 'b',
    ValueType.int8: 'i',
    ValueType.int16: 'i',
    ValueType.int32: 'i',
    ValueType.int64: 'i',
    ValueType.uint8: 'i',
    ValueType.uint16: 'i',
    ValueType.uint32: 'i',
    ValueType.uint64: 'i',
    ValueType.float16: 'f',
    ValueType.float32: 'f',
    ValueType.float64: 'f',
    ValueType.bytes: 's',
    ValueType.unicode: 'u',
}


cdef class ProtobufSerializeProvider(Provider):
    def __init__(self, data_serial_type=None, pickle_protocol=None):
        self.type = ProviderType.protobuf
        self.data_serial_type = data_serial_type
        self.pickle_protocol = pickle_protocol

    cdef inline void _set_slice(self, slice value, obj, tp=None):
        if value.start is not None:
            if isinstance(value.start, (int, np.integer)):
                obj.slice.start_val = value.start
            elif isinstance(value.start, str):
                obj.slice.start_label = value.start
            else:
                # dump if not integer or string
                obj.slice.start_obj = pickle.dumps(value.start, protocol=self.pickle_protocol)
        if value.stop is not None:
            if isinstance(value.stop, (int, np.integer)):
                obj.slice.stop_val = value.stop
            elif isinstance(value.stop, str):
                obj.slice.stop_label = value.stop
            else:
                # dump if not integer or string
                obj.slice.stop_obj = pickle.dumps(value.stop, protocol=self.pickle_protocol)
        if value.step is not None:
            obj.slice.step_val = value.step
        if all(it is None for it in (value.start, value.stop, value.step)):
            obj.slice.is_null = True

    cdef inline slice _get_slice(self, obj):
        if obj.slice.is_null:
            return slice(None)

        start_key = obj.slice.WhichOneof('start')
        if start_key is not None:
            start = getattr(obj.slice, start_key)
            if start_key == 'start_obj':
                start = pickle.loads(start)
        else:
            start = None
        stop_key = obj.slice.WhichOneof('stop')
        if stop_key is not None:
            stop = getattr(obj.slice, stop_key)
            if stop_key == 'stop_obj':
                stop = pickle.loads(stop)
        else:
            stop = None
        step = obj.slice.step_val if obj.slice.WhichOneof('step') else None
        return slice(start, stop, step)

    cdef inline void _set_arr(self, np.ndarray value, obj, tp=None):
        # special case for np.unicode and np.bytes_
        # cuz datadumps may fail due to pyarrow
        if value.ndim == 0 and value.dtype.kind in ('U', 'S'):
            value = value.astype(object)
        obj.arr = datadumps(value, serial_type=self.data_serial_type,
                            pickle_protocol=self.pickle_protocol)

    cdef inline np.ndarray _get_arr(self, obj):
        cdef object x

        x = obj.arr
        return dataloads(x) if x is not None and len(x) > 0 else None

    cdef inline void _set_dtype(self, value, obj, tp=None):
        if not isinstance(value, ExtensionDtype) and 'V' not in value.str:
            dtype = value.str
            if isinstance(dtype, unicode):
                dtype = dtype.encode('utf-8')
            obj.dtype = dtype
        else:
            obj.dtype = pickle.dumps(value, protocol=self.pickle_protocol)

    cdef inline object _get_dtype(self, obj):
        if obj.dtype is None or len(obj.dtype) == 0:
            return
        try:
            return np.dtype(obj.dtype)
        except (TypeError, ValueError):
            return pickle.loads(obj.dtype)

    cdef inline void _set_index(self, value, obj, tp=None) except *:
        obj.index = datadumps(value, serial_type=self.data_serial_type,
                              pickle_protocol=self.pickle_protocol)

    cdef inline _get_index(self, obj):
        x = obj.index
        return dataloads(x) if x is not None and len(x) > 0 else None

    cdef inline void _set_series(self, value, obj, tp=None) except *:
        obj.series = datadumps(value, serial_type=self.data_serial_type,
                               pickle_protocol=self.pickle_protocol)

    cdef inline _get_series(self, obj):
        x = obj.series
        return dataloads(x) if x is not None and len(x) > 0 else None

    cdef inline void _set_dataframe(self, value, obj, tp=None) except *:
        obj.dataframe = datadumps(value, serial_type=self.data_serial_type,
                                  pickle_protocol=self.pickle_protocol)

    cdef inline object _get_dataframe(self, obj):
        x = obj.dataframe
        return dataloads(x) if x is not None and len(x) > 0 else None

    cdef inline void _set_key(self, object value, obj, tp=None) except *:
        obj.key.key = value.key
        obj.key.id = value.id

    cdef inline KeyPlaceholder _get_key(self, obj):
        if obj.key.key is None or len(obj.key.key) == 0:
            return
        return KeyPlaceholder(obj.key.key, obj.key.id)

    cdef inline void _set_datetime64(self, value, obj, tp=None):
        if isinstance(value, pd.Timestamp):
            # convert to np.datetime64
            value = value.to_datetime64()
        bio = BytesIO()
        np.save(bio, value)
        obj.datetime64 = bio.getvalue()

    cdef inline object _get_datetime64(self, obj):
        cdef object x

        x = obj.datetime64
        if x is not None and len(x) > 0:
            # np.load will return a ndarray
            value = np.load(BytesIO(x))
            return value.dtype.type(value)
        else:
            return None

    cdef inline void _set_timedelta64(self, value, obj, tp=None):
        if isinstance(value, pd.Timedelta):
            # convert to np.timedelta64
            value = value.to_timedelta64()
        bio = BytesIO()
        np.save(bio, value)
        obj.timedelta64 = bio.getvalue()

    cdef inline object _get_timedelta64(self, obj):
        cdef object x

        x = obj.timedelta64
        if x is not None and len(x) > 0:
            # np.load will return a ndarray
            value = np.load(BytesIO(x))
            return value.dtype.type(value)
        else:
            return None

    cdef inline void _set_complex(self, value, obj, tp=None):
        obj.c.real = value.real
        obj.c.imag = value.imag

    cdef inline object _get_complex(self, obj):
        return complex(obj.c.real, obj.c.imag)

    cdef inline void _set_function(self, value, obj, tp=None):
        obj.function = serialize_function(value, pickle_protocol=self.pickle_protocol)

    cdef inline object _get_function(self, obj):
        x = obj.function
        return cloudpickle.loads(x) if x is not None and len(x) > 0 else None

    cdef inline void _set_namedtuple(self, value, obj, tp=None):
        obj.namedtuple = cloudpickle.dumps(value, protocol=self.pickle_protocol)

    cdef inline object _get_namedtuple(self, obj):
        x = obj.namedtuple
        return cloudpickle.loads(x) if x is not None and len(x) > 0 else None

    cdef inline void _set_tzinfo(self, value, obj, tp=None):
        obj.tzinfo = pickle.dumps(value, protocol=self.pickle_protocol)

    cdef inline object _get_tzinfo(self, obj):
        x = obj.tzinfo
        return pickle.loads(x) if x is not None and len(x) > 0 else None

    cdef inline void _set_interval_arr(self, value, obj, tp=None):
        obj.interval_arr.left = datadumps(value.left, serial_type=self.data_serial_type,
                                          pickle_protocol=self.pickle_protocol)
        obj.interval_arr.right = datadumps(value.right, serial_type=self.data_serial_type,
                                           pickle_protocol=self.pickle_protocol)
        obj.interval_arr.closed = value.closed
        self._set_dtype(value.dtype, obj.interval_arr)

    cdef inline object _get_interval_arr(self, obj):
        interval_arr = obj.interval_arr
        if not interval_arr.left:
            return
        left = dataloads(interval_arr.left)
        right = dataloads(interval_arr.right)
        closed = interval_arr.closed
        dtype = self._get_dtype(interval_arr)
        return IntervalArray.from_arrays(left, right,
                                         closed=closed, dtype=dtype)

    cdef inline void _set_freq(self, value, obj, tp=None):
        obj.freq = value.freqstr

    cdef inline object _get_freq(self, obj):
        return to_offset(obj.freq)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef inline void _set_list(self, list value, obj, tp=None, bint weak_ref=False) except *:
        cdef list res

        # list is special, cuz the internal object can be a specified type or any type
        if isinstance(obj, Value):
            for val in value:
                if weak_ref:
                    val = val()
                it_obj = obj.list.value.add()
                self._set_value(val, it_obj, tp=tp.type if tp is not None else tp)
        else:
            try:
                obj.extend(value)
            except TypeError:
                # not primitive type
                res = []
                for val in value:
                    if weak_ref:
                        val = val()
                    it_obj = Value()
                    self._set_value(val, it_obj, tp=tp.type if tp is not None else tp)
                    res.append(it_obj)
                obj.extend(res)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef inline list _get_list(self, obj, tp, list callbacks, bint weak_ref):
        cdef list res
        cdef int i
        cdef object it_obj

        obj = obj.list.value if isinstance(obj, Value) else obj
        res = []
        for i, it_obj in enumerate(obj):
            if not isinstance(it_obj, Value):
                res.append(it_obj if not weak_ref else weakref.ref(it_obj))
                continue

            val = self._get_value(it_obj, tp.type if tp is not None else tp,
                                  callbacks, weak_ref)
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef inline void _set_tuple(self, tuple value, obj, tp=None, bint weak_ref=False) except *:
        cdef list res

        if isinstance(obj, Value):
            obj.list.is_tuple = True
            if tp is not None and isinstance(tp.type, tuple):
                if len(tp.type) != len(value):
                    raise ValueError(f'Value length should be {len(tp.type)} '
                                     f'according to type definition, got {len(value)}')
                for it_type, val in zip(tp.type, value):
                    if weak_ref:
                        val = val()
                    if it_type == ValueType.key:
                        raise TypeError('`key` type is not allowed in a tuple')
                    it_obj = obj.list.value.add()
                    self._set_typed_value(val, it_obj, it_type)
            else:
                if tp == ValueType.key:
                    raise TypeError('`key` type is not allowed in a tuple')
                for val in value:
                    if weak_ref:
                        val = val()
                    it_obj = obj.list.value.add()
                    self._set_value(val, it_obj, tp=tp.type if tp is not None else tp)
        else:
            try:
                if weak_ref:
                    value = tuple(val() for val in value)
                obj.extend(value)
            except TypeError:
                # not primitive type
                res = []
                for val in value:
                    if weak_ref:
                        val = val()
                    it_obj = Value()
                    self._set_value(val, it_obj, tp=tp.type if tp is not None else tp)
                    res.append(it_obj)
                obj.extend(res)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef inline tuple _get_tuple(self, obj, tp, list callbacks, bint weak_ref):
        cdef list res

        values = obj.list.value if isinstance(obj, Value) else obj

        res = []
        if tp is not None and isinstance(tp.type, tuple):
            for it_type, it_obj in zip(tp.type, values):
                if isinstance(it_obj, Value):
                    res.append(self._get_value(it_obj, it_type,
                                               callbacks, weak_ref))
                else:
                    if weak_ref:
                        it_obj = weakref.ref(it_obj)
                    res.append(it_obj)
        else:
            for it_obj in values:
                if isinstance(it_obj, Value):
                    res.append(self._get_value(it_obj, tp.type if tp is not None else tp,
                                               callbacks, weak_ref))
                else:
                    if weak_ref:
                        it_obj = weakref.ref(it_obj)
                    res.append(it_obj)

        return tuple(res)

    cdef inline void _set_dict(self, value, obj, tp=None, bint weak_ref=False) except *:
        for k, v in value.items():
            if weak_ref:
                k, v = k(), v()
            key_obj = obj.dict.keys.value.add()
            if tp is not None and tp.key_type == ValueType.key:
                raise TypeError('`key` type is not allowd as dict key')
            self._set_value(k, key_obj, tp=tp.key_type if tp is not None else tp)
            value_obj = obj.dict.values.value.add()
            self._set_value(v, value_obj, tp=tp.value_type if tp is not None else tp)

    cdef inline object _get_dict(self, obj, tp, list callbacks, bint weak_ref):
        d = OrderedDict()
        for k_obj, v_obj in zip(obj.dict.keys.value, obj.dict.values.value):
            k = self._get_value(k_obj, tp.key_type if tp is not None else tp,
                                callbacks, weak_ref)
            v = self._get_value(v_obj, tp.value_type if tp is not None else tp,
                                callbacks, weak_ref)
            d[k] = v
            if isinstance(v, KeyPlaceholder):
                def cb(key, value):
                    def inner(subs):
                        o = subs[value.key, value.id]
                        if weak_ref:
                            o = weakref.ref(o)
                        d[key] = o
                    return inner
                callbacks.append(cb(k, v))

        return d

    cdef inline void _set_typed_value(self, value, obj, tp, bint weak_ref=False) except *:
        cdef str value_field

        if type(tp) not in (List, Tuple, Dict) and weak_ref:
            # not iterable, and is weak ref
            value = value()

        if value is None:
            # handle None
            obj.is_null = True
            return

        if tp in PRIMITIVE_TYPE_TO_VALUE_FIELD:
            value_field = PRIMITIVE_TYPE_TO_VALUE_FIELD[tp]
            setattr(obj, value_field, value)
        elif tp is ValueType.slice:
            self._set_slice(value, obj, tp)
        elif tp is ValueType.arr:
            self._set_arr(<np.ndarray>value, obj, tp)
        elif tp is ValueType.dtype:
            self._set_dtype(<np.dtype>value, obj, tp)
        elif tp is ValueType.index:
            self._set_index(value, obj, tp)
        elif tp is ValueType.series:
            self._set_series(value, obj, tp)
        elif tp is ValueType.dataframe:
            self._set_dataframe(value, obj, tp)
        elif tp is ValueType.key:
            self._set_key(value, obj, tp)
        elif tp is ValueType.datetime64:
            self._set_datetime64(value, obj, tp)
        elif tp is ValueType.timedelta64:
            self._set_timedelta64(value, obj, tp)
        elif tp is ValueType.function:
            self._set_function(value, obj, tp)
        elif tp is ValueType.tzinfo:
            self._set_tzinfo(value, obj, tp)
        elif tp is ValueType.interval_arr:
            self._set_interval_arr(value, obj, tp)
        elif tp is ValueType.freq:
            self._set_freq(value, obj, tp)
        elif tp in {ValueType.complex64, ValueType.complex128}:
            self._set_complex(value, obj, tp)
        elif tp is ValueType.namedtuple:
            self._set_namedtuple(value, obj, tp)
        elif isinstance(tp, Identity):
            value_field = PRIMITIVE_TYPE_TO_VALUE_FIELD[tp.type]
            setattr(obj, value_field, value)
        elif isinstance(tp, List):
            # list type
            self._set_list(<list>value, obj, tp, weak_ref=weak_ref)
        elif isinstance(tp, Tuple):
            # tuple type
            self._set_tuple(<tuple>value, obj, tp, weak_ref=weak_ref)
        elif isinstance(tp, Dict):
            # dict type
            self._set_dict(<dict>value, obj, tp, weak_ref=weak_ref)
        else:
            raise TypeError(f'Unknown type to serialize: {tp}')

    cdef inline void _set_untyped_value(self, value, obj, bint weak_ref=False) except *:
        if not isinstance(value, (list, tuple, dict)) and weak_ref:
            # not iterable, and is weak ref
            value = value()

        # we are not aware of the type, so try to infer it
        if value is None:
            # handle None
            obj.is_null = True
            return

        if isinstance(value, bool):
            obj.b = value
        elif isinstance(value, bytes):
            obj.s = value
        elif isinstance(value, unicode):
            obj.u = value
        elif isinstance(value, int):
            obj.i = value
        elif isinstance(value, float):
            obj.f = value
        elif isinstance(value, complex):
            self._set_complex(value, obj)
        elif isinstance(value, slice):
            self._set_slice(value, obj)
        elif isinstance(value, np.ndarray):
            self._set_arr(value, obj)
        elif isinstance(value, (np.dtype, ExtensionDtype)):
            self._set_dtype(value, obj)
        elif isinstance(value, pd.Index):
            self._set_index(value, obj)
        elif isinstance(value, pd.Series):
            self._set_series(value, obj)
        elif isinstance(value, pd.DataFrame):
            self._set_dataframe(value, obj)
        elif isinstance(value, HasKey):
            self._set_key(value, obj)
        elif isinstance(value, HasData):
            self._set_key(value.data, obj)
        elif isinstance(value, list):
            self._set_list(value, obj, tp=None, weak_ref=weak_ref)
        elif isinstance(value, tuple):
            if hasattr(value, '_fields'):
                # namedtuple
                self._set_namedtuple(value, obj, tp=None)
            else:
                self._set_tuple(value, obj, tp=None, weak_ref=weak_ref)
        elif isinstance(value, dict):
            self._set_dict(value, obj, tp=None, weak_ref=weak_ref)
        elif isinstance(value, (np.datetime64, pd.Timestamp)):
            self._set_datetime64(value, obj)
        elif isinstance(value, (np.timedelta64, pd.Timedelta)):
            self._set_timedelta64(value, obj)
        elif isinstance(value, np.number):
            self._set_untyped_value(value.item(), obj)
        elif isinstance(value, tzinfo):
            self._set_tzinfo(value, obj)
        elif isinstance(value, IntervalArray):
            self._set_interval_arr(value, obj)
        elif PDTick is not None and isinstance(value, PDTick):
            self._set_freq(value, obj)
        elif callable(value):
            self._set_function(value, obj)
        else:
            raise TypeError(f'Unknown type to serialize: {type(value)}')

    cdef inline void _set_value(cls, value, obj, tp=None, bint weak_ref=False) except *:
        if tp is None:
            cls._set_untyped_value(value, obj, weak_ref=weak_ref)
        else:
            cls._set_typed_value(value, obj, tp, weak_ref=weak_ref)

    cdef inline void _serial_reference_value(self, tag, model, value, value_pb) except *:
        cdef object field_obj

        if model is None:
            if type(value_pb) is Value and not hasattr(value, 'cls'):
                value_pb.typed.type_id = value.__serializable_index__
                value.serialize(self, value_pb.typed.value)
            else:
                if type(value_pb) is Value:
                    value_pb = value_pb.any_ref
                field_obj = value.cls(self)()
                value.serialize(self, obj=field_obj)
                value_pb.type_id = value.__serializable_index__
                value_pb.value = field_obj.SerializeToString()
        else:
            if isinstance(value, model):
                value.serialize(self, obj=value_pb)
            else:
                raise TypeError(f'Does not match type for reference field {tag}: '
                                f'expect {model}, got {type(value)}')

    cpdef serialize_field(self, Field field, model_instance, obj):
        cdef object value
        cdef object val
        cdef Field f
        cdef str tag
        cdef object field_obj
        cdef object add
        cdef object it_obj
        cdef bint matched
        cdef OneOfField oneoffield

        if isinstance(field, OneOfField):
            oneoffield = <OneOfField> field
            value = getattr(model_instance, oneoffield.attr, None)
            matched = False
            for f in oneoffield.fields:
                if isinstance(value, f.type.model):
                    f.serialize(self, model_instance, obj)
                    matched = True
                    return
            if not matched and value is not None:
                raise ValueError(f'Value {value} cannot match any type for OneOfField `{field.tag_name(self)}`')
            return

        value = getattr(model_instance, field.attr, field.default)
        if field.on_serialize and value is not None:
            value = field.on_serialize(value)
        tag = field.tag_name(self)
        try:
            field_obj = getattr(obj, tag)
        except AttributeError:
            raise AttributeError(f'no attribute {tag} for {model_instance}')

        if value is None:
            return

        if isinstance(field, ReferenceField):
            if field.weak_ref:
                field_obj = field_obj()
            self._serial_reference_value(tag, field.type.model, value, field_obj)
        elif isinstance(field, ListField):
            if type(field.type.type) == Reference:
                add = field_obj.list.value.add if isinstance(field_obj, Value) else field_obj.add
                for val in value:
                    if field.weak_ref:
                        val = val()
                    it_obj = add()
                    if val is not None:
                        self._serial_reference_value(tag, field.type.type.model, val, it_obj)
                    elif isinstance(it_obj, Value):
                        it_obj.is_null = True
            else:
                # repeated primitive type
                if not isinstance(value, list):
                    value = list(value)
                self._set_list(value, field_obj, tp=field.type, weak_ref=field.weak_ref)
        elif isinstance(field_obj, Value):
            try:
                self._set_value(value, field_obj, field.type, weak_ref=field.weak_ref)
            except TypeError:
                exc_info = sys.exc_info()
                raise TypeError(f'Failed to set field `{tag}` for {model_instance} with '
                                f'value {value}, reason: {exc_info[1]}') \
                    .with_traceback(exc_info[2]) from exc_info[1]
        elif isinstance(field, TupleField):
            try:
                self._set_tuple(value, field_obj, tp=field.type, weak_ref=field.weak_ref)
            except TypeError:
                exc_info = sys.exc_info()
                raise TypeError(f'Failed to set field `{tag}` for {model_instance} with '
                                f'value {value}, reason: {exc_info[1]}') \
                    .with_traceback(exc_info[2]) from exc_info[1]
        else:
            try:
                setattr(obj, tag, value)
            except TypeError:
                exc_info = sys.exc_info()
                raise TypeError(f'Failed to set field `{tag}` for {model_instance} with '
                                f'value {value}, reason: {exc_info[1]}') \
                    .with_traceback(exc_info[2]) from exc_info[1]

    cpdef serialize_attribute_as_dict(self, model_instance, obj=None):
        cdef object id_field
        cdef str attr
        cdef object d_obj
        cdef dict fields
        cdef str name
        cdef object field
        cdef object value

        if obj is None:
            obj = model_instance.cls(self)()

        if hasattr(model_instance, '_ID_FIELD') and not isinstance(obj, Value):
            for id_field in model_instance._ID_FIELD:
                id_field.serialize(self, model_instance, obj)

        attr = getattr(model_instance, 'attr_tag', None)
        if attr:
            d_obj = getattr(obj, attr)
        else:
            d_obj = obj

        if isinstance(d_obj, Value):
            fields = model_instance._FIELDS
            for name, field in fields.items():
                value = getattr(model_instance, name, None)
                if value is None:
                    continue
                tag = field.tag_name(self)

                k = d_obj.dict.keys.value.add()
                self._set_value(field.tag_name(self), k, tp=ValueType.string)
                v = d_obj.dict.values.value.add()
                if isinstance(field, ReferenceField):
                    if field.weak_ref:
                        value = value()
                    self._serial_reference_value(tag, field.type.model, value, v)
                elif isinstance(field, ListField) and type(field.type.type) == Reference:
                    for val in value:
                        if field.weak_ref:
                            val = val()
                        it_obj = v.list.value.add()
                        if val is not None:
                            self._serial_reference_value(tag, field.type.type.model, val, it_obj)
                        elif isinstance(it_obj, Value):
                            it_obj.is_null = True
                else:
                    self._set_value(value, v, tp=field.type, weak_ref=field.weak_ref)
        else:
            fields = model_instance._FIELDS
            for name, field in fields.items():
                value = getattr(model_instance, name, None)
                if value is not None:
                    field.serialize(self, model_instance, AttrWrapper(d_obj))

        return obj

    cdef inline object _get_typed_value(self, obj, tp, list callbacks, bint weak_ref):
        cdef str value_field

        if obj.is_null:
            return

        if weak_ref:
            ref = weakref.ref
        else:
            ref = lambda x: x

        if tp in PRIMITIVE_TYPE_TO_VALUE_FIELD:
            value_field = PRIMITIVE_TYPE_TO_VALUE_FIELD[tp]
            return ref(getattr(obj, value_field))
        elif tp in {ValueType.complex64, ValueType.complex128}:
            return ref(self._get_complex(obj))
        elif tp is ValueType.slice:
            return ref(self._get_slice(obj))
        elif tp is ValueType.arr:
            return ref(self._get_arr(obj))
        elif tp is ValueType.dtype:
            return ref(self._get_dtype(obj))
        elif tp is ValueType.index:
            return ref(self._get_index(obj))
        elif tp is ValueType.series:
            return ref(self._get_series(obj))
        elif tp is ValueType.dataframe:
            return ref(self._get_dataframe(obj))
        elif tp is ValueType.key:
            return self._get_key(obj)
        elif tp is ValueType.datetime64:
            return ref(self._get_datetime64(obj))
        elif tp is ValueType.timedelta64:
            return ref(self._get_timedelta64(obj))
        elif tp is ValueType.function:
            return ref(self._get_function(obj))
        elif tp is ValueType.tzinfo:
            return ref(self._get_tzinfo(obj))
        elif tp is ValueType.interval_arr:
            return ref(self._get_interval_arr(obj))
        elif tp is ValueType.freq:
            return ref(self._get_freq(obj))
        elif tp is ValueType.namedtuple:
            return ref(self._get_namedtuple(obj))
        elif isinstance(tp, Identity):
            value_field = PRIMITIVE_TYPE_TO_VALUE_FIELD[tp.type]
            return ref(getattr(obj, value_field))
        elif isinstance(tp, List):
            # list type
            return self._get_list(obj, tp, callbacks, weak_ref)
        elif isinstance(tp, Tuple):
            # tuple type
            return self._get_tuple(obj, tp, callbacks, weak_ref)
        elif isinstance(tp, Dict):
            # dict type
            return self._get_dict(obj, tp, callbacks, weak_ref)
        else:
            raise TypeError(f'Unknown type to deserialize: {tp}')

    cdef inline object _get_untyped_value(self, obj, list callbacks, bint weak_ref):
        cdef str field

        if weak_ref:
            ref = weakref.ref
        else:
            ref = lambda x: x

        # we are not aware of the type, so try to infer it
        field = obj.WhichOneof('value')

        if field is None or field == 'is_null':
            return
        elif field in 'bif':
            # primitive type
            return ref(getattr(obj, field))
        elif field == 'c':
            return ref(self._get_complex(obj))
        elif field == 's':
            # bytes
            b = obj.s
            if not isinstance(b, bytes):
                b = b.encode('utf-8')
            return ref(b)
        elif field == 'u':
            # unicode
            u = obj.u
            if not isinstance(u, unicode):
                u = u.decode('utf-8')
            return ref(u)
        elif field == 'list':
            if obj.list.is_tuple:
                return self._get_tuple(obj, None, callbacks, weak_ref)
            else:
                return self._get_list(obj, None, callbacks, weak_ref)
        elif field == 'dict':
            return self._get_dict(obj, None, callbacks, weak_ref)
        elif field == 'slice':
            return ref(self._get_slice(obj))
        elif field == 'arr':
            return ref(self._get_arr(obj))
        elif field == 'dtype':
            return ref(self._get_dtype(obj))
        elif field == 'index':
            return ref(self._get_index(obj))
        elif field == 'series':
            return ref(self._get_series(obj))
        elif field == 'dataframe':
            return ref(self._get_dataframe(obj))
        elif field == 'key':
            return self._get_key(obj)
        elif field == 'datetime64':
            return ref(self._get_datetime64(obj))
        elif field == 'timedelta64':
            return ref(self._get_timedelta64(obj))
        elif field == 'function':
            return ref(self._get_function(obj))
        elif field == 'tzinfo':
            return ref(self._get_tzinfo(obj))
        elif field == 'interval_arr':
            return ref(self._get_interval_arr(obj))
        elif field == 'freq':
            return ref(self._get_freq(obj))
        elif field == 'namedtuple':
            return ref(self._get_namedtuple(obj))
        else:
            raise TypeError('Unknown type to deserialize')

    cdef inline object _get_value(self, obj, tp, list callbacks, bint weak_ref):
        if tp is None:
            return self._get_untyped_value(obj, callbacks, weak_ref)
        else:
            return self._get_typed_value(obj, tp, callbacks, weak_ref)

    cdef inline object _on_deserial(self, Field field, x):
        x = x if x is not None else field.default
        if field.on_deserialize is None:
            return x
        return field.on_deserialize(x)

    cdef object _deserial_reference_value(self, model, value_pb, list callbacks, dict key_to_instance):
        cdef object f_obj

        if model is None:
            if type(value_pb) is Value and value_pb.typed.type_id != 0:
                model = get_serializable_by_index(value_pb.typed.type_id)
                f_obj = value_pb.typed.value
            else:
                if type(value_pb) is Value:
                    value_pb = value_pb.any_ref
                if value_pb.type_id == 0:
                    return None
                model = get_serializable_by_index(value_pb.type_id)
                f_obj = model.cls(self)()
                f_obj.ParseFromString(value_pb.value)
        else:
            f_obj = value_pb
        return model.deserialize(self, f_obj, callbacks, key_to_instance)

    def deserialize_field(self, Field field, model_instance, obj, list callbacks, dict key_to_instance):
        cdef str tag
        cdef str f_tag
        cdef Field f
        cdef object f_obj
        cdef object it_obj
        cdef object field_obj
        cdef object value
        cdef OneOfField oneoffield

        if isinstance(field, OneOfField):
            oneoffield = <OneOfField> field
            tag = oneoffield.tag_name(self)
            f_tag = obj.WhichOneof(tag)
            if f_tag is None:
                return
            f = next(f for f in oneoffield.fields if f.tag_name(self) == f_tag)
            f_obj = getattr(obj, f_tag)
            setattr(model_instance, f.attr,
                    f.type.model.deserialize(self, f_obj, callbacks, key_to_instance))
            return

        tag = field.tag_name(self)
        field_obj = getattr(obj, tag)

        if isinstance(field, ReferenceField):
            value = self._deserial_reference_value(field.type.model, field_obj, callbacks, key_to_instance)
            if value is None:
                return
            setattr(model_instance, field.attr, self._on_deserial(field, value))
        elif isinstance(field, ListField):
            if type(field.type.type) == Reference:
                field_obj = field_obj.list.value if isinstance(field_obj, Value) else field_obj
                value = [self._deserial_reference_value(
                    field.type.type.model, it_obj, callbacks, key_to_instance) for it_obj in field_obj]
                setattr(model_instance, field.attr, self._on_deserial(field, value))
            else:
                value = self._get_list(field_obj, field.type, callbacks, field.weak_ref)
                setattr(model_instance, field.attr,
                        self._on_deserial(field, value))
        elif isinstance(field_obj, Value):
            val = self._get_value(field_obj, field.type, callbacks, field.weak_ref)
            setattr(model_instance, field.attr, self._on_deserial(field, val))
            if isinstance(val, KeyPlaceholder):
                def cb(subs):
                    o = self._on_deserial(field, subs[val.key, val.id])
                    if field.weak_ref:
                        o = weakref.ref(o)
                    setattr(model_instance, field.attr, o)

                callbacks.append(cb)
        elif isinstance(field, TupleField):
            setattr(model_instance, field.attr,
                    self._on_deserial(field, self._get_tuple(field_obj, field.type, callbacks,
                                                             field.weak_ref)))
        else:
            setattr(model_instance, field.attr, self._on_deserial(field, getattr(obj, tag)))

    def deserialize_attribute_as_dict(self, model_cls, obj, list callbacks, dict key_to_instance):
        from ..core import Base

        cdef str attr
        cdef object d_obj
        cdef object kw
        cdef tuple id_fields
        cdef IdentityField id_field
        cdef object model_instance
        cdef dict tag_to_fields
        cdef Field field
        cdef Field f
        cdef OneOfField oneoffield
        cdef Field it_field
        cdef str tag
        cdef object o_tag
        cdef object value

        attr = getattr(model_cls, 'attr_tag', None)
        if attr:
            d_obj = getattr(obj, attr)
        else:
            d_obj = obj

        kw = AttrWrapper(dict())
        if hasattr(model_cls, '_ID_FIELD'):
            # get the id field
            if attr is None:
                if d_obj.is_null:
                    return
                id_fields = model_cls._ID_FIELD
                for id_field in id_fields:
                    tag = id_field.tag_name(self)
                    v = next(v for k, v in zip(d_obj.dict.keys.value, d_obj.dict.values.value)
                             if self._get_value(k, ValueType.string, callbacks, id_field.weak_ref) == tag)
                    setattr(kw, id_field.attr, self._get_value(v, id_field.type,
                                                               callbacks, id_field.weak_ref))
            else:
                for id_field in model_cls._ID_FIELD:
                    id_field.deserialize(self, kw, obj, callbacks, key_to_instance)
        model_instance = model_cls(**kw.asdict())

        tag_to_fields = dict()
        for field in model_instance._FIELDS.values():
            if isinstance(field, OneOfField):
                oneoffield = <OneOfField> field
                for f in oneoffield.fields:
                    tag_to_fields[f.tag_name(self)] = f
            else:
                tag_to_fields[field.tag_name(self)] = field

        if isinstance(d_obj, Value):
            if d_obj.is_null:
                return
            for k, v in zip(d_obj.dict.keys.value, d_obj.dict.values.value):
                tag = self._get_value(k, ValueType.string, callbacks, False)
                it_field = tag_to_fields[tag]
                if isinstance(it_field, ReferenceField):
                    value = self._deserial_reference_value(it_field.type.model, v, callbacks, key_to_instance)
                    if value is None:
                        return
                    setattr(model_instance, it_field.attr, self._on_deserial(it_field, value))
                elif isinstance(it_field, ListField) and type(it_field.type.type) == Reference:
                    value = [self._deserial_reference_value(
                        it_field.type.type.model, it_obj, callbacks, key_to_instance)
                        for it_obj in v.list.value]
                    setattr(model_instance, it_field.attr, self._on_deserial(it_field, value))
                else:
                    setattr(model_instance, it_field.attr,
                            self._get_value(v, it_field.type, callbacks, it_field.weak_ref))
        else:
            for o_tag in d_obj:
                it_field = tag_to_fields[o_tag]
                it_field.deserialize(self, model_instance, AttrWrapper(d_obj),
                                     callbacks, key_to_instance)

        if isinstance(model_instance, Base):
            key_to_instance[model_instance.key, model_instance.id] = model_instance
        return model_instance
