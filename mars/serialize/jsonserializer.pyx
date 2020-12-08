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

import base64
import pickle
import re
import sys
import weakref
from collections import OrderedDict
from datetime import tzinfo
from io import BytesIO

import cloudpickle
import numpy as np
cimport numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype
from pandas.arrays import IntervalArray

from ..utils import serialize_function
from .core cimport Provider, ValueType, ProviderType, \
    Field, List, Tuple, Dict, Identity, Reference, KeyPlaceholder, \
    ReferenceField, OneOfField, ListField, get_serializable_by_index
from .core import HasKey, HasData
from .dataserializer import dumps as datadumps, loads as dataloads

try:
    from pandas.tseries.offsets import Tick as PDTick
    from pandas.tseries.frequencies import to_offset
except ImportError:
    PDTick = to_offset = None
try:
    from re import Pattern as RE_Pattern
except ImportError:
    RE_Pattern = type(re.compile('a'))


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
    ValueType.namedtuple: 'namedtuple',
    ValueType.regex: 'regex',
    ValueType.pickled: 'pickled',
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
    def __init__(self, data_serial_type=None, pickle_protocol=None):
        self.type = ProviderType.json
        self.data_serial_type = data_serial_type
        self.pickle_protocol = pickle_protocol

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

    cdef inline dict _serialize_start_stop(self, pos):
        if isinstance(pos, int):
            return {'type': 'int',
                    'value': pos}
        elif isinstance(pos, str):
            return {'type': 'str',
                    'value': pos}
        else:
            return {'type': 'object',
                    'value': self._to_str(base64.b64encode(
                        pickle.dumps(pos, protocol=self.pickle_protocol)))}

    cdef inline dict _serialize_slice(self, slice value):
        return {
            'type': _get_name(ValueType.slice),
            'value': {
                'start': self._serialize_start_stop(value.start),
                'stop': self._serialize_start_stop(value.stop),
                'step': value.step
            }
        }

    cdef inline object _deserialize_start_stop(self, value):
        if value['type'] in ('int', 'str'):
            return value['value']
        else:
            return pickle.loads(base64.b64decode(value['value']))

    cdef inline slice _deserialize_slice(self, object obj, list callbacks):
        value = obj['value']
        return slice(self._deserialize_start_stop(value['start']),
                     self._deserialize_start_stop(value['stop']),
                     value['step'])

    cdef inline dict _serialize_arr(self, np.ndarray value):
        # special case for np.unicode and np.bytes_
        # cuz datadumps may fail due to pyarrow
        if value.ndim == 0 and value.dtype.kind in ('U', 'S'):
            value = value.astype(object)
        return {
            'type': _get_name(ValueType.arr),
            'value': self._to_str(base64.b64encode(datadumps(value, serial_type=self.data_serial_type,
                                                             pickle_protocol=self.pickle_protocol)))
        }

    cdef inline np.ndarray _deserialize_arr(self, object obj, list callbacks):
        cdef bytes bt

        value = obj['value']

        bt = self._to_bytes(base64.b64decode(value))

        if bt is not None:
            return dataloads(bt)

        return None

    cdef inline dict _serialize_dtype(self, value):
        cdef str v

        if not isinstance(value, ExtensionDtype) and 'V' not in value.str:
            v = value.str
        else:
            v = self._to_str(base64.b64encode(pickle.dumps(value, protocol=self.pickle_protocol)))
        return {
            'type': _get_name(ValueType.dtype),
            'value': v
        }


    cdef inline object _deserialize_dtype(self, object value, list callbacks):
        try:
            return np.dtype(value['value'])
        except TypeError:
            val = value['value']
            return pickle.loads(base64.b64decode(val))

    cdef inline dict _serialize_index(self, value):
        return {
            'type': _get_name(ValueType.index),
            'value': self._to_str(base64.b64encode(datadumps(value, serial_type=self.data_serial_type,
                                                             pickle_protocol=self.pickle_protocol)))
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
            'value': self._to_str(base64.b64encode(datadumps(value, serial_type=self.data_serial_type,
                                                             pickle_protocol=self.pickle_protocol)))
        }

    cdef inline dict _serialize_dataframe(self, value):
        return {
            'type': _get_name(ValueType.dataframe),
            'value': self._to_str(base64.b64encode(datadumps(value, serial_type=self.data_serial_type,
                                                             pickle_protocol=self.pickle_protocol)))
        }

    cdef inline dict _serialize_key(self, value):
        return {
            'type': _get_name(ValueType.key),
            'value': (value.key, value.id)
        }

    cdef inline KeyPlaceholder _deserialize_key(self, object obj, list callbacks):
        return KeyPlaceholder(*obj['value'])

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
                          for val, it_type in zip(value, tp.type)]
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
        bio = BytesIO()
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
            # np.load will return a ndarray
            value = np.load(BytesIO(v))
            return value.dtype.type(value)
        return None

    cdef inline dict _serialize_datetime64(self, value):
        if isinstance(value, pd.Timestamp):
            # convert to np.datetime64
            value = value.to_datetime64()
        return self._serialize_datetime64_timedelta64(value, ValueType.datetime64)

    cdef inline _deserialize_datetime64(self, obj, list callbacks):
        return self._deserialize_datetime64_timedelta64(obj, callbacks)

    cdef inline dict _serialize_timedelta64(self, value):
        if isinstance(value, pd.Timedelta):
            # convert to np.timedelta64
            value = value.to_timedelta64()
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

    cdef inline dict _serialize_namedtuple(self, value):
        return {
            'type': 'namedtuple',
            'value': self._to_str(base64.b64encode(
                cloudpickle.dumps(value, protocol=self.pickle_protocol)))
        }

    cdef inline _deserialize_namedtuple(self, obj, list callbacks):
        return cloudpickle.loads(base64.b64decode(obj['value']))

    cdef inline dict _serialize_regex(self, value):
        return {
            'type': 'regex',
            'value': {
                'pattern': value.pattern,
                'flags': value.flags,
            } if value is not None else None,
        }

    cdef inline _deserialize_regex(self, obj, list callbacks):
        val = obj['value']
        return re.compile(val['pattern'], flags=val['flags']) if val is not None else None

    cdef inline dict _serialize_function(self, value):
        return {
            'type': 'function',
            'value': self._to_str(base64.b64encode(
                serialize_function(value, pickle_protocol=self.pickle_protocol)))
        }

    cdef inline _deserialize_function(self, obj, list callbacks):
        cdef bytes v

        value = obj['value']
        v = base64.b64decode(value)

        if v is not None:
            return cloudpickle.loads(v)
        return None

    cdef inline dict _serialize_tzinfo(self, value):
        return {
            'type': 'tzinfo',
            'value': self._to_str(base64.b64encode(pickle.dumps(value, protocol=self.pickle_protocol)))
        }

    cdef inline _deserialize_tzinfo(self, obj, list callbacks):
        cdef bytes v

        value = obj['value']
        v = base64.b64decode(value)

        if v is not None:
            return pickle.loads(v)
        return None

    cdef inline _serialize_interval_arr(self, value):
        return {
            'type': 'interval_arr',
            'value': {
                'left': self._to_str(base64.b64encode(datadumps(value.left, serial_type=self.data_serial_type,
                                                                pickle_protocol=self.pickle_protocol))),
                'right': self._to_str(base64.b64encode(datadumps(value.right, serial_type=self.data_serial_type,
                                                                 pickle_protocol=self.pickle_protocol))),
                'closed': value.closed,
                'dtype': self._serialize_dtype(value.dtype)
            }
        }

    cdef inline _deserialize_interval_arr(self, obj, list callbacks):
        value = obj['value']

        left = dataloads(self._to_bytes(base64.b64decode(value['left'])))
        right = dataloads(self._to_bytes(base64.b64decode(value['right'])))
        closed = value['closed']
        dtype = self._deserialize_dtype(value['dtype'], callbacks)

        return IntervalArray.from_arrays(left, right, closed=closed, dtype=dtype)

    cdef inline _serialize_freq(self, value):
        return {
            'type': 'freq',
            'value': value.freqstr,
        }

    cdef inline _deserialize_freq(self, obj, list callbacks):
        value = obj['value']
        return to_offset(value)

    cdef inline _serialize_pickled(self, value):
        return {
            'type': 'pickled',
            'value': self._to_str(base64.b64encode(cloudpickle.dumps(value, protocol=self.pickle_protocol))),
        }

    cdef inline _deserialize_pickled(self, obj, list callbacks):
        value = obj['value']
        v = base64.b64decode(value)

        if v is not None:
            return cloudpickle.loads(v)
        return None

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
        elif tp is ValueType.datetime64:
            return self._serialize_datetime64(value)
        elif tp is ValueType.timedelta64:
            return self._serialize_timedelta64(value)
        elif tp is ValueType.function:
            return self._serialize_function(value)
        elif tp is ValueType.tzinfo:
            return self._serialize_tzinfo(value)
        elif tp is ValueType.interval_arr:
            return self._serialize_interval_arr(value)
        elif tp is ValueType.freq:
            return self._serialize_freq(value)
        elif tp is ValueType.namedtuple:
            return self._serialize_namedtuple(value)
        elif tp is ValueType.regex:
            return self._serialize_regex(value)
        elif isinstance(tp, List):
            if not isinstance(value, list):
                value = list(value)
            return self._serialize_list(value, tp, weak_ref=weak_ref)
        elif isinstance(tp, Tuple):
            return self._serialize_tuple(value, tp, weak_ref=weak_ref)
        elif isinstance(tp, Dict):
            return self._serialize_dict(value, tp, weak_ref=weak_ref)
        else:
            raise TypeError(f'Unknown type to serialize: {tp}')

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
        elif isinstance(value, int):
            return value
        elif isinstance(value, float):
            return value
        elif isinstance(value, complex):
            return self._serialize_complex(value, ValueType.complex128)
        elif isinstance(value, slice):
            return self._serialize_slice(value)
        elif isinstance(value, np.ndarray):
            return self._serialize_arr(value)
        elif isinstance(value, (np.dtype, ExtensionDtype)):
            return self._serialize_dtype(value)
        elif isinstance(value, pd.Index):
            return self._serialize_index(value)
        elif isinstance(value, pd.Series):
            return self._serialize_series(value)
        elif isinstance(value, pd.DataFrame):
            return self._serialize_dataframe(value)
        elif isinstance(value, HasKey):
            return self._serialize_key(value)
        elif isinstance(value, HasData):
            return self._serialize_key(value.data)
        elif isinstance(value, list):
            return self._serialize_list(value, tp=None, weak_ref=weak_ref)
        elif isinstance(value, tuple):
            if hasattr(value, '_fields'):
                # namedtuple, use cloudpickle to serialize
                return self._serialize_namedtuple(value)
            return self._serialize_tuple(value, tp=None, weak_ref=weak_ref)
        elif isinstance(value, dict):
            return self._serialize_dict(value, tp=None, weak_ref=weak_ref)
        elif isinstance(value, (np.datetime64, pd.Timestamp)):
            return self._serialize_datetime64(value)
        elif isinstance(value, (np.timedelta64, pd.Timedelta)):
            return self._serialize_timedelta64(value)
        elif isinstance(value, np.number):
            return self._serialize_untyped_value(value.item())
        elif isinstance(value, tzinfo):
            return self._serialize_tzinfo(value)
        elif isinstance(value, IntervalArray):
            return self._serialize_interval_arr(value)
        elif isinstance(value, RE_Pattern):
            return self._serialize_regex(value)
        elif PDTick is not None and isinstance(value, PDTick):
            return self._serialize_freq(value)
        elif callable(value):
            return self._serialize_function(value)
        else:
            try:
                return self._serialize_pickled(value)
            except:
                raise TypeError(f'Unknown type to serialize: {type(value)}') from None

    cdef inline object _serialize_value(self, value, tp=None, bint weak_ref=False):
        if tp is None:
            return self._serialize_untyped_value(value, weak_ref=weak_ref)
        else:
            return self._serialize_typed_value(value, tp, weak_ref=weak_ref)

    cdef inline _on_serial(self, Field field, x):
        x = x if x is not None else field.default
        return field.on_serialize(x) if field.on_serialize is not None else x

    cdef object _serialize_reference(self, tag, model, value, new_obj):
        if value is None:
            return None
        if model is None:
            new_obj['type_id'] = value.__serializable_index__
            new_obj['value'] = value.serialize(self, dict())
        else:
            if not isinstance(value, model):
                raise TypeError(f'Does not match type for reference field {tag}: '
                                f'expect {model}, got {type(value)}')
            value.serialize(self, new_obj)
        return new_obj

    cpdef serialize_field(self, Field field, model_instance, obj):
        cdef str tag
        cdef object new_obj
        cdef bint has_val
        cdef object value
        cdef OneOfField oneoffield

        if isinstance(field, ReferenceField):
            tag = field.tag_name(self)
            if hasattr(model_instance, field.attr):
                field_val = getattr(model_instance, field.attr)
                if field.weak_ref:
                    field_val = field_val()
                if field_val is not None:
                    new_obj = obj[tag] = dict()
                    value = self._on_serial(field, field_val)
                    self._serialize_reference(tag, field.type.model, value, new_obj)
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
                        raise ValueError(f'Only one of attributes({field.attrs}) can be specified')

                    new_obj = obj[tag] = dict()
                    value.serialize(self, new_obj)
                    return
            if not has_val and value is not None:
                raise ValueError(f'Value {value} cannot match any type for OneOfField `{field.tag_name(self)}`')
        elif isinstance(field, ListField) and type(field.type.type) == Reference:
            tag = field.tag_name(self)
            value = self._on_serial(field, getattr(model_instance, field.attr, None))
            if value is None:
                return
            new_obj = obj[tag] = list()
            for val in value:
                if field.weak_ref:
                    val = val()
                new_obj.append(self._serialize_reference(tag, field.type.type.model, val, dict()))
        else:
            tag = field.tag_name(self)
            try:
                val = self._on_serial(field, getattr(model_instance, field.attr, None))
            except (AttributeError, TypeError):
                tp, err, tb = sys.exc_info()
                raise tp(f'Fail to serialize field `{tag}` for {model_instance}, reason: {err}') \
                    .with_traceback(tb) from err
            if val is None:
                return
            try:
                obj[tag] = self._serialize_value(val, field.type, weak_ref=field.weak_ref)
            except (TypeError, ValueError):
                tp, err, tb = sys.exc_info()
                raise tp(f'Fail to serialize field `{tag}` for {model_instance}, reason: {err}') \
                    .with_traceback(tb) from err

    cdef inline _deserialize_value(self, obj, list callbacks, bint weak_ref):
        if not isinstance(obj, dict):
            return obj

        ref = lambda x: weakref.ref(x) if weak_ref else x
        tp = _get_type(obj['type'])

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
        elif tp is ValueType.function:
            return ref(self._deserialize_function(obj, callbacks))
        elif tp is ValueType.tzinfo:
            return ref(self._deserialize_tzinfo(obj, callbacks))
        elif tp is ValueType.interval_arr:
            return ref(self._deserialize_interval_arr(obj, callbacks))
        elif tp is ValueType.freq:
            return ref(self._deserialize_freq(obj, callbacks))
        elif tp is ValueType.list:
            return self._deserialize_list(obj, callbacks, weak_ref)
        elif tp is ValueType.tuple:
            return self._deserialize_tuple(obj, callbacks, weak_ref)
        elif tp is ValueType.dict:
            return self._deserialize_dict(obj, callbacks, weak_ref)
        elif tp is ValueType.namedtuple:
            return self._deserialize_namedtuple(obj, callbacks)
        elif tp is ValueType.regex:
            return self._deserialize_regex(obj, callbacks)
        elif tp is ValueType.pickled:
            return self._deserialize_pickled(obj, callbacks)
        else:
            raise TypeError(f'Unknown type to deserialize {obj["type"]}')

    cdef inline _on_deserial(self, Field field, x):
        x = x if x is not None else field.default
        return field.on_deserialize(x) if field.on_deserialize is not None else x

    cdef object _deserialize_reference(self, model, val, list callbacks, dict key_to_instance):
        if model is None:
            model = get_serializable_by_index(val['type_id'])
            if model is None:
                raise KeyError(f'Cannot find serializable class for type_id {val["type_id"]}')
            val = val['value']
        return model.deserialize(self, val, callbacks, key_to_instance)

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
                    self._on_deserial(field, self._deserialize_reference(
                        field.type.model, val, callbacks, key_to_instance)))
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
                            f'Only one of attributes({field.attrs}) can be specified')

                    setattr(model_instance, f.attr,
                            self._on_deserial(field, f.type.model.deserialize(self, obj[tag],
                                                                              callbacks, key_to_instance)))
        elif isinstance(field, ListField) and type(field.type.type) == Reference:
            tag = field.tag_name(self)
            if tag not in obj:
                return
            setattr(model_instance, field.attr,
                    self._on_deserial(
                        field, [self._deserialize_reference(field.type.type.model, it_obj, callbacks, key_to_instance)
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

