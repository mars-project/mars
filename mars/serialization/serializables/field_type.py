# Copyright 1999-2021 Alibaba Group Holding Ltd.
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

from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime, timedelta, tzinfo
from enum import Enum
from typing import Tuple, Type

import numpy as np
import pandas as pd

from ...utils import lazy_import

cupy = lazy_import('cupy', globals=globals())
cudf = lazy_import('cudf', globals=globals())


class PrimitiveType(Enum):
    bool = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    uint8 = 6
    uint16 = 7
    uint32 = 8
    uint64 = 9
    float16 = 10
    float32 = 11
    float64 = 12
    bytes = 13
    string = 14
    complex64 = 24
    complex128 = 25


_primitive_type_to_valid_types = {
    PrimitiveType.bool: (bool, np.bool_),
    PrimitiveType.int8: (int, np.int8),
    PrimitiveType.int16: (int, np.int16),
    PrimitiveType.int32: (int, np.int32),
    PrimitiveType.int64: (int, np.int64),
    PrimitiveType.uint8: (int, np.uint8),
    PrimitiveType.uint16: (int, np.uint16),
    PrimitiveType.uint32: (int, np.uint32),
    PrimitiveType.uint64: (int, np.uint64),
    PrimitiveType.float16: (float, np.float16),
    PrimitiveType.float32: (float, np.float32),
    PrimitiveType.float64: (float, np.float64),
    PrimitiveType.bytes: (bytes, np.bytes_),
    PrimitiveType.string: (str, np.unicode_),
    PrimitiveType.complex64: (complex, np.complex64),
    PrimitiveType.complex128: (complex, np.complex128),
}


class AbstractFieldType(ABC):
    __slots__ = ()

    @property
    @abstractmethod
    def type_name(self) -> str:
        """
        Type name.

        Returns
        -------
        type_name : str
        """

    @property
    def name(self) -> str:
        """
        Name of field type instance.

        Returns
        -------
        name : str
        """
        return self.type_name.capitalize()

    @property
    @abstractmethod
    def valid_types(self) -> Tuple[Type, ...]:
        """
        Valid types.

        Returns
        -------
        valid_types: tuple
            Valid types.
        """

    def validate(self, value):
        if value is not None and not isinstance(value, self.valid_types):
            raise TypeError(f'value needs to be instance '
                            f'of {self.valid_types}, got {type(value)}')

    def __call__(self, *args, **kwargs):
        return type(self)(*args, **kwargs)


class SingletonFieldType(AbstractFieldType, metaclass=ABCMeta):
    __slots__ = ()

    _instance = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            inst = super().__new__(cls, *args, **kw)
            cls._instance = inst
        return cls._instance


class PrimitiveFieldType(AbstractFieldType):
    __slots__ = 'type',

    _type_to_instances = dict()

    def __new__(cls, *args, **kwargs):
        primitive_type = args[0]
        try:
            return cls._type_to_instances[primitive_type]
        except KeyError:
            inst = cls._type_to_instances[primitive_type] = \
                super().__new__(cls)
            return inst

    def __init__(self, primitive_type: PrimitiveType):
        self.type = primitive_type

    @property
    def type_name(self) -> str:
        return self.type.name

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        return _primitive_type_to_valid_types[self.type]


class SliceType(SingletonFieldType):
    __slots__ = ()

    @property
    def type_name(self) -> str:
        return 'slice'

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        return slice,


class NDArrayType(SingletonFieldType):
    __slots__ = ()

    @property
    def type_name(self) -> str:
        return 'ndarray'

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        if cupy is None:
            return np.ndarray,
        else:
            return np.ndarray, cupy.ndarray


class DtypeType(SingletonFieldType):
    __slots__ = ()

    @property
    def type_name(self) -> str:
        return 'dtype'

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        return np.dtype, pd.api.extensions.ExtensionDtype


class KeyType(SingletonFieldType):
    __slots__ = ()

    @property
    def type_name(self) -> str:
        return 'dtype'

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        from ...core.entity import ENTITY_TYPE
        return ENTITY_TYPE


class DatetimeType(SingletonFieldType):
    __slots__ = ()

    @property
    def type_name(self) -> str:
        return 'datetime'

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        return datetime, pd.Timestamp


class TimedeltaType(SingletonFieldType):
    __slots__ = ()

    @property
    def type_name(self) -> str:
        return 'timedelta'

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        return timedelta, pd.Timedelta


class IndexType(SingletonFieldType):
    __slots__ = ()

    @property
    def type_name(self) -> str:
        return 'index'

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        if cudf is None:
            return pd.Index,
        else:
            return pd.Index, cudf.Index


class SeriesType(SingletonFieldType):
    __slots__ = ()

    @property
    def type_name(self) -> str:
        return 'series'

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        if cudf is None:
            return pd.Series,
        else:
            return pd.Series, cudf.Series


class DataFrameType(SingletonFieldType):
    __slots__ = ()

    @property
    def type_name(self) -> str:
        return 'dataframe'

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        if cudf is None:
            return pd.DataFrame,
        else:
            return pd.DataFrame, cudf.DataFrame


class FunctionType(SingletonFieldType):
    __slots__ = ()

    @property
    def type_name(self) -> str:
        return 'function'

    @property
    def valid_types(self) -> Tuple[Type, ...]:  # pragma: no cover
        return ()

    def validate(self, value):
        if value is not None and not callable(value):
            raise TypeError(f'value should be a function, '
                            f'got {type(value)}')


class NamedtupleType(SingletonFieldType):
    __slots__ = ()

    @property
    def type_name(self) -> str:
        return 'namedtuple'

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        return tuple,

    def validate(self, value):
        if not (isinstance(value, self.valid_types) and hasattr(value, '_fields')):
            raise TypeError(f'value should be instance of namedtuple, '
                            f'got {type(value)}')


class TZInfoType(SingletonFieldType):
    __slots__ = ()

    @property
    def type_name(self) -> str:
        return 'tzinfo'

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        return tzinfo,


class IntervalArrayType(SingletonFieldType):
    __slots__ = ()

    @property
    def type_name(self) -> str:
        return 'interval_array'

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        return pd.arrays.IntervalArray,


class AnyType(SingletonFieldType):
    __slots__ = ()

    @property
    def type_name(self) -> str:
        return 'any'

    @property
    def valid_types(self) -> Tuple[Type, ...]:  # pragma: no cover
        return ()

    def validate(self, value):
        # any type is valid
        return


class _CollectionType(AbstractFieldType, metaclass=ABCMeta):
    __slots__ = '_field_types',

    def __init__(self, *field_types):
        self._field_types = field_types
        if len(field_types) == 0:
            self._field_types = (AnyType(), Ellipsis)

    @property
    def name(self) -> str:
        base_name = super().name
        if self.is_homogeneous():
            if isinstance(self._field_types[0], AnyType):
                return base_name
            else:
                return f'{base_name}[{self._field_types[0].name}, ...]'
        else:
            field_type_names = ', '.join([ft.name for ft in self._field_types])
            return f'{base_name}[{field_type_names}]'

    def is_homogeneous(self):
        return len(self._field_types) == 1 or \
               (len(self._field_types) == 2 and self._field_types[1] is Ellipsis)

    def validate(self, value):
        if value is None:
            return
        if not isinstance(value, self.valid_types):
            raise TypeError(f'value should be instance of {self.valid_types}, '
                            f'got {type(value)}')
        if self.is_homogeneous():
            field_type: AbstractFieldType = self._field_types[0]
            if not isinstance(field_type, AnyType):
                for item in value:
                    try:
                        field_type.validate(item)
                    except TypeError:
                        raise TypeError(f'item should be instance of '
                                        f'{field_type.valid_types}, '
                                        f'got {type(item)}')
        else:
            if len(value) != len(self._field_types):
                raise ValueError(f'value should own {len(self._field_types)} items, '
                                 f'got {len(value)} items')
            for expect_field_type, item in zip(self._field_types, value):
                try:
                    expect_field_type.validate(item)
                except TypeError:
                    raise TypeError(f'item should be instance of '
                                    f'{expect_field_type.valid_types}, '
                                    f'got {type(item)}')


class ListType(_CollectionType):
    __slots__ = ()

    @property
    def type_name(self) -> str:
        return 'list'

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        return list,


class TupleType(_CollectionType):
    __slots__ = ()

    @property
    def type_name(self) -> str:
        return 'tuple'

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        return tuple,


class DictType(AbstractFieldType):
    __slots__ = 'key_type', 'value_type'

    key_type: AbstractFieldType
    value_type: AbstractFieldType

    def __init__(self,
                 key_type: AbstractFieldType = None,
                 value_type: AbstractFieldType = None):
        if key_type is None:
            key_type = AnyType()
        if value_type is None:
            value_type = AnyType()
        self.key_type = key_type
        self.value_type = value_type

    @property
    def type_name(self) -> str:
        return 'dict'

    @property
    def name(self) -> str:
        if isinstance(self.key_type, AnyType) and isinstance(self.value_type, AnyType):
            return 'Dict'
        else:
            return f'Dict[{self.key_type.name}, {self.value_type.name}]'

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        return dict,

    def validate(self, value):
        super().validate(value)
        if value is None:
            return
        for k, v in value.items():
            try:
                self.key_type.validate(k)
            except TypeError:
                raise TypeError(f'key should be instance of '
                                f'{self.key_type.valid_types}, got {type(k)}')
            try:
                self.value_type.validate(v)
            except TypeError:
                raise TypeError(f'value should be instance of '
                                f'{self.value_type.valid_types}, got {type(v)}')


class ReferenceType(AbstractFieldType):
    __slots__ = 'reference_type',

    reference_type: Type

    def __init__(self, reference_type: Type = None):
        if reference_type is None:
            reference_type = object
        self.reference_type = reference_type

    @property
    def type_name(self) -> str:
        return 'reference'

    @property
    def valid_types(self) -> Tuple[Type, ...]:
        return self.reference_type,


class FieldTypes:
    # primitive type
    bool = PrimitiveFieldType(PrimitiveType.bool)
    int8 = PrimitiveFieldType(PrimitiveType.int8)
    int16 = PrimitiveFieldType(PrimitiveType.int16)
    int32 = PrimitiveFieldType(PrimitiveType.int32)
    int64 = PrimitiveFieldType(PrimitiveType.int64)
    uint8 = PrimitiveFieldType(PrimitiveType.uint8)
    uint16 = PrimitiveFieldType(PrimitiveType.uint16)
    uint32 = PrimitiveFieldType(PrimitiveType.uint32)
    uint64 = PrimitiveFieldType(PrimitiveType.uint64)
    float16 = PrimitiveFieldType(PrimitiveType.float16)
    float32 = PrimitiveFieldType(PrimitiveType.float32)
    float64 = PrimitiveFieldType(PrimitiveType.float64)
    complex64 = PrimitiveFieldType(PrimitiveType.complex64)
    complex128 = PrimitiveFieldType(PrimitiveType.complex128)
    bytes = PrimitiveFieldType(PrimitiveType.bytes)
    string = PrimitiveFieldType(PrimitiveType.string)

    key = KeyType()

    # Python types
    slice = SliceType()
    datetime = DatetimeType()
    # alias of datetime
    datatime64 = DatetimeType()
    timedelta = TimedeltaType()
    # alias of timedelta
    timedelta64 = TimedeltaType()
    tzinfo = TZInfoType()
    function = FunctionType()
    namedtuple = NamedtupleType()
    reference = ReferenceType()
    any = AnyType()
    # equivalent to any
    pickled = AnyType()

    # collection
    list = ListType()
    tuple = TupleType()
    dict = DictType()

    # numpy
    ndarray = NDArrayType()
    # alias of ndarray
    arr = NDArrayType()
    dtype = DtypeType()

    # pandas
    index = IndexType()
    series = SeriesType()
    dataframe = DataFrameType()
    interval_array = IntervalArrayType()
    # alias of interval_array
    interval_arr = IntervalArrayType()
