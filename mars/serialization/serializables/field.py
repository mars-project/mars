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

import itertools
import importlib
import inspect
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Callable, Optional, Type, Union

from ...utils import no_default, _is_ci
from .field_type import (
    AbstractFieldType,
    FieldTypes,
    ListType,
    TupleType,
    DictType,
    ReferenceType,
)


class Field(ABC):
    __slots__ = (
        "_tag",
        "_default_value",
        "_default_factory",
        "_on_serialize",
        "_on_deserialize",
        "name",  # The __name__ of member_descriptor
        "get",  # The __get__ of member_descriptor
        "set",  # The __set__ of member_descriptor
        "__delete__",  # The __delete__ of member_descriptor
    )

    _tag: str
    _default_value: Any
    _default_factory: Optional[Callable]

    def __init__(
        self,
        tag: str,
        default: Any = no_default,
        default_factory: Optional[Callable] = None,
        on_serialize: Callable[[Any], Any] = None,
        on_deserialize: Callable[[Any], Any] = None,
    ):
        if (
            default is not no_default and default_factory is not None
        ):  # pragma: no cover
            raise ValueError("default and default_factory can not be specified both")

        self._tag = tag
        self._default_value = default
        self._default_factory = default_factory
        self._on_serialize = on_serialize
        self._on_deserialize = on_deserialize

    @property
    def tag(self):
        return self._tag

    @property
    def on_serialize(self):
        return self._on_serialize

    @property
    def on_deserialize(self):
        return self._on_deserialize

    @property
    @abstractmethod
    def field_type(self) -> AbstractFieldType:
        """
        Field type.

        Returns
        -------
        field_type : AbstractFieldType
             Field type.
        """

    def __get__(self, instance, owner=None):
        try:
            return self.get(instance, owner)
        except AttributeError:
            if self._default_value is not no_default:
                val = self._default_value
                self.set(instance, val)
                return val
            elif self._default_factory is not None:
                val = self._default_factory()
                self.set(instance, val)
                return val
            else:
                raise

    def __set__(self, instance, value) -> None:
        if _is_ci:  # pragma: no branch
            from ...core import is_kernel_mode

            if not is_kernel_mode():
                field_type = self.field_type
                try:
                    to_check_value = value
                    if to_check_value is not None and self._on_serialize:
                        to_check_value = self._on_serialize(to_check_value)
                    field_type.validate(to_check_value)
                except (TypeError, ValueError) as e:
                    raise type(e)(
                        f"Failed to set `{self.name}` for {type(instance).__name__} "
                        f"when environ CI=true is set: {str(e)}"
                    )
        self.set(instance, value)


class AnyField(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.any


class IdentityField(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.string


class BoolField(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.bool


class Int8Field(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.int8


class Int16Field(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.int16


class Int32Field(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.int32


class Int64Field(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.int64


class UInt8Field(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.uint8


class UInt16Field(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.uint16


class UInt32Field(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.uint32


class UInt64Field(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.uint64


class Float16Field(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.float16


class Float32Field(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.float32


class Float64Field(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.float64


class Complex64Field(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.complex64


class Complex128Field(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.complex128


class StringField(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.string


class BytesField(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.bytes


class KeyField(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.key


class NDArrayField(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.ndarray


class Datetime64Field(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.datetime


class Timedelta64Field(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.timedelta


class DataTypeField(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.dtype


class IndexField(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.index


class SeriesField(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.series


class DataFrameField(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.dataframe


class SliceField(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.slice


class FunctionField(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.function


class NamedTupleField(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.namedtuple


class TZInfoField(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.tzinfo


class IntervalArrayField(Field):
    __slots__ = ()

    @property
    def field_type(self) -> AbstractFieldType:
        return FieldTypes.interval_array


class _CollectionField(Field, metaclass=ABCMeta):
    __slots__ = ("_field_type",)

    def __init__(
        self,
        tag: str,
        field_type: AbstractFieldType = None,
        default: Any = no_default,
        default_factory: Optional[Callable] = None,
        on_serialize: Callable[[Any], Any] = None,
        on_deserialize: Callable[[Any], Any] = None,
    ):
        super().__init__(
            tag,
            default=default,
            default_factory=default_factory,
            on_serialize=on_serialize,
            on_deserialize=on_deserialize,
        )
        if field_type is None:
            field_type = FieldTypes.any
        collection_type = self._collection_type()
        if not isinstance(field_type, collection_type):
            field_type = collection_type(field_type, ...)
        self._field_type = field_type

    @classmethod
    @abstractmethod
    def _collection_type(cls) -> AbstractFieldType:
        """
        Collection type.

        Returns
        -------
        collection_type
        """

    @property
    def field_type(self) -> Type[AbstractFieldType]:
        return self._field_type


class ListField(_CollectionField):
    __slots__ = ()

    @classmethod
    def _collection_type(cls) -> Type[AbstractFieldType]:
        return ListType


class TupleField(_CollectionField):
    __slots__ = ()

    @classmethod
    def _collection_type(cls) -> Type[AbstractFieldType]:
        return TupleType


class DictField(Field):
    __slots__ = ("_field_type",)

    def __init__(
        self,
        tag: str,
        key_type: AbstractFieldType = None,
        value_type: AbstractFieldType = None,
        default: Any = no_default,
        default_factory: Optional[Callable] = None,
        on_serialize: Callable[[Any], Any] = None,
        on_deserialize: Callable[[Any], Any] = None,
    ):
        super().__init__(
            tag,
            default=default,
            default_factory=default_factory,
            on_serialize=on_serialize,
            on_deserialize=on_deserialize,
        )
        self._field_type = DictType(key_type, value_type)

    @property
    def field_type(self) -> AbstractFieldType:
        return self._field_type


class ReferenceField(Field):
    __slots__ = "_reference_type", "_field_type"

    def __init__(
        self,
        tag: str,
        reference_type: Union[str, Type] = None,
        default: Any = no_default,
        on_serialize: Callable[[Any], Any] = None,
        on_deserialize: Callable[[Any], Any] = None,
    ):
        super().__init__(
            tag,
            default=default,
            on_serialize=on_serialize,
            on_deserialize=on_deserialize,
        )
        self._reference_type = reference_type

        if not isinstance(reference_type, str):
            self._field_type = ReferenceType(reference_type)
        else:
            # need to bind dynamically
            self._field_type = None

    @property
    def field_type(self) -> AbstractFieldType:
        return self._field_type

    def get_field_type(self, instance):
        if self._field_type is None:
            # bind dynamically
            if self._reference_type == "self":
                reference_type = type(instance)
            elif isinstance(self._reference_type, str) and "." in self._reference_type:
                module, name = self._reference_type.rsplit(".", 1)
                reference_type = getattr(importlib.import_module(module), name)
            else:
                module = inspect.getmodule(instance)
                reference_type = getattr(module, self._reference_type)
            self._field_type = ReferenceType(reference_type)
        return self._field_type

    def __set__(self, instance, value):
        if _is_ci:
            from ...core import is_kernel_mode

            if not is_kernel_mode():
                field_type = self.get_field_type(instance)
                try:
                    to_check_value = value
                    if to_check_value is not None and self._on_serialize:
                        to_check_value = self._on_serialize(to_check_value)
                    field_type.validate(to_check_value)
                except (TypeError, ValueError) as e:
                    raise type(e)(
                        f"Failed to set `{self.name}` for {type(instance).__name__} "
                        f"when environ CI=true is set: {e}"
                    )
        self.set(instance, value)


class OneOfField(Field):
    __slots__ = "_reference_fields"

    def __init__(
        self,
        tag: str,
        default: Any = no_default,
        on_serialize: Callable[[Any], Any] = None,
        on_deserialize: Callable[[Any], Any] = None,
        **tag_to_reference_types,
    ):
        super().__init__(
            tag,
            default=default,
            on_serialize=on_serialize,
            on_deserialize=on_deserialize,
        )
        self._reference_fields = [
            ReferenceField(t, ref_type)
            for t, ref_type in tag_to_reference_types.items()
        ]

    @property
    def reference_fields(self):
        return self._reference_fields

    @property
    def field_type(self) -> AbstractFieldType:  # pragma: no cover
        # takes no effect here, just return AnyType()
        # we will do check in __set__ instead
        return FieldTypes.any

    def __set__(self, instance, value):
        if not _is_ci:  # pragma: no cover
            return self.set(instance, value)

        for reference_field in self._reference_fields:
            try:
                to_check_value = value
                if to_check_value is not None and self._on_serialize:
                    to_check_value = self._on_serialize(to_check_value)
                reference_field.get_field_type(instance).validate(to_check_value)
                self.set(instance, value)
                return
            except TypeError:
                continue
        valid_types = list(
            itertools.chain(
                *[
                    r.get_field_type(instance).valid_types
                    for r in self._reference_fields
                ]
            )
        )
        raise TypeError(
            f"Failed to set `{self.name}` for {type(instance).__name__} "
            f"when environ CI=true is set: type of instance cannot match any "
            f"of {valid_types}, got {type(value).__name__}"
        )
