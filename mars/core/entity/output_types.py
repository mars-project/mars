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

import functools
from enum import Enum

from .fuse import FUSE_CHUNK_TYPE
from .objects import OBJECT_TYPE, OBJECT_CHUNK_TYPE


class OutputType(Enum):
    object = 1
    tensor = 2
    dataframe = 3
    series = 4
    index = 5
    scalar = 6
    categorical = 7
    dataframe_groupby = 8
    series_groupby = 9

    @classmethod
    def serialize_list(cls, output_types):
        return [ot.value for ot in output_types] if output_types is not None else None

    @classmethod
    def deserialize_list(cls, output_types):
        return [cls(ot) for ot in output_types] if output_types is not None else None


_OUTPUT_TYPE_TO_CHUNK_TYPES = {OutputType.object: OBJECT_CHUNK_TYPE}
_OUTPUT_TYPE_TO_TILEABLE_TYPES = {OutputType.object: OBJECT_TYPE}
_OUTPUT_TYPE_TO_FETCH_CLS = {}


def register_output_types(output_type, tileable_types, chunk_types):
    _OUTPUT_TYPE_TO_TILEABLE_TYPES[output_type] = tileable_types
    _OUTPUT_TYPE_TO_CHUNK_TYPES[output_type] = chunk_types


def register_fetch_class(output_type, fetch_cls, fetch_shuffle_cls):
    _OUTPUT_TYPE_TO_FETCH_CLS[output_type] = (fetch_cls, fetch_shuffle_cls)


def get_tileable_types(output_type):
    return _OUTPUT_TYPE_TO_TILEABLE_TYPES[output_type]


def get_chunk_types(output_type):
    return _OUTPUT_TYPE_TO_CHUNK_TYPES[output_type]


def get_fetch_class(output_type):
    return _OUTPUT_TYPE_TO_FETCH_CLS[output_type]


@functools.lru_cache(100)
def _get_output_type_by_cls(cls):
    for tp in OutputType.__members__.values():
        try:
            tileable_types = _OUTPUT_TYPE_TO_TILEABLE_TYPES[tp]
            chunk_types = _OUTPUT_TYPE_TO_CHUNK_TYPES[tp]
            if issubclass(cls, (tileable_types, chunk_types)):
                return tp
        except KeyError:  # pragma: no cover
            continue
    raise TypeError('Output can only be tensor, dataframe or series')


def get_output_types(*objs, unknown_as=None):
    output_types = []
    for obj in objs:
        if obj is None:
            continue
        elif isinstance(obj, FUSE_CHUNK_TYPE):
            obj = obj.chunk

        try:
            output_types.append(_get_output_type_by_cls(type(obj)))
        except TypeError:
            if unknown_as is not None:
                output_types.append(unknown_as)
            else:  # pragma: no cover
                raise
    return output_types
