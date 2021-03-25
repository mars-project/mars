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

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Type

import numpy as np
import pandas as pd

from ...core import OBJECT_TYPE, OBJECT_CHUNK_TYPE
from ...dataframe.core import DtypesValue, IndexValue, \
    DATAFRAME_TYPE, SERIES_TYPE, INDEX_TYPE, \
    DATAFRAME_CHUNK_TYPE, SERIES_CHUNK_TYPE, INDEX_CHUNK_TYPE
from ...tensor.core import TensorOrder, TENSOR_TYPE, TENSOR_CHUNK_TYPE

PandasDtypeType = Union[np.dtype, pd.api.extensions.ExtensionDtype]

_type_to_meta_class = dict()


def _register_type(object_types: Tuple):
    def _call(meta):
        for obj_type in object_types:
            _type_to_meta_class[obj_type] = meta
        return meta
    return _call


def get_meta_type(object_type: Type) -> Type["_CommonMeta"]:
    try:
        return _type_to_meta_class[object_type]
    except KeyError:
        for m_type in object_type.__mro__:
            try:
                return _type_to_meta_class[m_type]
            except KeyError:
                continue
        raise


@dataclass
class _CommonMeta:
    """
    Class for common meta, for both tileable and chunk, or DataFrame, tensor etc.
    """
    object_id: str
    name: Any = None
    memory_size: int = None # size in memory
    store_size: int = None  # size that stored in storage
    extra: Dict = None


@dataclass
class _TileableMeta(_CommonMeta):
    nsplits: Tuple[Tuple[int]] = None


@_register_type(TENSOR_TYPE)
@dataclass
class TensorMeta(_TileableMeta):
    shape: Tuple[int] = None
    dtype: np.dtype = None
    order: TensorOrder = None


@_register_type(DATAFRAME_TYPE)
@dataclass
class DataFrameMeta(_TileableMeta):
    shape: Tuple[int] = None
    dtypes_value: DtypesValue = None
    index_value: IndexValue = None


@_register_type(SERIES_TYPE)
@dataclass
class SeriesMeta(_TileableMeta):
    shape: Tuple[int] = None
    dtype: PandasDtypeType = None
    index_value: IndexValue = None


@_register_type(INDEX_TYPE)
@dataclass
class IndexMeta(_TileableMeta):
    shape: Tuple[int] = None
    dtype: PandasDtypeType = None
    index_value: IndexValue = None


@_register_type(OBJECT_TYPE)
@dataclass
class ObjectMeta(_TileableMeta):
    pass


@_register_type(OBJECT_CHUNK_TYPE)
@dataclass
class _ChunkMeta(_CommonMeta):
    index: Tuple[int] = None
    bands: List[Tuple[str, str]] = None


@_register_type(TENSOR_CHUNK_TYPE)
@dataclass
class TensorChunkMeta(_ChunkMeta):
    shape: Tuple[int] = None
    dtype: np.dtype = None
    order: TensorOrder = None


@_register_type(DATAFRAME_CHUNK_TYPE)
@dataclass
class DataFrameChunkMeta(_ChunkMeta):
    shape: Tuple[int] = None
    dtypes_value: DtypesValue = None
    index_value: IndexValue = None


@_register_type(SERIES_CHUNK_TYPE)
@dataclass
class SeriesChunkMeta(_ChunkMeta):
    shape: Tuple[int] = None
    dtype: PandasDtypeType = None
    index_value: IndexValue = None


@_register_type(INDEX_CHUNK_TYPE)
@dataclass
class IndexChunkMeta(_ChunkMeta):
    shape: Tuple[int] = None
    dtype: PandasDtypeType = None
    index_value: IndexValue = None


@_register_type(OBJECT_CHUNK_TYPE)
@dataclass
class ObjectChunkMeta(_ChunkMeta):
    pass
