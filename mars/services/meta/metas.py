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

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np

from ...core import OBJECT_TYPE, OBJECT_CHUNK_TYPE
from ...dataframe.core import (
    DtypesValue,
    IndexValue,
    DATAFRAME_TYPE,
    SERIES_TYPE,
    INDEX_TYPE,
    DATAFRAME_GROUPBY_TYPE,
    SERIES_GROUPBY_TYPE,
    DATAFRAME_CHUNK_TYPE,
    SERIES_CHUNK_TYPE,
    INDEX_CHUNK_TYPE,
    DATAFRAME_GROUPBY_CHUNK_TYPE,
    SERIES_GROUPBY_CHUNK_TYPE,
    CATEGORICAL_TYPE,
    CATEGORICAL_CHUNK_TYPE,
    DATAFRAME_OR_SERIES_TYPE,
    DATAFRAME_OR_SERIES_CHUNK_TYPE,
)
from ...tensor.core import TensorOrder, TENSOR_TYPE, TENSOR_CHUNK_TYPE
from ...utils import dataslots
from .core import (
    _TileableMeta,
    _ChunkMeta,
    PandasDtypeType,
    register_meta_type,
)


"""
Create a separate module for metas to avoid direct
dependency on mars.dataframe
"""


@register_meta_type(TENSOR_TYPE)
@dataslots
@dataclass
class TensorMeta(_TileableMeta):
    shape: Tuple[int] = None
    dtype: np.dtype = None
    order: TensorOrder = None


@register_meta_type(DATAFRAME_TYPE)
@dataslots
@dataclass
class DataFrameMeta(_TileableMeta):
    shape: Tuple[int] = None
    dtypes_value: DtypesValue = None
    index_value: IndexValue = None


@register_meta_type(SERIES_TYPE)
@dataslots
@dataclass
class SeriesMeta(_TileableMeta):
    shape: Tuple[int] = None
    dtype: PandasDtypeType = None
    index_value: IndexValue = None


@register_meta_type(INDEX_TYPE)
@dataslots
@dataclass
class IndexMeta(_TileableMeta):
    shape: Tuple[int] = None
    dtype: PandasDtypeType = None
    index_value: IndexValue = None


@register_meta_type(DATAFRAME_GROUPBY_TYPE)
@dataslots
@dataclass
class DataFrameGroupByMeta(_TileableMeta):
    shape: Tuple[int] = None
    dtypes_value: DtypesValue = None
    index_value: IndexValue = None
    selection: List = None


@register_meta_type(SERIES_GROUPBY_TYPE)
@dataslots
@dataclass
class SeriesGroupByMeta(_TileableMeta):
    shape: Tuple[int] = None
    dtype: PandasDtypeType = None
    index_value: IndexValue = None
    selection: List = None


@register_meta_type(CATEGORICAL_TYPE)
@dataslots
@dataclass
class CategoricalMeta(_TileableMeta):
    shape: Tuple[int] = None
    dtype: PandasDtypeType = None
    categories_value: IndexValue = None


@register_meta_type(OBJECT_TYPE)
@dataslots
@dataclass
class ObjectMeta(_TileableMeta):
    pass


@register_meta_type(TENSOR_CHUNK_TYPE)
@dataslots
@dataclass
class TensorChunkMeta(_ChunkMeta):
    shape: Tuple[int] = None
    dtype: np.dtype = None
    order: TensorOrder = None


@register_meta_type(DATAFRAME_CHUNK_TYPE)
@dataslots
@dataclass
class DataFrameChunkMeta(_ChunkMeta):
    shape: Tuple[int] = None
    dtypes_value: DtypesValue = None
    index_value: IndexValue = None


@register_meta_type(SERIES_CHUNK_TYPE)
@dataslots
@dataclass
class SeriesChunkMeta(_ChunkMeta):
    shape: Tuple[int] = None
    dtype: PandasDtypeType = None
    index_value: IndexValue = None


@register_meta_type(INDEX_CHUNK_TYPE)
@dataslots
@dataclass
class IndexChunkMeta(_ChunkMeta):
    shape: Tuple[int] = None
    dtype: PandasDtypeType = None
    index_value: IndexValue = None


@register_meta_type(DATAFRAME_GROUPBY_CHUNK_TYPE)
@dataslots
@dataclass
class DataFrameGroupByChunkMeta(_ChunkMeta):
    shape: Tuple[int] = None
    dtypes_value: DtypesValue = None
    index_value: IndexValue = None
    selection: List = None


@register_meta_type(SERIES_GROUPBY_CHUNK_TYPE)
@dataslots
@dataclass
class SeriesGroupByChunkMeta(_ChunkMeta):
    shape: Tuple[int] = None
    dtype: PandasDtypeType = None
    index_value: IndexValue = None
    selection: List = None


@register_meta_type(CATEGORICAL_CHUNK_TYPE)
@dataslots
@dataclass
class CategoricalChunkMeta(_ChunkMeta):
    shape: Tuple[int] = None
    dtype: PandasDtypeType = None
    categories_value: IndexValue = None


@register_meta_type(OBJECT_CHUNK_TYPE)
@dataslots
@dataclass
class ObjectChunkMeta(_ChunkMeta):
    pass


@register_meta_type(DATAFRAME_OR_SERIES_TYPE)
@dataslots
@dataclass
class DataFrameOrSeriesMeta(_TileableMeta):
    data_type: str = None
    data_params: Dict[str, Any] = None


@register_meta_type(DATAFRAME_OR_SERIES_CHUNK_TYPE)
@dataslots
@dataclass
class DataFrameOrSeriesChunkMeta(_ChunkMeta):
    collapse_axis: int = None
    data_type: str = None
    data_params: Dict[str, Any] = None
