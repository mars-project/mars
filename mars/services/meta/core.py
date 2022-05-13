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
from typing import Any, Dict, List, Tuple, Union, Type

import numpy as np
import pandas as pd

from ...typing import BandType
from ...utils import dataslots, TypeDispatcher

PandasDtypeType = Union[np.dtype, pd.api.extensions.ExtensionDtype]

_meta_class_dispatcher = TypeDispatcher()


def register_meta_type(object_types: Tuple):
    def _call(meta_type: Type["_CommonMeta"]):
        _meta_class_dispatcher.register(object_types, meta_type)
        return meta_type

    return _call


def get_meta_type(object_type: Type) -> Type["_CommonMeta"]:
    return _meta_class_dispatcher.get_handler(object_type)


@dataslots
@dataclass
class _CommonMeta:
    """
    Class for common meta, for both tileable and chunk, or DataFrame, tensor etc.
    """

    object_id: str
    name: Any = None
    memory_size: int = None  # size in memory
    store_size: int = None  # size that stored in storage
    extra: Dict = None

    def merge_from(self, value: "_CommonMeta"):
        return self


@dataslots
@dataclass
class _TileableMeta(_CommonMeta):
    nsplits: Tuple[Tuple[int]] = None


@dataslots
@dataclass
class _ChunkMeta(_CommonMeta):
    index: Tuple[int] = None
    bands: List[BandType] = None
    # needed by ray ownership to keep object alive when worker died.
    object_refs: List[Any] = None

    def merge_from(self, value: "_ChunkMeta"):
        if value.bands:
            self.bands = list(set(self.bands) | set(value.bands))
        if value.object_refs:
            self.object_refs = list(set(self.object_refs) | set(value.object_refs))
        return self
