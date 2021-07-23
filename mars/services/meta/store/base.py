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

from abc import ABC, abstractmethod
from typing import Dict, List, Type

from ....typing import BandType
from ..core import _CommonMeta


class AbstractMetaStore(ABC):
    name = None

    def __init__(self, session_id: str, **kw):
        self._session_id = session_id

    @classmethod
    @abstractmethod
    async def create(cls, config) -> Dict:
        """
        Create a meta store. Do some initialization work.
        For instance, for database backend,
        db files including tables may be created first.
        This should be done when service starting.

        Parameters
        ----------
        config : dict
            config.

        Returns
        -------
        kwargs : dict
            kwargs to create a meta store.
        """

    @abstractmethod
    async def set_meta(self,
                       object_id: str,
                       meta: _CommonMeta):
        """
        Set meta.

        Parameters
        ----------
        object_id : str
            Object ID.
        meta : _CommonMeta
            Meta.
        """

    @abstractmethod
    async def get_meta(self,
                       object_id: str,
                       fields: List[str] = None,
                       error='raise') -> Dict:
        """
        Get meta.

        Parameters
        ----------
        object_id : str
            Object ID.
        fields : list
            Fields to filter, if not provided, get all fields.
        error : str
            'raise' or 'ignore'

        Returns
        -------
        meta: dict
            Meta.
        """

    @abstractmethod
    async def del_meta(self,
                       object_id: str):
        """
        Delete meta.

        Parameters
        ----------
        object_id : str
            Object ID.
        """

    @abstractmethod
    async def add_chunk_bands(self,
                              object_id: str,
                              bands: List[BandType]):
        """
        Add band to chunk.

        Parameters
        ----------
        object_id : str
            Object ID.
        bands : List[BandType]
            Band of chunk to add, shall be tuple of (worker, band).
        """


_meta_store_types: Dict[str, Type[AbstractMetaStore]] = dict()


def register_meta_store(meta_store: Type[AbstractMetaStore]):
    _meta_store_types[meta_store.name] = meta_store
    return meta_store


def get_meta_store(meta_store_name: str) -> Type[AbstractMetaStore]:
    return _meta_store_types[meta_store_name]
