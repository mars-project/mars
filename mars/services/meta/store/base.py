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

from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Any, Type

import numpy as np

from ....tensor.core import TensorOrder
from ....dataframe.core import IndexValue, DtypesValue


object_types = ('tensor', 'dataframe', 'series', 'index', 'object')


class AbstractMetaStore(ABC):
    name = None

    def __init__(self, session_id: str, **kw):
        self._session_id = session_id

    @abstractmethod
    async def set_tensor_meta(self,
                              object_id: str,
                              shape: Tuple[int] = None,
                              dtype: np.dtype = None,
                              order: TensorOrder = None,
                              nsplits: Tuple[Tuple[int]] = None,
                              physical_size: int = None,
                              extra: Dict = None):
        """
        Set tensor meta.

        Parameters
        ----------
        object_id : str
            Object id.
        shape : tuple
            Tensor Shape:
        dtype
            Tensor's dtype.
        order
            Tensor order, C or F.
        nsplits
            Tuple of tuple of ints.
        physical_size : int
            Raw physical size.
        extra
            Extra information.
        """

    @abstractmethod
    async def get_tensor_meta(self,
                              object_id: str,
                              fields: List[str] = None) -> Dict[str, Any]:
        """
        Get tensor's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        fields : field names.
            Fields to filter, if not specified, will get all fields.

        Returns
        -------
        meta : dict
            Tensor's meta.
        """

    @abstractmethod
    async def del_tensor_meta(self,
                              object_id: str):
        """
        Delete tensor's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        """

    @abstractmethod
    async def set_tensor_chunk_meta(self,
                                    object_id: str,
                                    shape: Tuple[int] = None,
                                    dtype: np.dtype = None,
                                    order: TensorOrder = None,
                                    index: Tuple[int] = None,
                                    physical_size: int = None,
                                    workers: List[str] = None,
                                    extra: Dict = None):
        """
        Set tensor chunk's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        shape : tuple
            Tensor chunk's shape.
        dtype
            Tensor chunk's dtype.
        order
            Tensor chunk's order, C or F.
        index : tuple
            Tensor chunk's index in which each number
            means the position on each axis.
        physical_size : int
            Raw physical size.
        workers : list
            Workers which owns the data.
        extra : dict
            Extra information.
        """

    @abstractmethod
    async def get_tensor_chunk_meta(self,
                                    object_id: str,
                                    fields: List[str] = None) -> Dict[str, Any]:
        """
        Get tensor chunk's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        fields : list
            Fields to filter, if not specified, will get all fields.

        Returns
        -------
        meta : dict
            Tensor chunk's meta.
        """

    @abstractmethod
    async def del_tensor_chunk_meta(self,
                                    object_id: str):
        """
        Delete tensor chunk's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        """

    @abstractmethod
    async def set_dataframe_meta(self,
                                 object_id: str,
                                 shape: Tuple[int] = None,
                                 dtypes_value: DtypesValue = None,
                                 index_value: IndexValue = None,
                                 nsplits: Tuple[Tuple[int]] = None,
                                 physical_size: int = None,
                                 extra: Dict = None):
        """
        Set DataFrame's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        shape : tuple
            DataFrame's shape.
        dtypes_value
            DataFrame's types.
        index_value
            DataFrame's index.
        nsplits
            Tuple of tuple of ints.
        physical_size: int
            Raw physical size.
        extra : dict
            Extra information.
        """

    @abstractmethod
    async def get_dataframe_meta(self,
                                 object_id: str,
                                 fields: List[str] = None) -> Dict[str, Any]:
        """
        Get DataFrame's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        fields : list
            Fields to filter, if not specified, will get all fields.

        Returns
        -------
        meta : dict
            Tensor chunk's meta.
        """

    @abstractmethod
    async def del_dataframe_meta(self,
                                 object_id: str):
        """
        Delete DataFrame's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        """

    @abstractmethod
    async def set_dataframe_chunk_meta(self,
                                       object_id: str,
                                       shape: Tuple[int] = None,
                                       dtypes_value: DtypesValue = None,
                                       index_value: IndexValue = None,
                                       index: Tuple[int] = None,
                                       physical_size: int = None,
                                       workers: List[str] = None,
                                       extra: Dict = None):
        """
        Set DataFrame chunk's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        shape : tuple
            DataFrame chunk's shape.
        dtypes_value
            DataFrame chunk's dtypes value.
        index_value
            DataFrame chunk's index value.
        index : tuple
            DataFrame chunk's index in which each number
            means the position on each axis.
        physical_size : int
            Raw physical size.
        workers : list
            Workers which owns the data.
        extra : dict
            Extra information.
        """

    @abstractmethod
    async def get_dataframe_chunk_meta(self,
                                       object_id: str,
                                       fields: List[str] = None) -> Dict[str, Any]:
        """
        Get DataFrame chunk's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        fields : list
            Fields to filter, if not specified, will get all fields.

        Returns
        -------
        meta : dict
            DataFrame chunk's meta.
        """

    @abstractmethod
    async def del_dataframe_chunk_meta(self,
                                       object_id: str):
        """
        Delete DataFrame chunk's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        """

    @abstractmethod
    async def set_series_meta(self,
                              object_id: str,
                              shape: Tuple[int] = None,
                              dtype: np.dtype = None,
                              index_value: IndexValue = None,
                              name: Any = None,
                              nsplits: Tuple[Tuple[int]] = None,
                              physical_size: int = None,
                              extra: Dict = None):
        """
        Set Series' meta.

        Parameters
        ----------
        object_id : str
            Object id.
        shape : tuple
            Series' shape.
        dtype
            Series' dtype
        index_value
            Series' index value.
        name
            Series' name
        nsplits
            Tuple of tuple of ints.
        physical_size
            Raw physical size.
        extra: dict
            Extra information.
        """

    @abstractmethod
    async def get_series_meta(self,
                              object_id: str,
                              fields: List[str] = None) -> Dict[str, Any]:
        """
        Get series' meta.

        Parameters
        ----------
        object_id : str
            Object id.
        fields : list
            Fields to filter, if not specified, will get all fields.

        Returns
        -------
        meta : dict
            Series' meta.
        """

    @abstractmethod
    async def del_series_meta(self,
                              object_id: str):
        """
        Delete series' meta.

        Parameters
        ----------
        object_id : str
            Object id.
        """

    @abstractmethod
    async def set_series_chunk_meta(self,
                                    object_id: str,
                                    shape: Tuple[int] = None,
                                    dtype: np.dtype = None,
                                    index_value: IndexValue = None,
                                    name: Any = None,
                                    index: Tuple[int] = None,
                                    physical_size: int = None,
                                    workers: List[str] = None,
                                    extra: Dict = None):
        """
        Set series chunk's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        shape : tuple
            Series chunk's shape.
        dtype
            Series chunk's dtype.
        index_value
            Series chunk's index value.
        name
            Name of series chunk.
        index : tuple
            Series chunk's index in which each number
            means the position on each axis.
        physical_size : int
            Raw physical size.
        workers : list
            Workers which owns the data.
        extra : dict
            Extra information.
        """

    @abstractmethod
    async def get_series_chunk_meta(self,
                                    object_id: str,
                                    fields: List[str] = None) -> Dict[str, Any]:
        """
        Get series chunk's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        fields : list
            Fields to filter, if not specified, will get all fields.

        Returns
        -------
        meta : dict
            Series chunk's meta.
        """

    @abstractmethod
    async def del_series_chunk_meta(self,
                                    object_id: str):
        """
        Delete series chunk's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        """

    @abstractmethod
    async def set_index_meta(self,
                             object_id: str,
                             shape: Tuple[int] = None,
                             dtype: np.dtype = None,
                             index_value: IndexValue = None,
                             name: Any = None,
                             nsplits: Tuple[Tuple[int]] = None,
                             physical_size: int = None,
                             extra: Dict = None):
        """
        Set index meta.

        Parameters
        ----------
        object_id : str
            Object id.
        shape : tuple
            Index's shape.
        dtype
            Index's dtype.
        index_value
            Index's index value.
        name
            Index's name.
        nsplits
            Tuple of tuple of ints.
        physical_size: int
            Raw physical size.
        extra: dict
            Extra information.
        """

    @abstractmethod
    async def get_index_meta(self,
                             object_id: str,
                             fields: List[str] = None) -> Dict[str, Any]:
        """
        Get index's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        fields : list
            Fields to filter, if not specified, will get all fields.

        Returns
        -------
        meta : dict
            Index's meta.
        """

    @abstractmethod
    async def del_index_meta(self,
                             object_id: str):
        """
        Delete index's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        """

    @abstractmethod
    async def set_index_chunk_meta(self,
                                   object_id: str,
                                   shape: Tuple[int] = None,
                                   dtype: np.dtype = None,
                                   index_value: IndexValue = None,
                                   name: Any = None,
                                   index: Tuple[int] = None,
                                   physical_size: int = None,
                                   workers: List[str] = None,
                                   extra: Dict = None):
        """
        Set index chunk's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        shape : tuple
            Index chunk's shape.
        dtype
            Index chunk's dtype
        index_value
            Index chunk's index_value
        name
            Index chunk's name.
        index : tuple
            Index chunk's index in which each number
            means the position on each axis.
        physical_size : int
            Raw physical size.
        workers : list
            Workers which owns the data.
        extra: dict
            Extra information.
        """

    @abstractmethod
    async def get_index_chunk_meta(self,
                                   object_id: str,
                                   fields: List[str] = None) -> Dict[str, Any]:
        """
        Get index chunk's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        fields : list
            Fields to filter, if not specified, will get all fields.

        Returns
        -------
        meta : dict
            Index chunk's meta.
        """

    @abstractmethod
    async def del_index_chunk_meta(self,
                                   object_id: str):
        """
        Delete index chunk's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        """

    @abstractmethod
    async def set_object_meta(self,
                              object_id: str = None,
                              physical_size: int = None,
                              nsplits: Tuple[Tuple[int]] = None,
                              extra: Dict = None):
        """
        Set object's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        nsplits
            Tuple of tuple of ints.
        physical_size : int
            Raw physical size.
        extra : dict
            Extra information.
        """

    @abstractmethod
    async def get_object_meta(self,
                              object_id: str,
                              fields: List[str] = None) -> Dict[str, Any]:
        """
        Get object's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        fields : list
            Fields to filter, if not specified, will get all fields.

        Returns
        -------
        meta : dict
            Object's meta.
        """

    @abstractmethod
    async def del_object_meta(self,
                              object_id: str):
        """
        Delete object meta.

        Parameters
        ----------
        object_id : str
            Object id.
        """

    @abstractmethod
    async def set_object_chunk_meta(self,
                                    object_id: str,
                                    index: Tuple[int] = None,
                                    physical_size: int = None,
                                    workers: List[str] = None,
                                    extra: Dict = None):
        """
        Set object chunk's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        index : tuple
            Object chunk's index in which each number
            means the position on each axis.
        physical_size : int
            Raw physical size.
        workers : list
            Workers which owns the data.
        extra: dict
            Extra information.
        """

    @abstractmethod
    async def get_object_chunk_meta(self,
                                    object_id: str,
                                    fields: List[str] = None) -> Dict[str, Any]:
        """
        Get object chunk's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        fields : list
            Fields to filter, if not specified, will get all fields.

        Returns
        -------
        meta : dict
            Index chunk's meta.
        """

    @abstractmethod
    async def del_object_chunk_meta(self,
                                    object_id: str):
        """
        Delete object chunk's meta.

        Parameters
        ----------
        object_id : str
            Object id.
        """


_meta_store_types: Dict[str, Type[AbstractMetaStore]] = dict()


def register_meta_store(meta_store: Type[AbstractMetaStore]):
    _meta_store_types[meta_store.name] = meta_store
    return meta_store


def get_meta_store(meta_store_name: str) -> Type[AbstractMetaStore]:
    return _meta_store_types[meta_store_name]
