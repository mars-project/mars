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
from typing import Any, List

from ....storage.base import StorageLevel
from ..core import DataInfo


class AbstractStorageAPI(ABC):
    @abstractmethod
    async def get(self,
                  data_key: str,
                  conditions: List = None,
                  error: str = 'raise') -> Any:
        """
        Get object by data key.

        Parameters
        ----------
        data_key: str
            date key to get.

        conditions: List
            Index conditions to pushdown

        error: str
            raise or ignore

        Returns
        -------
            object
        """

    @abstractmethod
    async def put(self, data_key: str,
                  obj: object,
                  level: StorageLevel = StorageLevel.MEMORY) -> DataInfo:
        """
        Put object into storage.

        Parameters
        ----------
        data_key: str
            data key to put.
        obj: object
            object to put.
        level: StorageLevel
            the storage level to put into, MEMORY as default

        Returns
        -------
        object information: ObjectInfo
            the put object information
        """
