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
from typing import Tuple, Union

import numpy as np

from ..core import MutableTensorInfo


class AbstractMutableAPI(ABC):
    @abstractmethod
    async def create_mutable_tensor(
        self,
        shape: tuple,
        dtype: Union[np.dtype, str],
        name: str = None,
        default_value: Union[int, float] = 0,
        chunk_size: Union[int, Tuple] = None,
    ) -> MutableTensorInfo:
        """
        Create a mutable tensor.

        Parameters
        ----------
        shape: tuple
            Shape of the mutable tensor.

        dtype: np.dtype or str
            Data type of the mutable tensor.

        chunk_size: int or tuple
            Chunk size of the mutable tensor.

        name: str, optional
            Name of the mutable tensor, a random name will be used if not specified.

        default_value: optional
            Default value of the mutable tensor. Default is 0.

        Returns
        -------
            MutableTensorInfo
        """

    @abstractmethod
    async def get_mutable_tensor(self, name: str) -> MutableTensorInfo:
        """
        Get the mutable tensor by name.

        Parameters
        ----------
        name: str
            Name of the mutable tensor to get.

        Returns
        -------
            MutableTensorInfo
        """

    @abstractmethod
    async def seal_mutable_tensor(self, name: str, timestamp=None):
        """
        Seal the mutable tensor by name.

        Parameters
        ----------
        name: str
            Name of the mutable tensor to seal.

        timestamp: optional
            Operations that happened before timestamp will be sealed, and later ones will be discard.

        Returns
        -------
            object
        """

    @abstractmethod
    async def read(self, name: str, index: object, timestamp=None):
        """
        Read value from mutable tensor.

        Parameters
        ----------
        name: str
            Name of mutable tensor to read.

        index:
            Index to read from the tensor.

        timestamp: optional
            Timestamp to read value that happened before then.
        """

    @abstractmethod
    async def write(self, name: str, index: object, value: object, timestamp=None):
        """
        Write value to mutable tensor.

        Parameters
        ----------
        name: str
            Name of the mutable tensor to write.

        index:
            Index to write to the tensor.

        value:
            The value that will be filled into the mutable tensor according to `index`.

        timestamp: optional
            Timestamp to associated with the newly touched value.
        """
