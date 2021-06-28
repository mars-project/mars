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
from typing import Dict, List, Optional


class AbstractMetaAPI(ABC):
    @abstractmethod
    async def get_chunk_meta(self,
                             object_id: str,
                             fields: List[str] = None,
                             error: str = 'raise') -> Optional[Dict]:
        """
        Get chunk meta

        Parameters
        ----------
        object_id
            Object ID
        fields
            Fields to obtain
        error
            Way to handle errors, 'raise' by default
        Returns
        -------
            Dict with fields as keys
        """
