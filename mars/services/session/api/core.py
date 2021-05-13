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
from typing import Union


class AbstractSessionAPI(ABC):
    @abstractmethod
    async def create_session(self, session_id: str) -> str:
        """
        Create session and return address.

        Parameters
        ----------
        session_id : str
            Session ID

        Returns
        -------
        address : str
            Session address.
        """

    @abstractmethod
    async def delete_session(self, session_id: str):
        """
        Delete session.

        Parameters
        ----------
        session_id : str
            Session ID.
        """

    @abstractmethod
    async def get_last_idle_time(self, session_id: Union[str, None] = None) -> Union[float, None]:
        """
        Get session last idle time.

        Parameters
        ----------
        session_id : str, None
            Session ID. None for all sessions.

        Returns
        -------
        last_idle_time: str
            The last idle time if the session(s) is idle else None.
        """
