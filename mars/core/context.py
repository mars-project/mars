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

import os
from abc import ABC, abstractmethod
from typing import List, Dict

from ..typing import BandType, SessionType


class Context(ABC):
    """
    Context that providing API that can be
    used inside `tile` and `execute`.
    """
    prev = None
    current = None

    def __init__(self,
                 session_id: str = None,
                 supervisor_address: str = None,
                 current_address: str = None,
                 band: BandType = None):
        if session_id is None:
            # try to get session id from environment
            session_id = os.environ.get('MARS_SESSION_ID')
            if session_id is None:
                raise ValueError('session_id should be provided '
                                 'to create a context')
        if supervisor_address is None:
            # try to get supervisor address from environment
            supervisor_address = os.environ.get('MARS_SUPERVISOR_ADDRESS')
            if supervisor_address is None:
                raise ValueError('supervisor_address should be provided '
                                 'to create a context')

        self.session_id = session_id
        self.supervisor_address = supervisor_address
        self.current_address = current_address
        self.band = band

    @abstractmethod
    def get_current_session(self) -> SessionType:
        """
        Get current session

        Returns
        -------
        session
        """

    @abstractmethod
    def get_supervisor_addresses(self) -> List[str]:
        """
        Get supervisor addresses.

        Returns
        -------
        supervisor_addresses : list
        """

    @abstractmethod
    def get_worker_addresses(self) -> List[str]:
        """
        Get worker addresses.

        Returns
        -------
        worker_addresses : list
        """

    @abstractmethod
    def get_total_n_cpu(self) -> int:
        """
        Get number of cpus.

        Returns
        -------
        number_of_cpu: int
        """

    @abstractmethod
    def get_chunks_result(self,
                          data_keys: List[str]) -> List:
        """
        Get result of chunks.

        Parameters
        ----------
        data_keys : list
            Data keys.

        Returns
        -------
        results : list
            Result of chunks
        """

    @abstractmethod
    def get_chunks_meta(self,
                        data_keys: List[str],
                        fields: List[str] = None,
                        error='raise') -> List[Dict]:
        """
        Get meta of chunks.

        Parameters
        ----------
        data_keys : list
            Data keys.
        fields : list
            Fields to filter.
        error : str
            raise, ignore

        Returns
        -------
        meta_list : list
            Meta list.
        """

    @abstractmethod
    def create_remote_object(self,
                             name: str,
                             object_cls, *args, **kwargs):
        """
        Create remote object.

        Parameters
        ----------
        name : str
            Object name.
        object_cls
            Object class.
        args
        kwargs

        Returns
        -------
        ref
        """

    @abstractmethod
    def get_remote_object(self, name: str):
        """
        Get remote object

        Parameters
        ----------
        name : str
            Object name.

        Returns
        -------
        ref
        """

    @abstractmethod
    def destroy_remote_object(self,
                              name: str):
        """
        Destroy remote object.

        Parameters
        ----------
        name : str
            Object name.
        """

    @abstractmethod
    def register_custom_log_path(self,
                                 session_id: str,
                                 tileable_op_key: str,
                                 chunk_op_key: str,
                                 worker_address: str,
                                 log_path: str):
        """
        Register custom log path.

        Parameters
        ----------
        session_id : str
            Session ID.
        tileable_op_key : str
            Key of tileable's op.
        chunk_op_key : str
            Kye of chunk's op.
        worker_address : str
            Worker address.
        log_path : str
            Log path.
        """

    def new_custom_log_dir(self) -> str:
        """
        New custom log dir.

        Returns
        -------
        custom_log_dir : str
            Custom log dir.
        """

    def set_running_operand_key(self, session_id: str, op_key: str):
        """
        Set key of running operand.

        Parameters
        ----------
        session_id : str
        op_key : str
        """

    def set_progress(self, progress: float):
        """
        Set progress of running operand.

        Parameters
        ----------
        progress : float
        """

    def __enter__(self):
        Context.prev = Context.current
        Context.current = self

    def __exit__(self, *_):
        Context.current = Context.prev
        Context.prev = None


def set_context(context: Context):
    Context.current = context


def get_context() -> Context:
    return Context.current
