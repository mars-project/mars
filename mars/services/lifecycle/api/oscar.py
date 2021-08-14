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

from typing import Dict, List, Union

from .... import oscar as mo
from ....lib.aio import alru_cache
from ..supervisor.tracker import LifecycleTrackerActor
from .core import AbstractLifecycleAPI


class LifecycleAPI(AbstractLifecycleAPI):
    def __init__(self,
                 session_id: str,
                 lifecycle_tracker_ref: Union[LifecycleTrackerActor, mo.ActorRef]):
        self._session_id = session_id
        self._lifecycle_tracker_ref = lifecycle_tracker_ref

    @classmethod
    @alru_cache(cache_exceptions=False)
    async def create(cls,
                     session_id: str,
                     address: str) -> "LifecycleAPI":
        """
        Create Lifecycle API.

        Parameters
        ----------
        session_id : str
            Session ID.
        address : str
            Supervisor address.

        Returns
        -------
        lifecycle_api
            Lifecycle API.
        """
        lifecycle_tracker_ref = await mo.actor_ref(
            address, LifecycleTrackerActor.gen_uid(session_id))
        return LifecycleAPI(session_id, lifecycle_tracker_ref)

    @mo.extensible
    async def track(self,
                    tileable_key: str,
                    chunk_keys: List[str]):
        """
        Track tileable.

        Parameters
        ----------
        tileable_key : str
            Tileable key.
        chunk_keys : list
            List of chunk keys.
        """
        return await self._lifecycle_tracker_ref.track(tileable_key, chunk_keys)

    @track.batch
    async def batch_track(self, args_list, kwargs_list):
        tracks = []
        for args, kwargs in zip(args_list, kwargs_list):
            tracks.append(self._lifecycle_tracker_ref.track.delay(*args, **kwargs))
        return await self._lifecycle_tracker_ref.track.batch(*tracks)

    async def incref_tileables(self, tileable_keys: List[str]):
        """
        Incref tileables.

        Parameters
        ----------
        tileable_keys : list
             List of tileable keys.
        """
        return await self._lifecycle_tracker_ref.incref_tileables(tileable_keys)

    async def decref_tileables(self, tileable_keys: List[str]):
        """
        Decref tileables.

        Parameters
        ----------
        tileable_keys : list
            List of tileable keys.
        """
        return await self._lifecycle_tracker_ref.decref_tileables(tileable_keys)

    async def get_tileable_ref_counts(self, tileable_keys: List[str]) -> List[int]:
        """
        Get ref counts of tileables.

        Parameters
        ----------
        tileable_keys : list
            List of tileable keys.

        Returns
        -------
        ref_counts : list
            List of ref counts.
        """
        return await self._lifecycle_tracker_ref.get_tileable_ref_counts(tileable_keys)

    async def incref_chunks(self, chunk_keys: List[str]):
        """
        Incref chunks.

        Parameters
        ----------
        chunk_keys : list
            List of chunk keys.
        """
        return await self._lifecycle_tracker_ref.incref_chunks(chunk_keys)

    async def decref_chunks(self, chunk_keys: List[str]):
        """
        Decref chunks

        Parameters
        ----------
        chunk_keys : list
            List of chunk keys.
        """
        return await self._lifecycle_tracker_ref.decref_chunks(chunk_keys)

    async def get_chunk_ref_counts(self, chunk_keys: List[str]) -> List[int]:
        """
        Get ref counts of chunks.

        Parameters
        ----------
        chunk_keys : list
            List of chunk keys.

        Returns
        -------
        ref_counts : list
            List of ref counts.
        """
        return await self._lifecycle_tracker_ref.get_chunk_ref_counts(chunk_keys)

    async def get_all_chunk_ref_counts(self) -> Dict[str, int]:
        """
        Get all chunk keys' ref counts.

        Returns
        -------
        key_to_ref_counts: dict
        """
        return await self._lifecycle_tracker_ref.get_all_chunk_ref_counts()


class MockLifecycleAPI(LifecycleAPI):
    @classmethod
    async def create(cls,
                     session_id: str,
                     address: str) -> "LifecycleAPI":
        from mars.services.lifecycle.supervisor.service import LifecycleSupervisorService
        service = LifecycleSupervisorService({}, address)
        await service.create_session(session_id)
        return await super().create(session_id=session_id, address=address)
